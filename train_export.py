"""Leakage-aware pre-trip fare category training and export.

Run from the project directory: python train_export.py
Only the provided TLC January 2023 Parquet data is used.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'yellow_tripdata_2023-01.parquet'
OUT = ROOT / 'models'
SEED = 42
# The original notebooks use actual trip duration, which is not a pre-trip feature.
# Estimated distance is an explicitly estimated pre-trip quantity, not actual fare.
FEATURES = ['PULocationID', 'DOLocationID', 'day_of_week', 'passenger_count',
            'trip_distance', 'pickup_hour', 'is_weekend']
CATEGORICAL = ['PULocationID', 'DOLocationID', 'day_of_week']
NUMERIC = ['passenger_count', 'trip_distance', 'pickup_hour', 'is_weekend']
LABELS = ['Low', 'Medium', 'High']


def load_data():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing taxi source dataset: {SOURCE}')
    df = pd.read_parquet(SOURCE, columns=[
        'tpep_pickup_datetime', 'trip_distance', 'fare_amount',
        'passenger_count', 'PULocationID', 'DOLocationID'
    ])
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.drop_duplicates().dropna(subset=['tpep_pickup_datetime', 'fare_amount',
                                              'trip_distance', 'PULocationID', 'DOLocationID'])
    df = df.loc[df['trip_distance'].between(0.1, 100, inclusive='neither') &
                df['fare_amount'].between(2.5, 300, inclusive='neither')].copy()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    df['is_weekend'] = df['tpep_pickup_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    for key in ['PULocationID', 'DOLocationID']:
        df[key] = df[key].astype('int64').astype(str)
    df['passenger_count'] = pd.to_numeric(df['passenger_count'], errors='coerce')
    df.loc[~df['passenger_count'].between(1, 6), 'passenger_count'] = np.nan
    if len(df) < 1000:
        raise ValueError('Insufficient valid source records.')
    return df


def pipeline(hidden):
    preprocess = ColumnTransformer([
        ('numerical', Pipeline([('impute', SimpleImputer(strategy='median')),
                                ('scale', StandardScaler())]), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ], sparse_threshold=0)
    return Pipeline([
        ('preprocess', preprocess),
        ('pca', PCA(n_components=24, svd_solver='randomized', random_state=SEED)),
        ('classifier', MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                                     max_iter=120, early_stopping=True,
                                     n_iter_no_change=8, random_state=SEED))
    ])


def main():
    df = load_data()
    # Reproducible capped subsample for practical Windows laptop training; no test mixing.
    if len(df) > 40000:
        df = df.sample(n=40000, random_state=SEED)
    X = df[FEATURES].copy()
    y_fare = df['fare_amount'].copy()
    x_train, x_test, fare_train, fare_test = train_test_split(
        X, y_fare, test_size=0.20, random_state=SEED)
    # Fit target category thresholds on training targets only.
    q1, q2 = map(float, fare_train.quantile([0.33, 0.67]))
    if q1 >= q2:
        raise ValueError('Fare quantiles not distinct, cannot construct 3 classes.')
    bins = [-np.inf, q1, q2, np.inf]
    y_train = pd.cut(fare_train, bins=bins, labels=LABELS).astype(str)
    y_test = pd.cut(fare_test, bins=bins, labels=LABELS).astype(str)
    # Hold out validation from training; never consult test set for model choice.
    xx_train, xx_val, yy_train, yy_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train)
    candidates = [(64,), (64, 32)]
    scores = []
    for hidden in candidates:
        model = pipeline(hidden)
        model.fit(xx_train, yy_train)
        p = model.predict(xx_val)
        scores.append({'hidden_layer_sizes': list(hidden),
                       'validation_weighted_f1': float(f1_score(yy_val, p, average='weighted')),
                       'validation_accuracy': float(accuracy_score(yy_val, p))})
        print('Validation:', scores[-1], flush=True)
    winner = max(scores, key=lambda row: row['validation_weighted_f1'])
    model = pipeline(tuple(winner['hidden_layer_sizes']))
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'weighted_precision': float(precision_score(y_test, predictions, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(y_test, predictions, average='weighted', zero_division=0)),
        'weighted_f1': float(f1_score(y_test, predictions, average='weighted', zero_division=0)),
        'roc_auc_ovr_weighted': float(roc_auc_score(y_test, probabilities, labels=model.classes_, multi_class='ovr', average='weighted')),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / 'classification_pipeline.joblib', compress=3)
    metadata = {
        'purpose': 'Pre-trip fare category estimate; estimated distance input is required',
        'model': 'MLPClassifier', 'selection': 'validation weighted F1 on pre-trip inputs',
        'source': str(SOURCE.relative_to(ROOT)), 'seed': SEED,
        'sample_records': len(df), 'train_records': len(x_train), 'test_records': len(x_test),
        'features': FEATURES, 'labels': LABELS,
        'fare_thresholds_usd': {'low_max': q1, 'medium_max': q2},
        'tuning': scores, 'selected_hyperparameters': winner,
        'heldout_test_metrics': metrics,
        'classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0),
        'probability_note': 'Uncalibrated model probabilities, not validated confidence intervals.',
        'evaluation_note': 'Not directly comparable to historical notebook metrics: trip duration excluded and preprocessing refitted.'
    }
    (OUT / 'classification_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Verify artifact after serialization.
    reloaded = joblib.load(OUT / 'classification_pipeline.joblib')
    assert reloaded.predict(x_test.iloc[:3]).tolist() == predictions[:3].tolist()
    print('\nSaved verified model:', OUT / 'classification_pipeline.joblib')
    print('Held-out test metrics:', json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
