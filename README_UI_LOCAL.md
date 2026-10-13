# NYC Taxi Machine Learning Dashboard

An interactive Streamlit viewer for the **actual notebook outputs**, with a separate reproducible **pre-trip fare category** model training/export workflow.

## Windows CMD setup

```cmd
cd /d D:\ml_review2_ui
py -3.12 -m pip install -r requirements.txt
py -3.12 train_export.py
py -3.12 -m streamlit run app.py
```

If `py -3.12` isn't registered, use `python` instead of `py -3.12`. Training can take several minutes depending on your CPU.

## Files

- `app.py`: Streamlit dashboard, preserving all original notebook-result pages.
- `train_export.py`: real data training/validation/testing + serialized preprocessing/PCA/classifier pipeline.
- `data/yellow_tripdata_2023-01.parquet`: original taxi dataset from the supplied notebooks archive.
- `notebooks.zip`: original user notebooks and preserved outputs (untouched).
- `models/classification_pipeline.joblib`: created **locally by `train_export.py`**; not fabricated or bundled.
- `models/classification_metadata.json`: training thresholds, selected configuration, test metrics, and feature schema; generated alongside artifact.

## Scientific caveats

The existing Classification B notebook reports **MLP** as its leading Part B model by accuracy (0.9564; weighted F1 0.9563; multiclass ROC-AUC 0.9949). Those are *saved notebook results*, not results of this newly trained pre-trip model.

The historical classification notebook includes the *actual trip duration* as a model input; that is unavailable before pickup, and therefore invalid for a pre-trip claim. The new model excludes actual duration and post-trip fares/charges/tips/payment information. It uses **estimated** trip distance, pickup/dropoff TLC zone IDs, pickup hour/weekday/weekend, and passenger count only. Estimated route distance must be available from the user or a route planner; entering actual trip distance after the ride changes the interpretation of the demo.

Fare classes are Low, Medium, High, determined using the 33rd and 67th percentile boundaries of the **training fares only**. Features are fitted entirely inside an sklearn `Pipeline` on the relevant training partition, including during tuning. A held-out 20% test partition is used only once after selection. Training is capped at a reproducible random subsample of 40,000 rows for laptop practicality; results only support that sampled dataset. The MLP is selected because the original saved outputs ranked it highest among Part B models; the script tests two MLP configurations on a validation set and does **not** retrain all ten algorithms. It does not claim that this is the best model in a new pre-trip all-ten comparison.

The 24-component PCA is fitted to encoded, scaled training features. Predictive probabilities are **not calibrated**; UI shows them as raw, uncalibrated model probabilities, not calibrated confidence. Verify new held-out metrics from the generated JSON rather than quoting historical metrics.

The exported `joblib` file is Python-pickle-based and should only be loaded from trusted local files. Use the same installed versions for training and serving whenever possible.

## Review 2 follow-ups

The notebooks themselves have not been modified. For a single fair 10-model classification table, rerun both Part A and Part B with a shared train/test split, pre-trip-safe features if making a pre-trip claim, and train-only preprocessing and tuning. This independent live model does not retroactively validate the old scores. Provide genuine student-authored insights for presentations; the capstone rules prohibit AI-generated analysis/interpretation and require acknowledging AI scaffolding assistance in the README.
