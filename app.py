import json, zipfile, io
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='NYC Taxi | ML Capstone',page_icon='🚕',layout='wide')
st.title('🚕 NYC Taxi · Machine Learning Capstone')
st.caption('Trip fare intelligence · Classification • Clustering • Regression')
root=Path(__file__).parent
archive=root/'notebooks.zip'
if not archive.exists():
 st.error('Place the provided notebooks.zip next to app.py.'); st.stop()

def output_text(out):
 if out.get('output_type')=='stream': return ''.join(out.get('text',[]))
 return ''.join(out.get('data',{}).get('text/plain',[]))

def load(name):
 with zipfile.ZipFile(archive) as z: return json.loads(z.read(f'notebooks/{name}.ipynb'))

def figures(cell):
 for out in cell.get('outputs',[]):
  data=out.get('data',{})
  if 'image/png' in data:
   import base64
   yield base64.b64decode(''.join(data['image/png']))

tracks={'Classification Part A':'Classification_A','Classification Part B':'Classification_B','Clustering':'Clustering','Regression':'Regression'}
page=st.sidebar.radio('Explore', ['Overview','Classification Part A','Classification Part B','Clustering','Regression','Live prediction'])
if page=='Overview':
 st.info('This dashboard displays the actual saved notebook outputs. No performance numbers are fabricated.')
 a,b,c=st.columns(3)
 a.metric('Classification models','10'); b.metric('Clustering algorithms','2'); c.metric('Regression models','10')
 st.caption('Explore evaluated models, visualizations, and an optional pre-trip fare category estimator.')
elif page=='Live prediction':
 st.subheader('Pre-trip fare category prediction')
 st.caption('Enter route and pickup details, including an **estimated** travel distance. Actual trip duration, fare and payment details are never inputs.')
 model_path=root/'models'/'classification_pipeline.joblib'
 metadata_path=root/'models'/'classification_metadata.json'
 if model_path.exists() and metadata_path.exists():
  import joblib
  @st.cache_resource
  def model_load(path, modified):
   return joblib.load(path)
  try:
   artifact=model_load(str(model_path),model_path.stat().st_mtime)
   meta=json.loads(metadata_path.read_text(encoding='utf-8'))
  except Exception as exc:
   st.error(f'Unable to load saved model: {exc}'); st.stop()
  if meta.get('features')!=list(artifact.feature_names_in_):
   st.error('Model input schema does not match metadata. Re-export the pipeline.'); st.stop()
  st.success('Trained model loaded and ready')
  with st.expander('Model provenance & held-out evaluation'):
   st.write('Model:',meta.get('model'))
   st.write('Training records:',meta.get('train_records'))
   st.write('Separate test records:',meta.get('test_records'))
   if meta.get('heldout_test_metrics'):
    st.json(meta['heldout_test_metrics'])
   st.caption(meta.get('evaluation_note',''))
  with st.form('trip'):
   a,b=st.columns(2)
   with a:
    pickup=st.number_input('NYC TLC pickup zone ID',min_value=1,max_value=265,value=161,step=1)
    dropoff=st.number_input('NYC TLC drop-off zone ID',min_value=1,max_value=265,value=236,step=1)
    distance=st.number_input('Estimated route distance (miles)',min_value=0.11,max_value=99.99,value=3.0,step=0.1)
   with b:
    passengers=st.number_input('Passengers',min_value=1,max_value=6,value=1,step=1)
    hour=st.slider('Pickup hour (24-hour clock)',0,23,12)
    day=st.selectbox('Pickup weekday',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
   submitted=st.form_submit_button('Predict fare category',type='primary',use_container_width=True)
  if submitted:
   x=pd.DataFrame([{'PULocationID':str(pickup),'DOLocationID':str(dropoff),
      'day_of_week':day,'passenger_count':int(passengers),
      'trip_distance':float(distance),'pickup_hour':int(hour),
      'is_weekend':int(day in ('Saturday','Sunday'))}],columns=meta['features'])
   try:
    prediction=str(artifact.predict(x)[0])
    if prediction not in meta['labels']:
     raise ValueError('Unknown category returned from model.')
    st.success(f'Predicted fare category: {prediction}')
    thresholds=meta.get('fare_thresholds_usd',{})
    if thresholds:
     st.caption(f"Categories defined from training fares: Low ≤ ${thresholds['low_max']:.2f}; Medium ≤ ${thresholds['medium_max']:.2f}; High above that.")
    if hasattr(artifact,'predict_proba'):
     probs=artifact.predict_proba(x)[0]
     data=pd.DataFrame({'Fare category':artifact.classes_,'Uncalibrated model probability':probs})
     st.bar_chart(data.set_index('Fare category'))
     st.caption('These are uncalibrated model scores, not guaranteed or independently calibrated confidence levels.')
   except Exception as exc:
    st.error(f'Could not make a prediction: {exc}')
 else:
  st.warning('Exported model missing. Run **python train_export.py** in the project folder first.')
  st.code('python train_export.py',language='powershell')
  st.caption('Requires the supplied Jan 2023 TLC Parquet file in data/. No dummy predictions are displayed.')
else:
 notebook=load(tracks[page]); cells=notebook['cells']
 st.subheader(page)
 st.caption(f"{sum(c['cell_type']=='code' for c in cells)} code cells · Saved notebook results")
 search=st.text_input('Filter sections',placeholder='e.g. confusion, ROC, elbow, PCA, silhouette')
 show_code=st.toggle('Show source code',False)
 count=0
 for idx,cell in enumerate(cells):
  src=''.join(cell['source'])
  if search and search.lower() not in src.lower(): continue
  if cell['cell_type']=='markdown':
   if not search: st.markdown(src)
   continue
  outs=cell.get('outputs',[])
  if not outs: continue
  count+=1
  with st.expander(f'Cell {idx} · {src.strip().splitlines()[0][:95] if src.strip() else "Output"}',expanded=bool(search)):
   if show_code: st.code(src,language='python')
   for out in outs:
    txt=output_text(out)
    if txt: st.text(txt[:14000])
    for pic in figures({'outputs':[out]}): st.image(pic,use_container_width=True)
 if not count: st.info('No matching saved output cells.')
