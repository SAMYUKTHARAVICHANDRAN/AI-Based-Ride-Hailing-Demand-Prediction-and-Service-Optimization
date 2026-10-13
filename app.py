"""NYC Taxi ML · interactive dashboard showing authentic notebook outputs."""
import base64
import html
import io
import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / 'notebooks'
NOTEBOOKS = {
    'Classification Part A': ('Classification_A', 'Foundational classifiers', '01'),
    'Classification Part B': ('Classification_B', 'Ensemble and neural models', '02'),
    'Clustering': ('Clustering', 'Unsupervised segmentation', '03'),
    'Regression': ('Regression', 'Fare estimation benchmarks', '04'),
}
PAGES = ['Overview', *NOTEBOOKS.keys(), 'Live Prediction']
st.set_page_config(page_title='NYC Taxi | Intelligence Studio', page_icon='🚕', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme:dark;}
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 83% 0%,#163146 0,transparent 34%), #08121e; color:#e9f1f8;}
[data-testid="stSidebar"] {background:#0c1928;border-right:1px solid #223347;}
[data-testid="stSidebar"] * {color:#e3edf7;}
[data-testid="stHeader"] {background:rgba(8,18,30,.65);}
.block-container {padding-top:2rem;max-width:1400px;padding-bottom:5rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em!important;}
h1 {font-size:2.6rem!important;} h2 {font-size:1.6rem!important;} h3 {font-size:1.16rem!important;}
.stButton>button[kind="primary"], [data-testid="stFormSubmitButton"] button {border:0!important;background:linear-gradient(95deg,#28d9c1,#42a8e6)!important;color:#071726!important;font-weight:800!important;border-radius:12px!important;min-height:3rem;}
[data-testid="stMetric"] {background:#101f30;border:1px solid #253c50;border-radius:16px;padding:19px 22px;}
[data-testid="stMetricLabel"] {color:#8ba4ba!important;} [data-testid="stMetricValue"] {color:#eaf8ff!important;}
[data-testid="stExpander"] {border:1px solid #263c51!important;background:#0e1b2b!important;border-radius:14px!important;margin-bottom:12px;}
[data-testid="stTabs"] button {color:#a4bdcf;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#23394c;}
.hero {background:linear-gradient(110deg,rgba(19,66,85,.96),rgba(16,32,52,.97));border:1px solid #26536a;border-radius:24px;padding:34px 38px;margin-bottom:24px;position:relative;overflow:hidden;}
.hero:after {content:'✦';position:absolute;right:48px;top:10px;font-size:175px;color:rgba(66,216,202,.10);}
.eyebrow {font-size:.7rem;font-weight:800;letter-spacing:.20em;text-transform:uppercase;color:#55e0ce;margin-bottom:14px;}
.hero h1 {font-size:2.9rem!important;line-height:1.12;margin:0 0 13px;color:#f4fcff;max-width:790px;}
.hero p {font-size:1.04rem;color:#b9d0df;max-width:750px;line-height:1.7;margin:0;}
.chip {display:inline-block;margin-top:23px;margin-right:9px;background:rgba(39,215,190,.10);border:1px solid #2e706c;color:#8deee0;font-size:.78rem;border-radius:100px;padding:7px 12px;}
.section-head {display:flex;justify-content:space-between;align-items:center;margin:24px 0 12px;}
.section-head span {color:#6fe5d3;font-size:.75rem;font-weight:800;letter-spacing:.14em;}
.tile {background:linear-gradient(140deg,#14293b,#0d1e2d);border:1px solid #274051;border-radius:18px;padding:22px;min-height:154px;margin-bottom:16px;}
.tile .number {color:#55d9cb;font-weight:700;font-size:.79rem;letter-spacing:.12em;}
.tile h3 {margin:14px 0 5px;color:#f0f7fc;}
.tile p {color:#a3bdcd;font-size:.85rem;line-height:1.55;margin-bottom:0;}
.note {background:#102435;border:1px solid #2b4a60;border-left:3px solid #48d7c8;border-radius:12px;padding:15px 19px;color:#b7d5e3;font-size:.88rem;margin:14px 0 25px;}
.plot-card {background:#0e1f30;border:1px solid #243d51;border-radius:15px;padding:10px;margin-bottom:12px;}
.stImage img {border-radius:12px;}
.small-muted {color:#8aa8bc;font-size:.81rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {border-radius:10px;padding:7px;margin-bottom:3px;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1b3044;}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_notebook(name, notebook_mtime):
    path = NOTEBOOK_DIR / f'{name}.ipynb'
    return json.loads(path.read_text(encoding='utf-8'))


def notebook(name):
    path = NOTEBOOK_DIR / f'{name}.ipynb'
    return read_notebook(name, path.stat().st_mtime_ns)


def output_text(o):
    if o.get('output_type') == 'stream':
        return ''.join(o.get('text', []))
    d = o.get('data', {})
    return ''.join(d.get('text/plain', []))


def plot_outputs(cells):
    for ci, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        code = ''.join(cell.get('source', []))
        for out in cell.get('outputs', []):
            data = out.get('data', {})
            if 'image/png' in data:
                try:
                    yield (ci, code, base64.b64decode(''.join(data['image/png'])))
                except (ValueError, TypeError):
                    continue


def nice_title(code, i):
    lines = [s.strip('# -\t') for s in code.splitlines() if s.strip()]
    for s in lines[:8]:
        if s.lower().startswith(('plt.title(', 'ax.set_title(', 'sns.', 'fig.', 'print(')):
            continue
        if s.startswith('#') and len(s) > 3:
            return s.strip('# ').strip()[:100]
    return f'Notebook visualization · {i + 1:02d}'


def badge(text):
    return '<span class="chip">' + html.escape(text) + '</span>'


def hero(kicker, heading, sub, chips=()):
    st.markdown('<div class="hero"><div class="eyebrow">' + html.escape(kicker) + '</div><h1>' + html.escape(heading) + '</h1><p>' + html.escape(sub) + '</p>' + ''.join(map(badge, chips)) + '</div>', unsafe_allow_html=True)


def navigate_to(target):
    st.session_state['nav_page'] = target


with st.sidebar:
    st.markdown('### 🚕  TAXI / INTELLIGENCE')
    st.caption('NEW YORK CITY  ·  ML CAPSTONE')
    st.divider()
    page = st.radio('WORKSPACE', PAGES, label_visibility='collapsed', key='nav_page')
    st.divider()
    st.markdown('**MODEL LAB**')
    st.caption('Real saved notebook outputs. Live predictions require the exported trained pipeline.')
    st.caption('Dataset: NYC TLC Yellow Taxi · January 2023')

missing = [name for name, _, _ in NOTEBOOKS.values() if not (NOTEBOOK_DIR / f'{name}.ipynb').exists()]
if missing:
    st.error('Missing notebooks in notebooks/: ' + ', '.join(missing)); st.stop()

if page == 'Overview':
    hero('Data science / portfolio workspace', 'NYC Taxi Intelligence Studio',
         'Explore the complete machine learning workflow through authentic training results, model comparisons, segmentation visuals, and an interactive pre-trip fare-category estimator.',
         ('NYC TLC dataset', 'Four supplied notebooks', 'No simulated results'))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Regression algorithms','10')
    c2.metric('Classification algorithms','10')
    c3.metric('Clustering approaches','2')
    c4.metric('Analysis tracks','3')
    st.markdown('<div class="section-head"><h2>Explore the workspace</h2><span>MODEL EXPLORER</span></div>', unsafe_allow_html=True)
    cards = [
       ('01','Classification Part A','Logistic regression, KNN, Naive Bayes, decision tree, and SVC.'),
       ('02','Classification Part B','Ensembles, boosting, bagging, and neural-network classifiers.'),
       ('03','Clustering','Discover passenger-trip groupings with K-Means and hierarchical clustering.'),
       ('04','Regression','Review ten fare-prediction algorithms, metrics, and diagnostics.')]
    for offset in (0,2):
        left,right = st.columns(2)
        for column, (num,title,desc) in zip((left,right), cards[offset:offset+2]):
            with column:
                st.markdown(f'<div class="tile"><div class="number">TRACK {num}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
                st.button(
                    f'Open {title} →',
                    key=f'open_{num}',
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(title,),
                )
    st.markdown('<div class="note">All model figures and printed metrics shown in the explorer are extracted from the saved notebook outputs. A saved output is historical evidence, not proof that the notebook currently reruns without error.</div>',unsafe_allow_html=True)
    st.subheader('Available visual evidence')
    overview = []
    for title,(name,_,_) in NOTEBOOKS.items():
        cells=notebook(name)['cells']; visuals=sum(1 for _ in plot_outputs(cells))
        overview.append({'Notebook':title, 'Code cells':sum(c['cell_type']=='code' for c in cells), 'Saved charts':visuals, 'Output cells':sum(bool(c.get('outputs')) for c in cells)})
    st.dataframe(pd.DataFrame(overview),hide_index=True,use_container_width=True)

elif page in NOTEBOOKS:
    name, subtitle, number = NOTEBOOKS[page]
    data = notebook(name)
    cells=data['cells']
    plots=list(plot_outputs(cells))
    hero(f'Track {number} / model explorer',page,subtitle,
         (f'{sum(c["cell_type"]=="code" for c in cells)} code cells',f'{len(plots)} saved charts', 'Original notebook outputs'))
    tab1,tab2,tab3 = st.tabs(['📊  Visual gallery','📋  Evaluation & output','📒  Notebook explorer'])
    with tab1:
        st.markdown('#### Visual gallery')
        if not plots: st.info('This notebook has no saved PNG chart outputs.')
        else:
            kw=st.text_input('Filter gallery by source code',placeholder='Search ROC, confusion, elbow, dendrogram, PCA ...',key='gallery_'+name)
            images=[p for p in plots if not kw or kw.lower() in p[1].lower()]
            st.caption(f'{len(images)} saved figures · displayed directly from notebook execution outputs')
            for j in range(0,len(images),2):
                cs=st.columns(2)
                for col,(index,source,png) in zip(cs,images[j:j+2]):
                    with col:
                        st.markdown('<div class="plot-card">',unsafe_allow_html=True)
                        st.image(png,use_container_width=True)
                        st.caption(f'Cell {index + 1} · Saved output')
                        st.markdown('</div>',unsafe_allow_html=True)
    with tab2:
        st.markdown('#### Notebook execution results')
        st.caption('Printed metrics and tables are rendered verbatim from saved outputs, not calculated or modified here.')
        filter_txt=st.text_input('Find a metric, table, or algorithm',placeholder='Accuracy, F1, silhouette, RMSE, AUC ...',key='filter_'+name)
        shown=0
        for i,cell in enumerate(cells):
            if cell.get('cell_type')!='code':continue
            source=''.join(cell.get('source',[])); pieces=[output_text(o).strip() for o in cell.get('outputs',[])]
            result='\n'.join(p for p in pieces if p)
            if not result or (filter_txt and filter_txt.lower() not in (source+' '+result).lower()):continue
            shown+=1
            with st.expander(f'Output {shown:02d}  ·  Cell {i+1}',expanded=bool(filter_txt)):
                st.code(result[:30000],language='text')
        if not shown: st.info('No saved text outputs match this filter.')
    with tab3:
        st.caption('Inspect the original notebook cell sources and their saved results without changing them.')
        show_code=st.toggle('Show code for expanded cells',value=False,key='code_'+name)
        search=st.text_input('Filter notebook cells',placeholder='Search code or markdown',key='source_'+name)
        matching=0
        for i,cell in enumerate(cells):
            source=''.join(cell.get('source',[]));
            if search and search.lower() not in source.lower():continue
            matching+=1
            with st.expander(f'{i+1:02d} · {cell["cell_type"].title()} · {source.strip().splitlines()[0][:95] if source.strip() else "Empty cell"}'):
                if cell['cell_type']=='markdown':st.markdown(source)
                elif show_code:st.code(source,language='python')
                for o in cell.get('outputs',[]):
                    t=output_text(o)
                    if t:st.code(t[:18000],language='text')
                    if 'image/png' in o.get('data',{}):
                        try:st.image(base64.b64decode(''.join(o['data']['image/png'])),use_container_width=True)
                        except (TypeError,ValueError):pass
        st.caption(f'{matching} notebook cells displayed')

else:
    hero('Interactive inference / classifier', 'Live Fare Category Prediction',
         'Estimate a Low, Medium, or High fare category from pickup details and an estimated trip distance using the trained pre-trip MLP pipeline.',
         ('No actual fares as input','No trip-duration leakage','Pipeline-powered'))
    model_path=ROOT/'models'/'classification_pipeline.joblib'
    meta_path=ROOT/'models'/'classification_metadata.json'
    if not model_path.exists() or not meta_path.exists():
        st.warning('A trained model artifact has not been exported yet. Live predictions are disabled rather than simulated.')
        st.code('python train_export.py',language='powershell')
        st.caption('This command requires the supplied NYC TLC parquet dataset in data/. It creates the model and evaluation metadata.')
    else:
        @st.cache_resource(show_spinner=False)
        def load_model(path, mtime):
            return joblib.load(path)
        try:
            model=load_model(str(model_path),model_path.stat().st_mtime)
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('features')!=list(model.feature_names_in_):
                raise ValueError('Exported model and metadata have different feature schemas')
        except Exception as exc:
            st.error(f'Model loading error: {exc}');st.stop()
        st.success('Trained classification pipeline loaded successfully.')
        with st.expander('Model provenance & held-out evaluation'):
            st.write('Model:',meta.get('model','Not specified'))
            st.write('Training rows:',meta.get('train_records','Not recorded'))
            st.write('Test rows:',meta.get('test_records','Not recorded'))
            if meta.get('heldout_test_metrics'):st.json(meta['heldout_test_metrics'])
            st.caption(meta.get('evaluation_note',''))
        st.markdown('#### Enter trip information')
        with st.form('predict_trip'):
            left,right=st.columns(2,gap='large')
            with left:
                pickup=st.number_input('Pickup TLC zone ID',1,265,161)
                dropoff=st.number_input('Drop-off TLC zone ID',1,265,236)
                distance=st.number_input('Estimated route distance (miles)',min_value=0.11,max_value=99.99,value=3.0,step=0.1)
            with right:
                passengers=st.number_input('Passengers',1,6,1)
                hour=st.slider('Pickup hour',0,23,12)
                day=st.selectbox('Pickup weekday',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            submitted=st.form_submit_button('Predict fare category →',use_container_width=True,type='primary')
        if submitted:
            try:
                values={'PULocationID':str(pickup),'DOLocationID':str(dropoff),'day_of_week':day,
                        'passenger_count':int(passengers),'trip_distance':float(distance),'pickup_hour':int(hour),
                        'is_weekend':int(day in ['Saturday','Sunday'])}
                x=pd.DataFrame([values],columns=meta['features'])
                prediction=str(model.predict(x)[0])
                if prediction not in meta['labels']:raise ValueError('Model returned a class not defined in metadata.')
                st.markdown('### Your fare-category prediction')
                st.metric('Predicted category',prediction)
                th=meta.get('fare_thresholds_usd')
                if th:st.caption(f"Training-defined boundaries: Low ≤ ${th['low_max']:.2f}; Medium ≤ ${th['medium_max']:.2f}; High > ${th['medium_max']:.2f}.")
                if hasattr(model,'predict_proba'):
                    probabilities=model.predict_proba(x)[0]
                    fig=go.Figure(go.Bar(x=list(model.classes_),y=list(probabilities),marker_color=['#28d9c1','#4f94cd','#e7a86d'][:len(probabilities)]))
                    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title='Uncalibrated class probabilities',yaxis_range=[0,1],height=330,margin=dict(l=20,r=20,t=50,b=20))
                    st.plotly_chart(fig,use_container_width=True)
                    st.caption('These are uncalibrated model probabilities, not validated confidence levels.')
            except Exception as exc:st.error(f'Prediction could not be generated: {exc}')

st.markdown('<div class="small-muted" style="margin-top:40px;border-top:1px solid #20384b;padding-top:16px">NYC Taxi Intelligence Studio  ·  Machine Learning Capstone  ·  Authentic saved notebook outputs</div>',unsafe_allow_html=True)
