"""CrossMind research workbench. Run: python -m streamlit run app/main.py"""
from pathlib import Path
import json
import sys
import uuid
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
st.set_page_config(page_title='CrossMind AI', page_icon='◈', layout='wide')
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from crossmind.inference import Predictor
except (ImportError, OSError, AttributeError) as dependency_error:
    st.caption('CROSSMIND / SAVED RESEARCH RESULTS')
    st.title('Observe motion. Investigate what comes next.')
    st.error('Live model inference is unavailable: a required numerical library could not load. Saved results below come from previously completed runs.')
    with st.expander('Environment error'):
        st.code(str(dependency_error))
    st.subheader('Previously measured experiments')
    comparison_path = ROOT/'experiments/jaad_default/comparison.json'
    if comparison_path.exists():
        results = json.loads(comparison_path.read_text())
        lines = ['| Model | F1 | ROC-AUC | minADE@3 | minFDE@3 |', '|---|---:|---:|---:|---:|']
        for name, values in results.items():
            cells = [f"{values[k]['mean']:.4f}" if k in values else '—' for k in ['f1','roc_auc','minADE','minFDE']]
            lines.append('| '+name+' | '+' | '.join(cells)+' |')
        st.markdown('\n'.join(lines))
        st.caption('Three-seed means from the recorded JAAD pilot. Normalized-image distances; uncalibrated models. No new predictions are being generated.')
    preview = ROOT/'reports/demo_preview.jpg'
    if preview.exists():
        st.image(str(preview), caption='Previously recorded real JAAD demonstration, joint model seed 17', width='stretch')
    report = ROOT/'reports/PILOT_REPORT.md'
    if report.exists():
        st.download_button('Download pilot report', report.read_bytes(), 'CrossMind-pilot-report.md', 'text/markdown')
    demo = ROOT/'runs/joint_demo/annotated.mp4'
    if demo.exists():
        st.download_button('Download recorded demonstration', demo.read_bytes(), 'CrossMind-demo.mp4', 'video/mp4')
    st.info('An environment approved to load PyTorch and Pandas is required to resume live inference and full interface verification. No security settings have been changed.')
    st.stop()

def read_data(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}

st.markdown('''<style>
.stApp {background: #0b111b; color: #e7edf7;}
section[data-testid="stSidebar"] {background:#101c2b;}
div[data-testid="stMetric"] {background:#152236;padding:18px;border-radius:12px;}
h1 {letter-spacing:-1px;} .block-container {padding-top:4.5rem;}
header[data-testid="stHeader"] {background:#0b111b;}
div[data-testid="stMetricValue"] {font-size:1.5rem;}
</style>''', unsafe_allow_html=True)
st.caption('CROSSMIND / PEDESTRIAN BEHAVIOR LAB')
st.title('Observe motion. Investigate what comes next.')
st.write('A research workspace for crossing initiation and uncertain pedestrian trajectories.')
st.warning('Pilot models · image coordinates · uncalibrated scores. Crossing onset is conditional on the person not already crossing.')

checkpoints = sorted((ROOT/'experiments/jaad_default').glob('*/best.pt'))
with st.sidebar:
    st.subheader('Prediction model')
    names = ['Constant velocity']+[p.parent.name for p in checkpoints]
    name = st.selectbox('Model / seed', names)
    checkpoint = next((p for p in checkpoints if p.parent.name == name), None)
    st.caption('Only checkpoints actually trained on this machine appear here.')
    st.divider()
    st.write('Solid line — observed history')
    st.write('Dashed lines — model forecasts')
    st.write('Dotted line — annotated future, when available')
    st.caption('Pose display is available in video analysis. Pose and scene features are not inputs to these pilot models.')

samples, video, research = st.tabs(['Track explorer', 'Video analysis', 'Research comparison'])
with samples:
    path = ROOT/'data/processed/jaad_default/test.npz'
    if not path.exists():
        st.info('Real annotation samples unavailable. Run dataset preparation first.')
    else:
        data = read_data(path)
        index = st.slider('Held-out observation window', 0, len(data['x'])-1, 0)
        st.caption(str(data['keys'][index]))
        observed, target = data['positions'][index], data['y'][index]
        cfg = json.loads((path.parent/'manifest.json').read_text())['config']
        hz = cfg['source_fps']/cfg['sample_step']
        times = np.arange(len(observed))/hz
        if checkpoint:
            predictor = Predictor(checkpoint)
            prediction = predictor(observed, times, data['x'][index, :, 6:])
        else:
            from crossmind.trajectory import predict
            prediction = {'paths': predict(observed, times, times[-1]+np.arange(1,len(target)+1)/hz).tolist(),
                          'initiation_probability': None, 'mode_scores': []}
        left, right = st.columns([3, 1])
        with left:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#101c2b')
            ax.set_facecolor('#101c2b')
            ax.plot(*observed.T, color='#51cde8', linewidth=3, label='Observed')
            ax.scatter(*observed[-1], color='white', s=45, zorder=5)
            for k, future in enumerate(prediction['paths']):
                points = np.vstack([observed[-1], future])
                ax.plot(*points.T, '--', linewidth=2, label=f'Forecast {k+1}')
            ax.plot(*(target+observed[-1]).T, ':', color='#eeeeee', label='Annotated future')
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='datalim')
            ax.tick_params(colors='#c8d4e4')
            ax.set_xlabel('Image x / width', color='white')
            ax.set_ylabel('Image y / height', color='white')
            ax.legend()
            ax.grid(alpha=.15)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            p = prediction['initiation_probability']
            st.metric('Crossing-onset score', f'{p:.1%}' if p is not None else 'Unavailable')
            st.metric('Forecast horizon', f"{cfg['prediction_seconds']} s")
            st.metric('Annotated onset within horizon', 'Yes' if data['label'][index] else 'No')
            if prediction['mode_scores']:
                st.write('Mixture scores — calibration not established')
                st.bar_chart(pd.DataFrame({'score': prediction['mode_scores']}, index=[f'Mode {i+1}' for i in range(len(prediction['mode_scores']))]))
            st.caption('This viewer uses ground-truth track observations. Live detector errors create an additional distribution shift.')

with video:
    st.subheader('Inspect a video')
    recorded = ROOT/'runs/joint_demo'
    if (recorded/'predictions.json').exists():
        if st.button('Open recorded JAAD demonstration'):
            st.session_state['last_video_report'] = json.loads((recorded/'predictions.json').read_text())
            st.session_state['last_video_dir'] = str(recorded)
        st.caption('The recorded demonstration uses the trained joint model, seed 17. Attribution: JAAD, Rasouli, Kotseruba and Tsotsos (2017).')
    uploaded = st.file_uploader('Upload permitted footage', type=['mp4', 'avi', 'mov'])
    weights = ROOT/'assets/yolo11n-pose.pt'
    if not weights.exists():
        st.info('Detector weights unavailable. Follow the setup instructions to download them.')
    limit = st.slider('Maximum frames to process', 30, 900, 150, step=30)
    show_pose = st.checkbox('Show pose skeletons', True)
    if st.button('Analyze video', disabled=uploaded is None or not weights.exists()):
        folder = ROOT/'runs'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        source = folder/'input.mp4'
        source.write_bytes(uploaded.getvalue())
        progress = st.progress(0)
        preview = st.empty()
        def update(i, total, frame):
            progress.progress(min(i/total, 1.))
            if i % 10 == 0:
                preview.image(frame, channels='BGR', use_container_width=True)
        try:
            from crossmind.video import process_video
            report = process_video(source, folder, weights, checkpoint, limit, show_pose, update)
            st.session_state['last_video_report'] = report
            st.session_state['last_video_dir'] = str(folder)
            progress.progress(1.)
        except Exception as exc:
            st.error(f'Analysis failed: {exc}')
    if 'last_video_report' in st.session_state:
        report = st.session_state['last_video_report']
        if Path(st.session_state['last_video_dir']) == recorded and (ROOT/'reports/demo_preview.jpg').exists():
            st.image(str(ROOT/'reports/demo_preview.jpg'), caption='Frame from the actual recorded joint-model run', width='stretch')
        a, b, c = st.columns(3)
        ids = sorted({t['id'] for f in report['predictions'] for t in f['tracks']})
        a.metric('Tracks observed', len(ids))
        b.metric('Processing FPS', f"{report['processing_fps']:.1f}")
        c.metric('Current crossing state', 'Unavailable')
        st.caption(f"Analyzed with {report['model']}. Constant-frame-rate input assumed; forecasts are uncalibrated.")
        if ids:
            ident = st.selectbox('Inspect pedestrian ID', ids)
            records = [{'time_s': f['timestamp'], 'detection_confidence': t['confidence'],
                        'onset_score': (t['prediction'] or {}).get('initiation_probability')}
                       for f in report['predictions'] for t in f['tracks'] if t['id'] == ident]
            st.dataframe(pd.DataFrame(records), width='stretch')
        folder = Path(st.session_state['last_video_dir'])
        st.download_button('Download annotated video', (folder/'annotated.mp4').read_bytes(), 'crossmind.mp4', 'video/mp4')
        st.download_button('Download predictions', (folder/'predictions.json').read_bytes(), 'predictions.json', 'application/json')

with research:
    st.subheader('Measured experiments')
    st.caption('Official clip splits, custom at-risk cohort. Scores are not directly comparable with published JAAD crossing benchmarks.')
    metrics = sorted((ROOT/'experiments/jaad_default').glob('*/metrics.json'))
    rows = []
    for path in metrics:
        r = json.loads(path.read_text())
        rows.append({'model': r['task'], 'seed': r['seed'], 'epochs': r['epochs'],
                     'test samples': r['test']['count'],
                     'F1': r['test'].get('intention', {}).get('f1'),
                     'Brier': r['test'].get('intention', {}).get('brier'),
                     'minADE': r['test'].get('trajectory', {}).get('minADE'),
                     'minFDE': r['test'].get('trajectory', {}).get('minFDE')})
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')
        for path in metrics:
            with st.expander(path.parent.name+' training history'):
                history = pd.DataFrame(json.loads((path.parent/'history.json').read_text())).set_index('epoch')
                st.line_chart(history)
    else:
        st.info('No completed experiments yet. Metrics will appear after actual training.')
    ablation_path = ROOT/'experiments/ablations/comparison.json'
    if ablation_path.exists():
        st.subheader('Feature ablations')
        ablations = json.loads(ablation_path.read_text())
        st.dataframe(pd.DataFrame([{'features': name, **{key: value['mean'] for key, value in result.items()}}
                                  for name, result in ablations.items()]), width='stretch')
        st.caption('Three-seed means, identical windows. The current neighbor features did not consistently improve prediction; see the report for variation and limitations.')
