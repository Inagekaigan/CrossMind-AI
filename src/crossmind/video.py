"""Video detection, ByteTrack IDs, pose and timestamped trajectory overlays."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .trajectory import predict
from .inference import Predictor


def dashed_path(image, points, color):
    for a, b in zip(points[:-1], points[1:]):
        length = np.linalg.norm(b-a)
        if not np.isfinite(length) or length > 20000:
            continue
        for start in np.arange(0, length, 12):
            p = a+(b-a)*start/max(length, 1)
            q = a+(b-a)*min(start+7, length)/max(length, 1)
            cv2.line(image, tuple(p.astype(int)), tuple(q.astype(int)), color, 2, cv2.LINE_AA)


def process_video(source, output_dir, weights, checkpoint=None, max_frames=300, pose=True, progress=None):
    from ultralytics import YOLO
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError('max_frames must be a positive integer')
    if not Path(weights).is_file():
        raise FileNotFoundError('Detector weights unavailable. Download the documented weights first.')
    model = YOLO(str(weights))
    predictor = Predictor(checkpoint) if checkpoint else None
    config = predictor.config if predictor else dict(source_fps=30, sample_step=3, observation_seconds=1, prediction_seconds=2)
    hz = config['source_fps']/config['sample_step']
    count = round(config['observation_seconds']*hz)+1
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Video could not be opened')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid frame rate')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer, histories, log = None, defaultdict(lambda: deque(maxlen=600)), []
    started = time.perf_counter()
    try:
        for index in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(output_dir/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Output video encoder unavailable')
            timestamp = index/fps  # Constant-frame-rate inputs; metadata recorded below.
            tick = time.perf_counter()
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0],
                                 conf=.15, imgsz=1280, verbose=False, device='cpu')[0]
            detector_ms = (time.perf_counter()-tick)*1000
            people = []
            if result.boxes is not None and result.boxes.id is not None:
                for box, ident, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().tolist(), result.boxes.conf.tolist()):
                    people.append((ident, box, confidence, np.array([(box[0]+box[2])/2/w, box[3]/h])))
            canvas = result.plot(boxes=False, labels=False, probs=False, kpt_line=True) if pose and result.keypoints is not None else frame.copy()
            frame_log = {'frame': index, 'timestamp': timestamp, 'detector_tracker_ms': detector_ms, 'tracks': []}
            for ident, box, confidence, position in people:
                hist = histories[ident]
                if hist and timestamp-hist[-1][0] > .2:
                    hist.clear()
                radius = predictor.config.get('neighbor_radius', .2) if predictor else .2
                neighbors = [p-position for other, _, _, p in people if other != ident and np.linalg.norm(p-position) <= radius]
                social = np.r_[np.mean(neighbors, axis=0), len(neighbors)] if neighbors else np.zeros(3)
                hist.append((timestamp, position, social))
                pts = np.array([v[1]*[w, h] for v in hist], np.int32)
                if len(pts) > 1:
                    cv2.polylines(canvas, [pts], False, (240, 200, 50), 2)
                prediction = None
                if timestamp-hist[0][0] >= config['observation_seconds']-1e-6:
                    desired = timestamp-np.arange(count-1, -1, -1)/hz
                    available = np.array([v[0] for v in hist])
                    ids = np.abs(available[:, None]-desired).argmin(axis=0)
                    # Reject missing observations instead of extrapolating gaps into model inputs.
                    if len(set(ids)) == count and np.max(np.abs(available[ids]-desired)) <= .51/fps+1e-6:
                        positions = np.array([hist[i][1] for i in ids])
                        neighbor_values = np.array([hist[i][2] for i in ids])
                        if predictor:
                            prediction = predictor(positions, desired, neighbor_values)
                        else:
                            future = timestamp+np.arange(1, round(config['prediction_seconds']*hz)+1)/hz
                            prediction = {'paths': predict(positions, desired, future).tolist(),
                                          'initiation_probability': None, 'mode_scores': [], 'status': 'constant_velocity baseline'}
                if prediction:
                    for mode, path in enumerate(prediction['paths']):
                        pts_future = np.vstack([position, path])*[w, h]
                        dashed_path(canvas, pts_future, [(80, 210, 130), (220, 140, 80), (160, 90, 220)][mode % 3])
                probability = prediction.get('initiation_probability') if prediction else None
                label = f'ID {ident}' + (f' onset* {probability:.0%}' if probability is not None else '')
                a, b, c, d = box.astype(int)
                cv2.rectangle(canvas, (a, b), (c, d), (80, 210, 130), 2)
                cv2.putText(canvas, label, (a, max(20, b-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)
                frame_log['tracks'].append({'id': ident, 'box': box.tolist(), 'confidence': confidence,
                                            'position': position.tolist(), 'prediction': prediction})
            # Expire absent IDs to bound memory in long videos.
            for ident in list(histories):
                if timestamp-histories[ident][-1][0] > 2:
                    del histories[ident]
            cv2.putText(canvas, 'CrossMind | research pilot | solid: observed / dashed: forecast', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            cv2.putText(canvas, '* onset conditional on not crossing; uncalibrated', (15, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            writer.write(canvas)
            frame_log['pipeline_ms'] = (time.perf_counter()-tick)*1000
            log.append(frame_log)
            if progress:
                progress(index+1, max_frames, canvas)
    finally:
        cap.release()
        if writer:
            writer.release()
    if not log:
        raise ValueError('Video contains no decodable frames')
    elapsed = time.perf_counter()-started
    report = {'source': str(source), 'frames': len(log), 'source_fps': fps,
              'timestamp_policy': 'frame index / metadata FPS; constant-frame-rate video required',
              'elapsed_seconds': elapsed, 'processing_fps': len(log)/elapsed,
              'model': str(checkpoint) if checkpoint else 'constant_velocity',
              'current_crossing_state': 'unavailable', 'scene_features': 'unavailable',
              'pose_used_by_predictor': False, 'predictions': log}
    (output_dir/'predictions.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--max-frames', type=int, default=300)
    a = p.parse_args()
    r = process_video(a.source, a.output, a.weights, a.checkpoint, a.max_frames)
    print(json.dumps({k: r[k] for k in ['frames', 'elapsed_seconds', 'processing_fps']}))
