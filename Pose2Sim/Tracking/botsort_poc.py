#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###################################################
    ## BoT-SORT + OSNet ReID POC on a single camera  ##
    ###################################################

    Re-tracks a Pose2Sim camera offline using BoT-SORT (boxmot) with OSNet
    appearance embeddings, by deriving per-person bounding boxes from the
    existing pose-estimation JSON files. Produces an annotated MP4 plus
    console stats so you can visually validate ID stability through
    occlusions before integrating boxmot into the main pipeline.

    Why this exists:
      The default sports2d tracker in poseEstimation.py is a frame-to-frame
      Hungarian on keypoint distance with no appearance memory. It fails
      whenever a person leaves the frame for a few seconds and others
      reshuffle. BoT-SORT + OSNet handles long-term re-ID via appearance
      embeddings, which is essential for sessions where patients repeatedly
      walk out of view.

    Requires:
      pip install boxmot
      OSNet weights are auto-downloaded on first run.

    Usage:
      botsort_poc -j <json_folder> -i <video.mp4>
      botsort_poc -j <json_folder> -i <video.mp4> -o <output.mp4> --device cuda:0
      from Pose2Sim.Tracking import botsort_poc; botsort_poc.botsort_poc_func(json_folder=r'<path>', input=r'<video>')
'''


## INIT
import os
# Mitigate libiomp5md.dll double-load between numpy / torch / sklearn on Windows.
# Must be set BEFORE importing numpy/torch (boxmot pulls torch in).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import re
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import cv2


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
BBOX_KPT_CONF_THRESHOLD = 0.3   # min keypoint confidence to include in bbox
BBOX_MIN_VALID_KPTS = 5         # below this, the person detection is dropped
BBOX_PADDING_X = 0.10           # horizontal padding (10% of bbox width)
BBOX_PADDING_Y_TOP = 0.15       # top padding (head halo)
BBOX_PADDING_Y_BOT = 0.05       # bottom padding

# Pose model's body keypoints span the first 26 entries of pose_keypoints_2d
# in COCO_133_WRIST -> using only body keypoints for the bbox avoids hand /
# face outliers when limbs are partially visible.
BODY_KPT_COUNT = 26


## FUNCTIONS
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def bbox_from_keypoints_flat(kpts_flat):
    '''
    Derive a padded bbox + mean confidence from a flat OpenPose-style
    pose_keypoints_2d list.

    OUTPUTS:
    - (x1, y1, x2, y2, conf) or None if too few reliable keypoints
    '''
    kpts = np.asarray(kpts_flat, dtype=np.float32).reshape(-1, 3)
    if kpts.size == 0:
        return None
    # Use body keypoints first if available (more stable than hand/face)
    body = kpts[:BODY_KPT_COUNT]
    valid_mask = (~np.isnan(body[:, 0])) & (body[:, 2] > BBOX_KPT_CONF_THRESHOLD)
    valid = body[valid_mask]
    if len(valid) < BBOX_MIN_VALID_KPTS:
        # Fall back to all keypoints
        valid_mask = (~np.isnan(kpts[:, 0])) & (kpts[:, 2] > BBOX_KPT_CONF_THRESHOLD)
        valid = kpts[valid_mask]
        if len(valid) < BBOX_MIN_VALID_KPTS:
            return None

    x1, y1 = float(valid[:, 0].min()), float(valid[:, 1].min())
    x2, y2 = float(valid[:, 0].max()), float(valid[:, 1].max())
    w, h = x2 - x1, y2 - y1
    x1 -= w * BBOX_PADDING_X
    x2 += w * BBOX_PADDING_X
    y1 -= h * BBOX_PADDING_Y_TOP
    y2 += h * BBOX_PADDING_Y_BOT
    conf = float(np.clip(valid[:, 2].mean() / 6.0, 0.0, 1.0))  # RTMW scale ~0-7 -> 0-1
    return (x1, y1, x2, y2, conf)


def load_frame_detections(json_path):
    '''
    Load all valid detections from a Pose2Sim JSON file as a (N, 6) array
    of [x1, y1, x2, y2, conf, cls=0] rows.
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)
    dets = []
    for person in data.get('people', []):
        kpts_flat = person.get('pose_keypoints_2d', [])
        bbox = bbox_from_keypoints_flat(kpts_flat)
        if bbox is None:
            continue
        x1, y1, x2, y2, conf = bbox
        dets.append([x1, y1, x2, y2, conf, 0])
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(dets, dtype=np.float32)


def color_for_id(track_id):
    '''Deterministic BGR color from a track ID.'''
    rng = np.random.default_rng(int(track_id) * 9973 + 17)
    return tuple(int(c) for c in rng.integers(60, 230, size=3))


def botsort_poc_func(json_folder, input, output=None, tracks_json_path=None,
                     device='cuda:0',
                     reid_model='lmbn_n_market.pt', half=True,
                     max_frames=None,
                     track_buffer=30, cmc_method='ecc',
                     appearance_thresh=0.25, proximity_thresh=0.5,
                     new_track_thresh=0.7, match_thresh=0.8):
    '''
    Run BoT-SORT + OSNet ReID on one camera's pose JSONs + video.

    INPUTS:
    - json_folder: str. Folder with Pose2Sim per-frame JSONs (one per frame).
    - input: str. Path to the camera's video file.
    - output: str or None. Path of the annotated MP4 to write. Defaults to
      "<json_folder>_botsort.mp4" next to the JSON folder.
    - device: str. 'cpu', 'cuda:0', ...
    - reid_model: str. boxmot-supported ReID checkpoint name (auto-downloads).
    - half: bool. FP16 inference (GPU only).
    - max_frames: int or None. Limit processing for quick smoke tests.

    OUTPUTS:
    - Annotated MP4 written to disk.
    - Returns a dict with per-track stats.
    '''
    # Lazy import: boxmot is heavy and optional
    try:
        from boxmot.trackers.botsort.botsort import BotSort
        from boxmot.reid.core.reid import ReID
    except ImportError as e:
        raise ImportError("boxmot is required: `pip install boxmot`") from e

    json_folder = Path(json_folder)
    video_path = Path(input)
    assert json_folder.is_dir(), f"Not a folder: {json_folder}"
    assert video_path.is_file(), f"Not a file: {video_path}"

    json_paths = sorted(json_folder.glob('*.json'), key=lambda p: natural_sort_key(p.name))
    assert len(json_paths) > 0, f"No JSON files in {json_folder}"

    if output is None:
        output = str(json_folder.parent / (json_folder.name + '_botsort.mp4'))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = min(len(json_paths), total_video_frames)
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    logging.info(f"Video: {video_path.name} ({width}x{height} @ {fps:.1f} fps, "
                 f"{total_video_frames} frames)")
    logging.info(f"JSONs: {len(json_paths)} files")
    logging.info(f"Processing {n_frames} frames")
    logging.info(f"Device: {device}, ReID weights: {reid_model}, half: {half}")
    logging.info(f"BotSort tuning: track_buffer={track_buffer} ({track_buffer/30:.1f}s "
                 f"@30fps), cmc={cmc_method}, appearance_thresh={appearance_thresh}")

    reid_backend = ReID(
        weights=Path(reid_model),
        device=device,
        half=half if device != 'cpu' else False,
    ).model

    tracker = BotSort(
        reid_model=reid_backend,
        track_buffer=track_buffer,
        cmc_method=cmc_method,
        appearance_thresh=appearance_thresh,
        proximity_thresh=proximity_thresh,
        new_track_thresh=new_track_thresh,
        match_thresh=match_thresh,
        with_reid=True,
        frame_rate=int(fps),
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Cannot open writer: {output}"

    # Stats accumulators
    track_first_seen = {}   # id -> frame index of first detection
    track_last_seen = {}    # id -> frame index of last detection
    track_hit_count = {}    # id -> nb frames the track was active
    # Per-frame detections, for the post-hoc merger to consume.
    # frame_idx -> list of dicts {id, bbox=[x1,y1,x2,y2], conf}
    per_frame_dets = []

    for f_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame {f_idx}: video read failed, stopping")
            break

        dets = load_frame_detections(json_paths[f_idx])
        tracks = tracker.update(dets, frame)
        # tracks: (M, 8) -> [x1, y1, x2, y2, id, conf, cls, det_ind]

        frame_records = []
        for tr in tracks:
            x1, y1, x2, y2, tid, conf = (
                float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3]),
                int(tr[4]), float(tr[5])
            )
            track_first_seen.setdefault(tid, f_idx)
            track_last_seen[tid] = f_idx
            track_hit_count[tid] = track_hit_count.get(tid, 0) + 1
            frame_records.append({
                'id': tid,
                'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                'conf': round(conf, 3),
            })

            color = color_for_id(tid)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID {tid}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 8),
                          (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
        per_frame_dets.append(frame_records)

        # HUD: frame index + active track count
        hud = f"frame {f_idx+1}/{n_frames}   active tracks {len(tracks)}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        if (f_idx + 1) % 200 == 0:
            logging.info(f"  ...{f_idx + 1}/{n_frames} frames processed")

    cap.release()
    writer.release()

    # ---- Dump tracks JSON sidecar for the post-hoc merger ----
    if tracks_json_path is None:
        tracks_json_path = output.with_name(output.stem + '_tracks.json')
    else:
        tracks_json_path = Path(tracks_json_path)
        tracks_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tracks_json_path, 'w') as fjson:
        json.dump({
            'video': str(video_path),
            'json_folder': str(json_folder),
            'width': width,
            'height': height,
            'fps': fps,
            'n_frames': n_frames,
            'tracker': 'botsort',
            'reid_model': reid_model,
            'frames': per_frame_dets,
        }, fjson)
    logging.info(f"Tracks JSON saved to: {tracks_json_path}")

    # ---- Stats ----
    durations = {tid: track_last_seen[tid] - track_first_seen[tid] + 1
                 for tid in track_first_seen}
    long_tracks = [tid for tid, d in durations.items() if d >= 100]
    very_long = [tid for tid, d in durations.items() if d >= n_frames * 0.5]

    logging.info("")
    logging.info(f"==== BoT-SORT POC stats ====")
    logging.info(f"Total tracks created: {len(track_first_seen)}")
    logging.info(f"Tracks with >= 100 frames lifespan: {len(long_tracks)}")
    logging.info(f"Tracks covering >= 50% of session: {len(very_long)} -> IDs {sorted(very_long)}")
    if durations:
        median_dur = int(np.median(list(durations.values())))
        mean_dur = float(np.mean(list(durations.values())))
        logging.info(f"Median track duration: {median_dur} frames "
                     f"(mean {mean_dur:.0f})")
    logging.info(f"Annotated MP4 saved to: {output}")

    return {
        'n_tracks_total': len(track_first_seen),
        'n_tracks_long': len(long_tracks),
        'long_track_ids': sorted(long_tracks),
        'durations': durations,
        'output_path': str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-j', '--json_folder', required=True,
                        help="Folder with Pose2Sim per-frame JSONs.")
    parser.add_argument('-i', '--input', required=True,
                        help="Camera video file (mp4).")
    parser.add_argument('-o', '--output', default=None,
                        help="Output annotated MP4. Default: <json_folder>_botsort.mp4")
    parser.add_argument('--device', default='cuda:0',
                        help="cpu, cuda:0, ...")
    parser.add_argument('--reid_model', default='lmbn_n_market.pt',
                        help="boxmot ReID checkpoint name (must exist in <env>/site-packages/models/)")
    parser.add_argument('--no-half', action='store_true',
                        help="Disable FP16 inference")
    parser.add_argument('--max_frames', type=int, default=None,
                        help="Cap nb of frames (smoke tests)")
    parser.add_argument('--track_buffer', type=int, default=30,
                        help="Frames a track is kept alive after going lost. "
                             "Default 30 (1s @30fps) so BoT-SORT only bridges short occlusions "
                             "and every longer absence creates a new track that goes through "
                             "the merger + manual review (no wrong auto re-IDs).")
    parser.add_argument('--cmc_method', default='ecc',
                        help="Camera motion compensation: ecc (default), orb, sift, sof.")
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help="Cosine distance threshold for ReID matching (lower = stricter)")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help="Spatial IoU gate before considering appearance match")
    parser.add_argument('--new_track_thresh', type=float, default=0.7,
                        help="Detection confidence required to spawn a new track")
    parser.add_argument('--match_thresh', type=float, default=0.8,
                        help="Hungarian cost threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    botsort_poc_func(
        json_folder=args.json_folder,
        input=args.input,
        output=args.output,
        device=args.device,
        reid_model=args.reid_model,
        half=not args.no_half,
        max_frames=args.max_frames,
        track_buffer=args.track_buffer,
        cmc_method=args.cmc_method,
        appearance_thresh=args.appearance_thresh,
        proximity_thresh=args.proximity_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
    )


if __name__ == '__main__':
    main()
