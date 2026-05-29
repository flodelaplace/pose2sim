#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #######################################################
    ## Closed-world post-hoc merger for BoT-SORT outputs ##
    #######################################################

    Reads the tracks JSON sidecar produced by `botsort_poc.py` and
    intelligently merges temporally-disjoint fragments into a small set of
    persistent identities, using three priors that complement BoT-SORT's
    built-in ReID:

    1. Appearance  : LMBN/OSNet embedding cosine similarity, computed by
                     re-cropping a few representative frames per track and
                     pooling their ReID embeddings. Independent from
                     BoT-SORT's internal logic, so the merger can keep
                     improving without touching BoT-SORT.
    2. Spatial     : Euclidean pixel distance between the last bbox center
                     of track A and the first bbox center of track B. A
                     person who disappeared near the right edge is unlikely
                     to re-emerge on the opposite side of the frame.
    3. Entry/exit  : Discrete edge label ('left', 'right', 'top', 'bottom',
       edge match    'middle') for A's exit point and B's entry point.
                     Matching edges add a bonus; opposite edges add a
                     penalty.

    Closed-world prior: a hard cap on the expected number of distinct
    persons in the session (e.g. 6 patients + 2 staff = 8). The greedy
    merger keeps fusing the highest-scoring candidate pair until either
    that cap is reached or the next pair's score falls below a confidence
    threshold.

    Output:
      - <input>_merged_tracks.json : mapping original_id -> final_id and
        the consolidated per-frame detection list.
      - <input>_merged.mp4         : annotated MP4 with the merged IDs,
        for visual validation.

    Usage:
      botsort_merger -t <tracks.json>
      botsort_merger -t <tracks.json> -n 8 --auto_thresh 0.55
      from Pose2Sim.Tracking import botsort_merger; botsort_merger.merge_func(tracks_json=r'<path>')
'''


## INIT
import os
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
N_SAMPLES_PER_TRACK = 8          # nb of frames to crop for embedding
EDGE_MARGIN_FRAC = 0.15          # bbox center within X% of border = edge
EXIT_ENTRY_WINDOW = 5            # last/first N frames used for edge dir
SPATIAL_SIGMA_PX = 400.0         # spatial score = exp(-dist / sigma)

# Integer-label convention for patient vs staff. The classifier doesn't
# care about the meaning, it just learns features per label; we use the
# numeric range to decide how to display the label and to allocate
# separate counters for "new patient" and "new staff" picks.
STAFF_ID_OFFSET = 1000           # labels >= this are staff members
                                 #   display: P3, P14   |  S1, S2, ...


def format_id(label):
    '''Pretty-print a global label as 'P<n>' or 'S<n>' based on range.'''
    try:
        n = int(label)
    except (TypeError, ValueError):
        return str(label)
    if n >= STAFF_ID_OFFSET:
        return f"S{n - STAFF_ID_OFFSET + 1}"
    return f"P{n}"

# Weights for the three priors (must sum to 1 for clarity).
# Sanity check on Demo_Seance cam 1 (6 co-visible patients in distinct
# sportswear) gave torso-histogram similarities in [0.07, 0.55] between
# any two distinct patients, so histograms are the most discriminating
# of the three priors and deserve the largest weight.
W_APPEARANCE = 0.65
W_SPATIAL = 0.20
W_EDGE = 0.15

# Edge match bonus / penalty (added to weighted base score)
EDGE_BONUS_MATCH = 1.0           # both 'right' (or both 'left' etc.)
EDGE_PENALTY_OPPOSITE = -0.5     # 'right' vs 'left' -> teleport, unlikely
# 'middle' on either side -> 0 (no information)

OPPOSITE_EDGE = {
    'left': 'right', 'right': 'left',
    'top': 'bottom', 'bottom': 'top',
}


## FUNCTIONS
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def edge_label(cx, cy, w, h):
    '''Discrete label for which edge a bbox center is closest to.'''
    fx = cx / w
    fy = cy / h
    if fx < EDGE_MARGIN_FRAC:
        return 'left'
    if fx > 1.0 - EDGE_MARGIN_FRAC:
        return 'right'
    if fy < EDGE_MARGIN_FRAC:
        return 'top'
    if fy > 1.0 - EDGE_MARGIN_FRAC:
        return 'bottom'
    return 'middle'


def histogram_similarity(h1, h2):
    '''Bhattacharyya-based similarity between two normalized histograms.
    Returns 1.0 if identical, 0.0 if totally different. Pre-trained ReID
    models (OSNet/LMBN) fail to discriminate at the distances typical of
    multi-camera mocap setups, so we fall back on a hand-crafted torso
    color descriptor that is well suited to sportswear with varied
    colors.'''
    if h1 is None or h2 is None:
        return 0.0
    h1f = h1.reshape(-1, 1).astype(np.float32)
    h2f = h2.reshape(-1, 1).astype(np.float32)
    # OpenCV's Bhattacharyya: 0 = identical, 1 = totally different
    dist = cv2.compareHist(h1f, h2f, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - float(dist))


# Torso ROI inside a person bbox. Pose models tend to put the bbox around
# the full silhouette plus some padding; the central column avoids the
# background that leaks in near the edges and the head/legs areas that
# vary a lot with pose.
TORSO_X_RANGE = (0.20, 0.80)   # horizontal: middle 60% of bbox width
TORSO_Y_RANGE = (0.20, 0.55)   # vertical: shoulder area down to mid-torso

# 3D HSV histogram bins. H is the most discriminating channel for varied
# clothing; S and V add coarse intensity info while staying robust to
# lighting changes across the trial.
HIST_BINS = [16, 8, 4]
HIST_RANGES = [0, 180, 0, 256, 0, 256]


def extract_torso_hist(frame, bbox):
    '''Extract a normalized HSV torso histogram from a person bbox.'''
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = frame.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None
    tx1 = x1 + int(TORSO_X_RANGE[0] * bw)
    tx2 = x1 + int(TORSO_X_RANGE[1] * bw)
    ty1 = y1 + int(TORSO_Y_RANGE[0] * bh)
    ty2 = y1 + int(TORSO_Y_RANGE[1] * bh)
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def build_track_summaries(per_frame_dets, video_path, width, height):
    '''
    For each track ID, gather first/last frame, first/last bbox center,
    exit/entry edge label, and a mean appearance descriptor (torso HSV
    histogram) across N_SAMPLES_PER_TRACK representative frames.

    Histograms are extracted in a SINGLE sequential video pass to avoid
    per-detection seeks, which are the bottleneck on long videos.
    '''
    # Pass 1: pool per-track detections by frame
    by_id = {}  # tid -> list of (frame_idx, bbox)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id.setdefault(r['id'], []).append((f_idx, r['bbox']))

    # Decide which (frame, tid, bbox) triples need a histogram
    needed_per_frame = {}  # frame_idx -> [(tid, bbox), ...]
    sample_items_per_tid = {}
    for tid, items in by_id.items():
        items.sort(key=lambda x: x[0])
        if len(items) <= N_SAMPLES_PER_TRACK:
            sample_items = items
        else:
            idx_sample = np.linspace(0, len(items) - 1,
                                     N_SAMPLES_PER_TRACK).astype(int)
            sample_items = [items[i] for i in idx_sample]
        sample_items_per_tid[tid] = sample_items
        for f_idx, bbox in sample_items:
            needed_per_frame.setdefault(f_idx, []).append((tid, bbox))

    # Sequential video pass: extract every needed histogram in one go
    hist_per_tid = {tid: [] for tid in by_id}
    if needed_per_frame:
        last_needed = max(needed_per_frame.keys())
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        f_idx = 0
        while f_idx <= last_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx in needed_per_frame:
                for tid, bbox in needed_per_frame[f_idx]:
                    h = extract_torso_hist(frame, bbox)
                    if h is not None:
                        hist_per_tid[tid].append(h)
            f_idx += 1
        cap.release()

    summaries = {}
    for tid, items in by_id.items():
        first_frame, first_bbox = items[0]
        last_frame, last_bbox = items[-1]
        lifespan = last_frame - first_frame + 1

        hists = hist_per_tid.get(tid, [])
        if not hists:
            mean_embed = None
        else:
            mean_embed = np.mean(np.stack(hists, axis=0), axis=0)
            # Re-normalize the pooled histogram so it stays in [0, 1] norm
            s = float(np.sum(mean_embed))
            if s > 1e-9:
                mean_embed /= s

        # Edge labels from exit and entry windows (last/first 5 detections)
        last_window = items[-min(EXIT_ENTRY_WINDOW, len(items)):]
        first_window = items[:min(EXIT_ENTRY_WINDOW, len(items))]
        last_center = bbox_center(last_window[-1][1])
        first_center = bbox_center(first_window[0][1])
        exit_edge = edge_label(*last_center, width, height)
        entry_edge = edge_label(*first_center, width, height)

        summaries[tid] = {
            'tid': tid,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'lifespan': lifespan,
            'n_detections': len(items),
            'first_center': first_center,
            'last_center': last_center,
            'exit_edge': exit_edge,
            'entry_edge': entry_edge,
            'embedding': mean_embed,
            'merged_from': [tid],  # list of original ids this group contains
        }

    return summaries


def score_pair(A, B):
    '''
    Score how likely tracks A and B are the same person, assuming A ends
    before B starts. Returns a float roughly in [-0.5, 1.5] (clip later).

    Components reported separately so the review UI can show them.
    '''
    # Appearance (Bhattacharyya similarity of torso HSV histograms)
    app = histogram_similarity(A['embedding'], B['embedding'])
    # Spatial proximity (last center of A vs first center of B)
    d_px = float(np.hypot(A['last_center'][0] - B['first_center'][0],
                          A['last_center'][1] - B['first_center'][1]))
    spatial = float(np.exp(-d_px / SPATIAL_SIGMA_PX))
    # Edge match (A exits, B enters)
    a_edge, b_edge = A['exit_edge'], B['entry_edge']
    if a_edge == 'middle' or b_edge == 'middle':
        edge = 0.0
    elif a_edge == b_edge:
        edge = EDGE_BONUS_MATCH
    elif OPPOSITE_EDGE.get(a_edge) == b_edge:
        edge = EDGE_PENALTY_OPPOSITE
    else:
        edge = 0.0
    total = W_APPEARANCE * app + W_SPATIAL * spatial + W_EDGE * edge
    return {
        'total': total,
        'appearance': app,
        'spatial': spatial,
        'edge': edge,
        'd_px': d_px,
        'a_exit': a_edge,
        'b_entry': b_edge,
    }


def build_candidate_pairs(summaries, min_gap_frames=0):
    '''
    All (A, B) pairs where A ends >= min_gap_frames before B starts.
    No temporal overlap allowed (would mean both visible simultaneously =
    must be different physical persons in a single camera).
    '''
    ids = list(summaries.keys())
    pairs = []
    for i, A_id in enumerate(ids):
        A = summaries[A_id]
        for B_id in ids:
            if A_id == B_id:
                continue
            B = summaries[B_id]
            if B['first_frame'] - A['last_frame'] <= min_gap_frames:
                continue  # overlap or touching = not a re-ID candidate
            s = score_pair(A, B)
            pairs.append((s['total'], A_id, B_id, s))
    pairs.sort(key=lambda p: -p[0])
    return pairs


def merge_groups(summaries, n_target=8, auto_thresh=0.55,
                 review_thresh=0.35):
    '''
    Greedy closed-world merging. Maintains a "current" dict of merged
    groups. At each step, recomputes the best valid pair and merges,
    until either:
      - len(groups) <= n_target AND best score < auto_thresh, or
      - best score < auto_thresh (no more confident merges).

    Pairs in [review_thresh, auto_thresh) are returned as `needs_review`
    so the user can validate them in a UI.
    '''
    groups = {tid: dict(s) for tid, s in summaries.items()}
    auto_merges = []
    needs_review = []

    while True:
        pairs = build_candidate_pairs(groups)
        if not pairs:
            break
        best_score, A_id, B_id, detail = pairs[0]

        # Reached target N: only auto-merge above auto_thresh from now on
        if len(groups) <= n_target and best_score < auto_thresh:
            # Surface remaining decent pairs for manual review
            for sc, a, b, det in pairs:
                if sc >= review_thresh and sc < auto_thresh:
                    needs_review.append({
                        'a': a, 'b': b, 'score': sc, 'detail': det,
                    })
            break

        if best_score < auto_thresh:
            # No more confident merges: surface review candidates
            for sc, a, b, det in pairs:
                if sc >= review_thresh:
                    needs_review.append({
                        'a': a, 'b': b, 'score': sc, 'detail': det,
                    })
            break

        # ---- Merge B into A (A is the chronologically earlier track) ----
        A, B = groups[A_id], groups[B_id]
        # Update group properties
        A['last_frame'] = B['last_frame']
        A['last_center'] = B['last_center']
        A['exit_edge'] = B['exit_edge']  # exit edge becomes the merged group's
        A['n_detections'] += B['n_detections']
        A['merged_from'] = A['merged_from'] + B['merged_from']
        # Pool histograms weighted by lifespan, then re-normalize
        if A['embedding'] is not None and B['embedding'] is not None:
            wA = max(1, A['lifespan'])
            wB = max(1, B['lifespan'])
            pooled = (A['embedding'] * wA + B['embedding'] * wB) / (wA + wB)
            s = float(np.sum(pooled))
            if s > 1e-9:
                pooled /= s
            A['embedding'] = pooled
        elif A['embedding'] is None:
            A['embedding'] = B['embedding']
        A['lifespan'] = A['last_frame'] - A['first_frame'] + 1
        del groups[B_id]
        auto_merges.append({
            'kept': A_id, 'merged': B_id, 'score': best_score,
            'detail': detail,
        })

    return groups, auto_merges, needs_review


def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 9973 + 17)
    return tuple(int(c) for c in rng.integers(60, 230, size=3))


def render_merged_video(per_frame_dets, id_to_final, video_path,
                        output_mp4, width, height, fps):
    '''Rebuild an annotated MP4 with the merged track IDs.'''
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))
    n_frames = len(per_frame_dets)
    for f_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        for rec in per_frame_dets[f_idx]:
            original_id = rec['id']
            final_id = id_to_final.get(original_id, original_id)
            x1, y1, x2, y2 = rec['bbox']
            color = color_for_id(final_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            label = f"{format_id(final_id)}  (was {original_id})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 8),
                          (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)
        n_unique = len(set(id_to_final.get(r['id'], r['id'])
                           for r in per_frame_dets[f_idx]))
        hud = f"frame {f_idx+1}/{n_frames}   unique persons {n_unique}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()


def merge_func(tracks_json, output_mp4=None, output_json=None,
               n_target=8, auto_thresh=1.5, review_thresh=0.20,
               reid_model='lmbn_n_market.pt', device='cuda:0', half=True,
               min_track_length=5):
    '''
    Run the post-hoc merger.

    INPUTS:
    - tracks_json: str. Path to the JSON produced by botsort_poc.
    - output_mp4: str or None. Annotated MP4 with merged IDs.
    - output_json: str or None. Final mapping + per-frame consolidated data.
    - n_target: int. Expected number of distinct persons (closed-world cap).
    - auto_thresh: float. Pairs >= this score are merged automatically.
    - review_thresh: float. Pairs in [review_thresh, auto_thresh) are flagged
      for manual validation (returned in `needs_review`).
    - reid_model: name of the ReID weights file (must be in <env>/site-packages/models/).
    - min_track_length: filter out tracks shorter than this many detections
      before merging (cuts noise from spurious 1-2 frame blips).

    OUTPUTS:
    - Returns a dict with merge decisions, group summaries, and the path to
      the produced files.
    '''
    tracks_json = Path(tracks_json)
    with open(tracks_json, 'r') as f:
        data = json.load(f)
    video_path = Path(data['video'])
    width, height = data['width'], data['height']
    fps = float(data['fps'])
    per_frame_dets = data['frames']
    n_frames = data['n_frames']

    logging.info(f"Loaded {tracks_json.name}: {n_frames} frames, "
                 f"video {video_path.name} ({width}x{height} @ {fps:.1f})")

    # Drop tracks too short to be reliable
    track_count = {}
    for recs in per_frame_dets:
        for r in recs:
            track_count[r['id']] = track_count.get(r['id'], 0) + 1
    kept_ids = {tid for tid, n in track_count.items() if n >= min_track_length}
    dropped_ids = set(track_count.keys()) - kept_ids
    if dropped_ids:
        logging.info(f"Dropping {len(dropped_ids)} tracks shorter than "
                     f"{min_track_length} detections: {sorted(dropped_ids)}")
    per_frame_dets = [[r for r in recs if r['id'] in kept_ids]
                      for recs in per_frame_dets]

    logging.info(f"Building track summaries (torso color histogram + spatial)...")
    summaries = build_track_summaries(per_frame_dets, video_path,
                                      width, height)
    logging.info(f"  {len(summaries)} tracks to consider")
    for tid in sorted(summaries.keys()):
        s = summaries[tid]
        logging.info(f"  T{tid:>3}: frames {s['first_frame']:>4}-"
                     f"{s['last_frame']:>4}  ({s['lifespan']:>4}f, "
                     f"{s['n_detections']:>4}det)  exit={s['exit_edge']:<6} "
                     f"entry={s['entry_edge']:<6}")

    logging.info("")
    logging.info(f"Greedy merging (N_target={n_target}, "
                 f"auto>={auto_thresh}, review>={review_thresh})...")
    groups, auto_merges, needs_review = merge_groups(
        summaries, n_target=n_target,
        auto_thresh=auto_thresh, review_thresh=review_thresh)

    logging.info(f"  {len(auto_merges)} automatic merges")
    for m in auto_merges:
        d = m['detail']
        logging.info(f"    {m['merged']:>3} -> {m['kept']:>3}  "
                     f"score={m['score']:.2f}  "
                     f"(app {d['appearance']:.2f}, sp {d['spatial']:.2f} "
                     f"d={d['d_px']:.0f}px, "
                     f"edges {d['a_exit']}->{d['b_entry']})")
    logging.info(f"  {len(needs_review)} pairs need manual review:")
    for r in needs_review[:20]:
        d = r['detail']
        logging.info(f"    {r['a']:>3} ?-> {r['b']:>3}  "
                     f"score={r['score']:.2f}  "
                     f"(app {d['appearance']:.2f}, sp {d['spatial']:.2f} "
                     f"d={d['d_px']:.0f}px, "
                     f"edges {d['a_exit']}->{d['b_entry']})")

    # Build id_to_final mapping: original tid -> final group id (lowest tid in group)
    id_to_final = {}
    for group_id, g in groups.items():
        final_id = min(g['merged_from'])
        for orig in g['merged_from']:
            id_to_final[orig] = final_id

    final_count = len(set(id_to_final.values()))
    logging.info("")
    logging.info(f"==== Merge result ====")
    logging.info(f"Original tracks: {len(track_count)}")
    logging.info(f"After length filter: {len(kept_ids)}")
    logging.info(f"After merging: {final_count} distinct persons")

    if output_mp4 is None:
        output_mp4 = str(tracks_json.with_name(tracks_json.stem.replace('_tracks', '') + '_merged.mp4'))
    if output_json is None:
        output_json = str(tracks_json.with_name(tracks_json.stem.replace('_tracks', '') + '_merged.json'))

    output_mp4 = Path(output_mp4)
    output_json = Path(output_json)

    logging.info(f"Rendering merged MP4 -> {output_mp4}")
    render_merged_video(per_frame_dets, id_to_final, video_path,
                        output_mp4, width, height, fps)

    # Save merged result
    with open(output_json, 'w') as f:
        json.dump({
            'source_tracks_json': str(tracks_json),
            'video': str(video_path),
            'n_target': n_target,
            'auto_thresh': auto_thresh,
            'review_thresh': review_thresh,
            'id_to_final': {str(k): int(v) for k, v in id_to_final.items()},
            'auto_merges': [
                {'kept': m['kept'], 'merged': m['merged'],
                 'score': m['score'], 'detail': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                  for k, v in m['detail'].items()}}
                for m in auto_merges
            ],
            'needs_review': [
                {'a': r['a'], 'b': r['b'],
                 'score': r['score'],
                 'detail': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in r['detail'].items()}}
                for r in needs_review
            ],
            'final_groups': [
                {'final_id': min(g['merged_from']),
                 'orig_ids': g['merged_from'],
                 'first_frame': g['first_frame'],
                 'last_frame': g['last_frame']}
                for g in groups.values()
            ],
        }, f, indent=2)
    logging.info(f"Merged JSON saved -> {output_json}")

    return {
        'id_to_final': id_to_final,
        'auto_merges': auto_merges,
        'needs_review': needs_review,
        'final_count': final_count,
        'output_mp4': str(output_mp4),
        'output_json': str(output_json),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--tracks_json', required=True,
                        help="Tracks JSON produced by botsort_poc.py.")
    parser.add_argument('-o', '--output_mp4', default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('-n', '--n_target', type=int, default=8,
                        help="Expected number of distinct persons (closed-world cap)")
    parser.add_argument('--auto_thresh', type=float, default=1.5,
                        help="Pairs >= this score merged automatically. Default 1.5 = no auto merge "
                             "(every re-entry goes to manual review). Lower (e.g. 0.85) for very rare safe auto merges.")
    parser.add_argument('--review_thresh', type=float, default=0.20,
                        help="Pairs >= this score are surfaced to the manual review UI")
    parser.add_argument('--reid_model', default='lmbn_n_market.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-half', action='store_true')
    parser.add_argument('--min_track_length', type=int, default=5,
                        help="Drop tracks with fewer than N detections. Keep low (5) so brief "
                             "returning tracks still get a chance to be matched in manual review.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    merge_func(
        tracks_json=args.tracks_json,
        output_mp4=args.output_mp4,
        output_json=args.output_json,
        n_target=args.n_target,
        auto_thresh=args.auto_thresh,
        review_thresh=args.review_thresh,
        reid_model=args.reid_model,
        device=args.device,
        half=not args.no_half,
        min_track_length=args.min_track_length,
    )


if __name__ == '__main__':
    main()
