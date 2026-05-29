#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #################################################################
    ## Fix ID swaps that happen during crossings (esp. black frames) ##
    #################################################################

    BoT-SORT's Kalman + appearance association can flip the IDs of two
    persons when they cross in the image and the per-frame tracker
    momentarily loses both detections (typically a 1-3 frame black gap
    coming from the hub-side resync, listed in `<video>.dropped.json`).

    What this tool does:
      1. Read the camera's `.dropped.json` sidecar (if present) to get the
         set of "risky frames" where a black frame happened.
      2. For each risky frame F, look at every pair of tracks that
         co-existed both *just before* and *just after* F.
      3. Verify whether their IDs got swapped using two independent
         signals:
           - APPEARANCE (Bhattacharyya similarity of torso HSV histograms):
             does the *before* histogram of A match the *after* histogram
             of B (and vice versa)?
           - TRAJECTORY (linear extrapolation of the bbox center): if A's
             motion continued through F, would A's post-event detection be
             where it actually is, or would it be at B's post-event spot?
      4. Decision rule:
           - Both signals say "yes swap" with high confidence  -> AUTO swap
           - Only one of the two signals supports a swap        -> REVIEW
           - Neither supports                                    -> ignore

    Apply the auto swaps to the tracks JSON, log the review cases, save
    a `_fixed_tracks.json` that becomes the input of the merger.

    This is the ONLY case where the pipeline auto-modifies BoT-SORT's
    output without asking the user: the gap is tiny (<5 frames), the
    spatial constraint is strong (crossings are short events), and we
    require two independent signals to agree.

    Usage:
      botsort_swap_fixer -t <tracks.json>
      botsort_swap_fixer -t <tracks.json> --dropped_json <sidecar.json>
      from Pose2Sim.Tracking import botsort_swap_fixer
      botsort_swap_fixer.fix_swaps(tracks_json=r'<path>')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import re
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

from Pose2Sim.Tracking.botsort_merger import (
    extract_torso_hist, histogram_similarity, bbox_center,
)


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
WINDOW_FRAMES = 8                # nb of frames before/after F to sample
MIN_DETECTIONS_PER_WINDOW = 2    # need at least this many to compare
CROSSING_DISTANCE_PX = 250.0     # bboxes within this distance = crossing
TRAJECTORY_FIT_FRAMES = 6        # nb of recent positions used for velocity
TRAJECTORY_MATCH_PX = 150.0      # predicted-vs-observed distance for "match"

# Decision thresholds (both must hold for AUTO swap)
AUTO_COLOR_THRESH = 0.55         # cross-match histogram similarity required
AUTO_TRAJECTORY_REQUIRED = True

# Review thresholds (lower bar than AUTO, surfaces ambiguous cases)
REVIEW_COLOR_THRESH = 0.40


## FUNCTIONS
def find_dropped_sidecar(video_path):
    '''Find the .dropped.json sidecar that pairs with a video.'''
    video_path = Path(video_path)
    candidate = video_path.with_suffix('.dropped.json')
    if candidate.is_file():
        return candidate
    # Fallback: same basename in the same folder
    stem = video_path.stem
    sib = video_path.parent / f"{stem}.dropped.json"
    if sib.is_file():
        return sib
    return None


def load_dropped_frames(sidecar_path):
    '''Return a set of frame indices that are black for this camera.'''
    if sidecar_path is None or not Path(sidecar_path).is_file():
        return set()
    with open(sidecar_path, 'r') as f:
        data = json.load(f)
    raw = data.get('dropped_frame_indices') or data.get('dropped') or []
    return set(int(x) for x in raw)


def cluster_risky_frames(dropped_set, max_gap=2):
    '''
    Group consecutive (or nearly consecutive) black frames into a single
    "event" centered on the gap, so we look at the boundaries once per
    contiguous gap rather than once per individual black frame.

    Returns a list of (event_start_frame, event_end_frame) tuples.
    '''
    if not dropped_set:
        return []
    sorted_f = sorted(dropped_set)
    events = []
    cur_start = cur_end = sorted_f[0]
    for f in sorted_f[1:]:
        if f - cur_end <= max_gap:
            cur_end = f
        else:
            events.append((cur_start, cur_end))
            cur_start = cur_end = f
    events.append((cur_start, cur_end))
    return events


def pool_window_histogram(cap, items_in_window):
    '''Mean torso histogram across a list of (frame_idx, bbox) detections.
    Kept for callers that already do their own seeking; new code should
    use `pool_window_histogram_cached` after pre-computing the cache.'''
    hists = []
    for f_idx, bbox in items_in_window:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h = extract_torso_hist(frame, bbox)
        if h is not None:
            hists.append(h)
    if not hists:
        return None
    m = np.mean(np.stack(hists, axis=0), axis=0)
    s = float(np.sum(m))
    if s > 1e-9:
        m /= s
    return m


def pool_window_histogram_cached(items_in_window, hist_cache):
    '''Same as pool_window_histogram but reads pre-computed histograms
    from a `(frame_idx, tuple(bbox)) -> hist` cache. ~10x faster on
    long videos because no per-detection seek is needed.'''
    hists = []
    for f_idx, bbox in items_in_window:
        key = (f_idx, tuple(bbox))
        h = hist_cache.get(key)
        if h is not None:
            hists.append(h)
    if not hists:
        return None
    m = np.mean(np.stack(hists, axis=0), axis=0)
    s = float(np.sum(m))
    if s > 1e-9:
        m /= s
    return m


def precompute_event_histograms(per_frame_dets, events, video_path, window):
    '''Sequential single-pass video read that builds a histogram for every
    detection sitting in [event-window, event+window] for any of the
    listed events. The cache is keyed by `(frame_idx, tuple(bbox))` so
    later label swaps don't invalidate it.'''
    needed_frames = set()
    for event_start, event_end in events:
        lo = max(0, event_start - window)
        hi = min(len(per_frame_dets) - 1, event_end + window)
        for f in range(lo, hi + 1):
            needed_frames.add(f)
    if not needed_frames:
        return {}
    last_needed = max(needed_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    hist_cache = {}
    f_idx = 0
    while f_idx <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx in needed_frames:
            for r in per_frame_dets[f_idx]:
                bbox = r['bbox']
                key = (f_idx, tuple(bbox))
                if key in hist_cache:
                    continue
                h = extract_torso_hist(frame, bbox)
                if h is not None:
                    hist_cache[key] = h
        f_idx += 1
    cap.release()
    return hist_cache


def fit_velocity(positions):
    '''Linear least-squares velocity from a list of (frame_idx, x, y).'''
    if len(positions) < 2:
        return (0.0, 0.0)
    t = np.array([p[0] for p in positions], dtype=float)
    x = np.array([p[1] for p in positions], dtype=float)
    y = np.array([p[2] for p in positions], dtype=float)
    dt_span = t.max() - t.min()
    if dt_span < 1e-3:
        return (0.0, 0.0)
    # Slope via simple end-to-end average (robust enough for short windows)
    vx = (x[-1] - x[0]) / dt_span
    vy = (y[-1] - y[0]) / dt_span
    return (vx, vy)


def collect_detections_by_id(per_frame_dets):
    '''Returns dict tid -> sorted list of (frame_idx, bbox).'''
    by_id = defaultdict(list)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    return {tid: sorted(items, key=lambda x: x[0]) for tid, items in by_id.items()}


def items_in_window(track_items, lo, hi):
    '''Detections of a track in [lo, hi] inclusive.'''
    return [it for it in track_items if lo <= it[0] <= hi]


def detect_swap_at_event(event, by_id, hist_cache, fps,
                         window=WINDOW_FRAMES,
                         min_dets=MIN_DETECTIONS_PER_WINDOW,
                         crossing_px=CROSSING_DISTANCE_PX,
                         fit_frames=TRAJECTORY_FIT_FRAMES,
                         traj_match_px=TRAJECTORY_MATCH_PX,
                         auto_color_thresh=AUTO_COLOR_THRESH,
                         review_color_thresh=REVIEW_COLOR_THRESH):
    '''
    Examine all pairs of tracks present in both windows around a risky
    event. Return list of detected swaps with confidence label.

    Each result is a dict:
      {'a': tid_a, 'b': tid_b, 'event': (start, end),
       'apply_frame': fpost,           # first frame >= event.end+1 where both tracks reappear
       'color': (sim_aa_bb, sim_ab_ba),   # (A_before<->A_after, A_before<->B_after / mirrored)
       'trajectory': dict,
       'confidence': 'auto' | 'review' | None,
       'reason': str}
    '''
    event_start, event_end = event
    pre_lo, pre_hi = event_start - window, event_start - 1
    post_lo, post_hi = event_end + 1, event_end + window
    if pre_lo < 0 or post_hi < 0:
        return []

    # Track IDs that have detections in both windows
    active_ids = [tid for tid, items in by_id.items()
                  if (len(items_in_window(items, pre_lo, pre_hi)) >= min_dets
                      and len(items_in_window(items, post_lo, post_hi)) >= min_dets)]
    if len(active_ids) < 2:
        return []

    # Last bbox center before the event for each candidate
    last_pre_center = {}
    for tid in active_ids:
        pre_items = items_in_window(by_id[tid], pre_lo, pre_hi)
        last_pre_center[tid] = bbox_center(pre_items[-1][1])

    results = []
    # Only consider pairs that were spatially close just before F (= crossing)
    for i, a_id in enumerate(active_ids):
        for b_id in active_ids[i + 1:]:
            d_pre = float(np.hypot(last_pre_center[a_id][0] - last_pre_center[b_id][0],
                                   last_pre_center[a_id][1] - last_pre_center[b_id][1]))
            if d_pre > crossing_px:
                continue

            pre_a = items_in_window(by_id[a_id], pre_lo, pre_hi)
            post_a = items_in_window(by_id[a_id], post_lo, post_hi)
            pre_b = items_in_window(by_id[b_id], pre_lo, pre_hi)
            post_b = items_in_window(by_id[b_id], post_lo, post_hi)

            # ---- Color cross-match ----
            h_pre_a = pool_window_histogram_cached(pre_a, hist_cache)
            h_post_a = pool_window_histogram_cached(post_a, hist_cache)
            h_pre_b = pool_window_histogram_cached(pre_b, hist_cache)
            h_post_b = pool_window_histogram_cached(post_b, hist_cache)
            if any(h is None for h in (h_pre_a, h_post_a, h_pre_b, h_post_b)):
                continue
            sim_self_a = histogram_similarity(h_pre_a, h_post_a)
            sim_self_b = histogram_similarity(h_pre_b, h_post_b)
            sim_cross_ab = histogram_similarity(h_pre_a, h_post_b)
            sim_cross_ba = histogram_similarity(h_pre_b, h_post_a)
            color_supports_swap = (
                sim_cross_ab > sim_self_a and sim_cross_ba > sim_self_b
                and min(sim_cross_ab, sim_cross_ba) >= review_color_thresh
            )
            color_strong = (
                sim_cross_ab > sim_self_a + 0.05 and sim_cross_ba > sim_self_b + 0.05
                and min(sim_cross_ab, sim_cross_ba) >= auto_color_thresh
            )

            # ---- Trajectory cross-match ----
            # Use last fit_frames detections of each track to fit velocity
            pre_a_pos = [(f, *bbox_center(b)) for f, b in pre_a[-fit_frames:]]
            pre_b_pos = [(f, *bbox_center(b)) for f, b in pre_b[-fit_frames:]]
            vel_a = fit_velocity(pre_a_pos)
            vel_b = fit_velocity(pre_b_pos)
            # Project A & B forward to the first post-event frame
            apply_frame = post_a[0][0]
            apply_frame_b = post_b[0][0]
            anchor_a = pre_a_pos[-1]
            anchor_b = pre_b_pos[-1]
            pred_a = (anchor_a[1] + vel_a[0] * (apply_frame - anchor_a[0]),
                      anchor_a[2] + vel_a[1] * (apply_frame - anchor_a[0]))
            pred_b = (anchor_b[1] + vel_b[0] * (apply_frame_b - anchor_b[0]),
                      anchor_b[2] + vel_b[1] * (apply_frame_b - anchor_b[0]))
            obs_a = bbox_center(post_a[0][1])
            obs_b = bbox_center(post_b[0][1])

            d_pred_a_to_obs_a = float(np.hypot(pred_a[0] - obs_a[0], pred_a[1] - obs_a[1]))
            d_pred_a_to_obs_b = float(np.hypot(pred_a[0] - obs_b[0], pred_a[1] - obs_b[1]))
            d_pred_b_to_obs_b = float(np.hypot(pred_b[0] - obs_b[0], pred_b[1] - obs_b[1]))
            d_pred_b_to_obs_a = float(np.hypot(pred_b[0] - obs_a[0], pred_b[1] - obs_a[1]))

            traj_supports_swap = (
                d_pred_a_to_obs_b < d_pred_a_to_obs_a
                and d_pred_b_to_obs_a < d_pred_b_to_obs_b
            )
            traj_strong = (
                traj_supports_swap
                and d_pred_a_to_obs_b < traj_match_px
                and d_pred_b_to_obs_a < traj_match_px
            )

            # ---- Decision ----
            if color_strong and traj_strong:
                conf = 'auto'
                reason = 'both color and trajectory support swap'
            elif color_supports_swap and traj_supports_swap:
                conf = 'review'
                reason = 'weak agreement (one signal not strong)'
            elif color_supports_swap or traj_supports_swap:
                conf = 'review'
                reason = 'only one signal supports swap'
            else:
                conf = None
                reason = 'neither signal supports swap'

            if conf is None:
                continue

            results.append({
                'a': a_id, 'b': b_id,
                'event_start': event_start, 'event_end': event_end,
                'apply_frame': apply_frame,
                'pre_dist_px': d_pre,
                'color': {
                    'self_a': sim_self_a, 'self_b': sim_self_b,
                    'cross_ab': sim_cross_ab, 'cross_ba': sim_cross_ba,
                },
                'trajectory': {
                    'd_pred_a_obs_a': d_pred_a_to_obs_a,
                    'd_pred_a_obs_b': d_pred_a_to_obs_b,
                    'd_pred_b_obs_b': d_pred_b_to_obs_b,
                    'd_pred_b_obs_a': d_pred_b_to_obs_a,
                    'support_swap': traj_supports_swap,
                    'strong': traj_strong,
                },
                'confidence': conf,
                'reason': reason,
            })
    return results


def apply_swap(per_frame_dets, a_id, b_id, apply_from_frame):
    '''
    Relabel detections from `apply_from_frame` onwards: every id==a_id
    becomes b_id and vice versa.
    '''
    for f_idx in range(apply_from_frame, len(per_frame_dets)):
        for rec in per_frame_dets[f_idx]:
            if rec['id'] == a_id:
                rec['id'] = b_id
            elif rec['id'] == b_id:
                rec['id'] = a_id


def fix_swaps(tracks_json, output_json=None, dropped_json=None,
              window=WINDOW_FRAMES,
              auto_color_thresh=AUTO_COLOR_THRESH,
              review_color_thresh=REVIEW_COLOR_THRESH,
              crossing_px=CROSSING_DISTANCE_PX):
    '''Detect and auto-correct ID swaps. See module docstring.'''
    tracks_json = Path(tracks_json)
    with open(tracks_json, 'r') as f:
        data = json.load(f)
    video_path = Path(data['video'])
    per_frame_dets = data['frames']
    fps = float(data['fps'])

    # Locate dropped sidecar
    if dropped_json is None:
        dropped_json = find_dropped_sidecar(video_path)
        if dropped_json is None:
            logging.warning(f"No .dropped.json sidecar found next to "
                            f"{video_path.name}; will scan ALL track gaps "
                            f"instead (slower, less targeted)")

    dropped_set = load_dropped_frames(dropped_json) if dropped_json else set()
    events = cluster_risky_frames(dropped_set)
    logging.info(f"Loaded {len(dropped_set)} dropped frames -> "
                 f"{len(events)} risky events to scan")

    # Pre-compute torso histograms for every detection within ±window of any
    # event, in a single sequential video pass. Keyed by (frame, tuple(bbox))
    # so swap-induced label changes don't invalidate cached entries.
    logging.info("Pre-computing torso histograms in a single video pass...")
    hist_cache = precompute_event_histograms(per_frame_dets, events,
                                             video_path, window)
    logging.info(f"  {len(hist_cache)} histograms cached")

    auto_log = []
    review_log = []

    # Re-collect by_id after each auto swap, because the labels change.
    for event in events:
        by_id = collect_detections_by_id(per_frame_dets)
        candidates = detect_swap_at_event(
            event, by_id, hist_cache, fps,
            window=window,
            crossing_px=crossing_px,
            auto_color_thresh=auto_color_thresh,
            review_color_thresh=review_color_thresh,
        )
        for cand in candidates:
            if cand['confidence'] == 'auto':
                auto_log.append(cand)
                apply_swap(per_frame_dets, cand['a'], cand['b'],
                           cand['apply_frame'])
                logging.info(
                    f"  AUTO swap @event[{event[0]}-{event[1]}] "
                    f"P{cand['a']}<->P{cand['b']} from frame {cand['apply_frame']} "
                    f"(color cross {cand['color']['cross_ab']:.2f}/"
                    f"{cand['color']['cross_ba']:.2f} > self "
                    f"{cand['color']['self_a']:.2f}/{cand['color']['self_b']:.2f}; "
                    f"traj OK)"
                )
            else:
                review_log.append(cand)
                logging.info(
                    f"  REVIEW swap @event[{event[0]}-{event[1]}] "
                    f"P{cand['a']}<->P{cand['b']} -- {cand['reason']}: "
                    f"color cross {cand['color']['cross_ab']:.2f}/"
                    f"{cand['color']['cross_ba']:.2f} vs self "
                    f"{cand['color']['self_a']:.2f}/{cand['color']['self_b']:.2f}; "
                    f"traj {'support' if cand['trajectory']['support_swap'] else 'reject'}"
                )

    logging.info("")
    logging.info(f"==== Swap fixer summary ====")
    logging.info(f"  events scanned       : {len(events)}")
    logging.info(f"  AUTO swaps applied   : {len(auto_log)}")
    logging.info(f"  REVIEW pairs flagged : {len(review_log)}")

    if output_json is None:
        output_json = str(tracks_json.with_name(
            tracks_json.stem + '_fixed.json'))
    output_json = Path(output_json)
    data['frames'] = per_frame_dets
    data['swap_fixer'] = {
        'source_tracks_json': str(tracks_json),
        'dropped_json': str(dropped_json) if dropped_json else None,
        'window_frames': window,
        'auto_color_thresh': auto_color_thresh,
        'review_color_thresh': review_color_thresh,
        'crossing_px': crossing_px,
        'auto_swaps': [
            {k: (v if not isinstance(v, dict) else {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                                    for kk, vv in v.items()})
             for k, v in s.items()}
            for s in auto_log
        ],
        'review_swaps': [
            {k: (v if not isinstance(v, dict) else {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                                    for kk, vv in v.items()})
             for k, v in s.items()}
            for s in review_log
        ],
    }
    with open(output_json, 'w') as f:
        json.dump(data, f)
    logging.info(f"Fixed tracks JSON saved -> {output_json}")
    return {
        'output_json': str(output_json),
        'auto_swaps': auto_log,
        'review_swaps': review_log,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--tracks_json', required=True)
    parser.add_argument('-o', '--output_json', default=None)
    parser.add_argument('--dropped_json', default=None,
                        help="Path to .dropped.json sidecar (auto-detected if omitted)")
    parser.add_argument('--window', type=int, default=WINDOW_FRAMES)
    parser.add_argument('--auto_color_thresh', type=float, default=AUTO_COLOR_THRESH)
    parser.add_argument('--review_color_thresh', type=float, default=REVIEW_COLOR_THRESH)
    parser.add_argument('--crossing_px', type=float, default=CROSSING_DISTANCE_PX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    fix_swaps(
        tracks_json=args.tracks_json,
        output_json=args.output_json,
        dropped_json=args.dropped_json,
        window=args.window,
        auto_color_thresh=args.auto_color_thresh,
        review_color_thresh=args.review_color_thresh,
        crossing_px=args.crossing_px,
    )


if __name__ == '__main__':
    main()
