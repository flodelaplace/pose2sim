#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###################################################################
    ## Split tracks at silent ID swaps (mid-track appearance shifts) ##
    ###################################################################

    BoT-SORT's frame-to-frame matching can transfer an ID from person A
    to person B during a crossing without any physical jump (the two
    persons swap places at adjacent pixel coordinates) and without a
    black frame (the swap is just an association mistake). The track
    keeps its ID but the bbox now follows person B instead of A.

    This step scans each track in a SLIDING WINDOW over torso color
    histograms and detects when the appearance shifts past a similarity
    threshold for several consecutive samples. Such a sustained shift is
    the signature of a silent ID swap. We split the track at the first
    frame of the new appearance segment so that the post-swap segment
    becomes a separate track ID (which the merger + manual review will
    then re-attribute correctly).

    Why a SLIDING WINDOW (not a single-frame check):
      - A person turning around briefly changes their torso histogram.
      - A single low-similarity frame is therefore not enough — the
        change must persist over several samples to count.

    Usage:
      botsort_split_appearance -t <tracks.json>
      botsort_split_appearance -t <tracks.json> --window 8 --sim_threshold 0.4
      from Pose2Sim.Tracking import botsort_split_appearance
      botsort_split_appearance.split_appearance(tracks_json=r'<path>')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

from Pose2Sim.Tracking.botsort_merger import (
    extract_torso_hist, histogram_similarity,
)
from Pose2Sim.Tracking.botsort_split_jumps import (
    detections_by_id, apply_splits,
)


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
DEFAULT_SAMPLE_EVERY = 3        # extract a torso histogram every N frames
DEFAULT_WINDOW = 8              # nb of samples on each side of a candidate split
DEFAULT_SIM_THRESHOLD = 0.40    # sliding-window similarity below this = shift
DEFAULT_MIN_TRACK_LEN = 30      # don't bother splitting very short tracks
DEFAULT_MIN_GAP_BETWEEN_SPLITS = 30  # consecutive splits must be at least Nf apart


def extract_all_track_histograms(by_id, video_path, sample_every,
                                 min_track_len):
    '''
    Compute torso HSV histograms for all tracks in a SINGLE sequential
    pass through the video. About 10x faster than seeking per track,
    because random-access seeks in H.264 video cost ~30-100ms each
    (decoder has to walk from the nearest keyframe).

    Returns dict { tid: [(frame_idx, hist), ...] }, sorted by frame.
    '''
    # Determine which (frame, tid) pairs we need a histogram for
    needed_per_frame = defaultdict(list)   # frame_idx -> [(tid, bbox), ...]
    track_eligible = {}                    # tid -> True if track is long enough
    for tid, items in by_id.items():
        if len(items) < min_track_len:
            track_eligible[tid] = False
            continue
        track_eligible[tid] = True
        sampled = items[::sample_every]
        for f_idx, bbox in sampled:
            needed_per_frame[f_idx].append((tid, bbox))

    if not needed_per_frame:
        return {}
    last_needed_frame = max(needed_per_frame.keys())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    histograms_by_tid = defaultdict(list)
    f_idx = 0
    while f_idx <= last_needed_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx in needed_per_frame:
            for tid, bbox in needed_per_frame[f_idx]:
                h = extract_torso_hist(frame, bbox)
                if h is not None:
                    histograms_by_tid[tid].append((f_idx, h))
        f_idx += 1

    cap.release()
    return histograms_by_tid


def detect_appearance_discontinuities(histograms, window, sim_threshold,
                                      min_gap_between_splits):
    '''
    For each interior position i (window <= i < N-window), compare the
    mean histogram of samples [i-window, i) against that of [i, i+window).
    Mark i as a split candidate if their Bhattacharyya similarity is
    below threshold. Deduplicate adjacent candidates by keeping only the
    minimum-similarity one within each cluster.

    Returns list of dicts with split frame, before/after similarities.
    '''
    n = len(histograms)
    if n < 2 * window + 1:
        return []
    # Pre-compute cumulative sums for fast window means
    cum = np.zeros((n + 1, histograms[0][1].shape[0]), dtype=np.float64)
    for i, (_, h) in enumerate(histograms):
        cum[i + 1] = cum[i] + h

    def window_mean(lo, hi):  # mean of histograms[lo:hi]
        m = (cum[hi] - cum[lo]) / max(1, hi - lo)
        s = float(m.sum())
        if s > 1e-9:
            m = m / s
        return m

    cluster_best = []  # list of dicts representing local minima of similarity
    for i in range(window, n - window):
        before = window_mean(i - window, i)
        after = window_mean(i, i + window)
        sim = histogram_similarity(before, after)
        if sim < sim_threshold:
            cluster_best.append({
                'sample_idx': i,
                'split_frame': histograms[i][0],
                'sim': sim,
                'sim_before_self': histogram_similarity(
                    window_mean(i - 2 * window, i - window) if i >= 2 * window else before,
                    before
                ),
            })

    # Deduplicate clusters of close-by candidates (keep the lowest sim)
    if not cluster_best:
        return []
    cluster_best.sort(key=lambda x: x['split_frame'])
    deduped = [cluster_best[0]]
    for c in cluster_best[1:]:
        last = deduped[-1]
        if c['split_frame'] - last['split_frame'] < min_gap_between_splits:
            if c['sim'] < last['sim']:
                deduped[-1] = c
        else:
            deduped.append(c)
    return deduped


def split_appearance(tracks_json, output_json=None,
                     sample_every=DEFAULT_SAMPLE_EVERY,
                     window=DEFAULT_WINDOW,
                     sim_threshold=DEFAULT_SIM_THRESHOLD,
                     min_track_len=DEFAULT_MIN_TRACK_LEN,
                     min_gap_between_splits=DEFAULT_MIN_GAP_BETWEEN_SPLITS):
    tracks_json = Path(tracks_json)
    with open(tracks_json, 'r') as f:
        data = json.load(f)
    video_path = Path(data['video'])
    per_frame_dets = data['frames']
    fps = float(data['fps'])

    by_id = detections_by_id(per_frame_dets)
    if not by_id:
        logging.info("No detections — nothing to split.")
        return {}

    next_id_start = max(by_id.keys()) + 1
    splits_by_tid = {}

    # Single sequential pass over the video to extract histograms for
    # every (track, sampled frame) pair at once.
    logging.info(f"Extracting torso histograms ({len(by_id)} tracks)...")
    histograms_all = extract_all_track_histograms(
        by_id, video_path, sample_every, min_track_len)
    logging.info(f"  histograms ready for {len(histograms_all)} tracks")

    for tid, items in by_id.items():
        if len(items) < min_track_len:
            continue
        histograms = histograms_all.get(tid, [])
        if len(histograms) < 2 * window + 1:
            continue
        discs = detect_appearance_discontinuities(
            histograms, window, sim_threshold, min_gap_between_splits)
        # Convert discontinuities to the (split_frame,) format expected by apply_splits
        if discs:
            splits_by_tid[tid] = [
                {'split_frame': d['split_frame'],
                 'sim_before_after': d['sim'],
                 'prev_frame': histograms[max(0, d['sample_idx']-1)][0],
                 'sample_idx': d['sample_idx']}
                for d in discs
            ]

    n_splits = sum(len(v) for v in splits_by_tid.values())
    logging.info(f"Scanned {len(by_id)} tracks (min len {min_track_len}), "
                 f"found {n_splits} appearance shifts in {len(splits_by_tid)} tracks")
    for tid in sorted(splits_by_tid.keys()):
        for sp in splits_by_tid[tid]:
            logging.info(f"  P{tid}: appearance shift @ frame {sp['split_frame']}  "
                         f"(window similarity dropped to {sp['sim_before_after']:.2f})")

    last_id = apply_splits(per_frame_dets, splits_by_tid, next_id_start)
    new_ids_added = last_id - next_id_start

    data['frames'] = per_frame_dets
    data['split_appearance'] = {
        'source_tracks_json': str(tracks_json),
        'sample_every': sample_every,
        'window': window,
        'sim_threshold': sim_threshold,
        'min_track_len': min_track_len,
        'min_gap_between_splits': min_gap_between_splits,
        'n_splits': n_splits,
        'n_new_ids': new_ids_added,
        'splits_by_tid': {str(tid): sp_list for tid, sp_list in splits_by_tid.items()},
    }

    if output_json is None:
        output_json = str(tracks_json.with_name(
            tracks_json.stem + '_appsplit.json'))
    with open(output_json, 'w') as f:
        json.dump(data, f)
    logging.info(f"Appearance-split tracks JSON saved -> {output_json}")
    logging.info(f"  Created {new_ids_added} new track IDs")
    return {
        'output_json': output_json,
        'n_splits': n_splits,
        'n_new_ids': new_ids_added,
        'splits_by_tid': splits_by_tid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--tracks_json', required=True)
    parser.add_argument('-o', '--output_json', default=None)
    parser.add_argument('--sample_every', type=int, default=DEFAULT_SAMPLE_EVERY,
                        help="Extract one histogram every N frames within a track.")
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                        help="Samples on each side of a candidate split.")
    parser.add_argument('--sim_threshold', type=float, default=DEFAULT_SIM_THRESHOLD,
                        help="Bhattacharyya similarity below which a sustained shift "
                             "is considered a swap.")
    parser.add_argument('--min_track_len', type=int, default=DEFAULT_MIN_TRACK_LEN)
    parser.add_argument('--min_gap_between_splits', type=int,
                        default=DEFAULT_MIN_GAP_BETWEEN_SPLITS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    split_appearance(
        tracks_json=args.tracks_json,
        output_json=args.output_json,
        sample_every=args.sample_every,
        window=args.window,
        sim_threshold=args.sim_threshold,
        min_track_len=args.min_track_len,
        min_gap_between_splits=args.min_gap_between_splits,
    )


if __name__ == '__main__':
    main()
