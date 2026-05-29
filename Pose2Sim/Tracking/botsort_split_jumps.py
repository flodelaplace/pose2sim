#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #################################################################
    ## Split tracks at physically impossible bbox jumps             ##
    #################################################################

    BoT-SORT's frame-to-frame association is based on Kalman-predicted
    positions and IoU. When one person leaves and another appears nearby
    a few frames later, BoT-SORT can keep the leaver's ID and transfer
    it to the newcomer — silently producing a track whose bbox CENTER
    teleports across the frame. A human cannot move that fast, so the
    track is by definition wrong from the jump onwards.

    This step scans each track for frame-to-frame center jumps that
    exceed what a person can physically travel at the current frame rate,
    and SPLITS the track at every such jump. The post-jump segment gets
    a fresh track ID, so the merger + manual review will then treat it
    as a separate person (and the user can decide whether to re-associate
    it with someone or call it a new person).

    Default thresholds (for 30 fps mocap, person ~5-10 m from camera):
      - max speed = 40 px/frame  (~10 m/s, covers running)
      - min absolute jump = 60 px (don't split on detection noise)

    Usage:
      botsort_split_jumps -t <tracks.json>
      botsort_split_jumps -t <tracks.json> --max_speed 40 --min_jump 60
      from Pose2Sim.Tracking import botsort_split_jumps
      botsort_split_jumps.split_func(tracks_json=r'<path>')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import math
import logging
import argparse
from pathlib import Path
from collections import defaultdict


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
DEFAULT_MAX_SPEED_PX_PER_FRAME = 40.0   # ~10 m/s at typical mocap distance
DEFAULT_MIN_ABSOLUTE_JUMP_PX = 60.0     # avoid splitting on detection noise
DEFAULT_MAX_GAP_FRAMES = 60             # ignore very long gaps in same track
                                        # (those should have been killed by
                                        # BoT-SORT's track_buffer; if they
                                        # weren't, the merger handles them).


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def detections_by_id(per_frame_dets):
    by_id = defaultdict(list)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    return {tid: sorted(items, key=lambda x: x[0]) for tid, items in by_id.items()}


def find_split_frames(items, max_speed, min_jump, max_gap):
    '''
    Walk a single track's detections and return the list of frame indices
    where the track should be split (the listed frame is the FIRST frame
    of the post-split segment).
    '''
    splits = []
    if len(items) < 2:
        return splits
    for i in range(1, len(items)):
        f_prev, bbox_prev = items[i - 1]
        f_cur, bbox_cur = items[i]
        df = f_cur - f_prev
        if df <= 0 or df > max_gap:
            # If gap is huge, it means BoT-SORT kept the track alive across
            # an absence; the merger handles those, skip here.
            continue
        cx_p, cy_p = bbox_center(bbox_prev)
        cx_c, cy_c = bbox_center(bbox_cur)
        dist = math.hypot(cx_c - cx_p, cy_c - cy_p)
        speed = dist / df
        if speed > max_speed and dist > min_jump:
            splits.append({
                'split_frame': f_cur, 'prev_frame': f_prev,
                'dist_px': dist, 'frame_gap': df, 'speed_px_f': speed,
            })
    return splits


def apply_splits(per_frame_dets, splits_by_tid, next_id_start):
    '''
    Apply the splits to the per-frame detections (in-place). For each
    track that has splits, we rename detections at frame >= split_frame
    to a new id (one new id per split point).

    Returns the next available id after all renames.
    '''
    next_id = next_id_start
    # Build a dict tid -> list of (split_frame, new_id), sorted
    renames = {}
    for tid, sp_list in splits_by_tid.items():
        renames_for_tid = []
        for sp in sp_list:
            renames_for_tid.append((sp['split_frame'], next_id))
            next_id += 1
        renames_for_tid.sort()
        renames[tid] = renames_for_tid

    for f_idx in range(len(per_frame_dets)):
        for rec in per_frame_dets[f_idx]:
            tid = rec['id']
            if tid not in renames:
                continue
            new_id = tid
            for split_frame, replacement_id in renames[tid]:
                if f_idx >= split_frame:
                    new_id = replacement_id
                else:
                    break  # later splits are ahead, can't apply yet
            if new_id != tid:
                rec['id'] = new_id
    return next_id


def split_func(tracks_json, output_json=None,
               max_speed=DEFAULT_MAX_SPEED_PX_PER_FRAME,
               min_jump=DEFAULT_MIN_ABSOLUTE_JUMP_PX,
               max_gap=DEFAULT_MAX_GAP_FRAMES):
    tracks_json = Path(tracks_json)
    with open(tracks_json, 'r') as f:
        data = json.load(f)
    per_frame_dets = data['frames']
    fps = float(data['fps'])

    by_id = detections_by_id(per_frame_dets)
    if not by_id:
        logging.info("No detections — nothing to split.")
        return {}

    next_id_start = max(by_id.keys()) + 1

    splits_by_tid = {}
    for tid, items in by_id.items():
        sp_list = find_split_frames(items, max_speed, min_jump, max_gap)
        if sp_list:
            splits_by_tid[tid] = sp_list

    n_splits = sum(len(v) for v in splits_by_tid.values())
    logging.info(f"Scanned {len(by_id)} tracks, found {n_splits} impossible jumps "
                 f"in {len(splits_by_tid)} tracks")
    for tid in sorted(splits_by_tid.keys()):
        for sp in splits_by_tid[tid]:
            logging.info(f"  P{tid}: jump @ frame {sp['split_frame']} "
                         f"(prev {sp['prev_frame']}, gap {sp['frame_gap']}f, "
                         f"{sp['dist_px']:.0f}px = {sp['speed_px_f']:.1f}px/f "
                         f"~{sp['speed_px_f'] * fps / 100:.1f}m/s assumed)")

    last_id = apply_splits(per_frame_dets, splits_by_tid, next_id_start)
    new_ids_added = last_id - next_id_start

    data['frames'] = per_frame_dets
    data['split_jumps'] = {
        'source_tracks_json': str(tracks_json),
        'max_speed_px_per_frame': max_speed,
        'min_absolute_jump_px': min_jump,
        'max_gap_frames': max_gap,
        'n_splits': n_splits,
        'n_new_ids': new_ids_added,
        'splits_by_tid': {str(tid): sp_list for tid, sp_list in splits_by_tid.items()},
    }

    if output_json is None:
        output_json = str(tracks_json.with_name(
            tracks_json.stem + '_split.json'))
    with open(output_json, 'w') as f:
        json.dump(data, f)
    logging.info(f"Split tracks JSON saved -> {output_json}")
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
    parser.add_argument('--max_speed', type=float,
                        default=DEFAULT_MAX_SPEED_PX_PER_FRAME,
                        help="Maximum bbox center speed in px/frame above which "
                             "the displacement is considered impossible.")
    parser.add_argument('--min_jump', type=float,
                        default=DEFAULT_MIN_ABSOLUTE_JUMP_PX,
                        help="Don't split if the absolute displacement is below "
                             "this (avoids splitting on detection noise).")
    parser.add_argument('--max_gap', type=int,
                        default=DEFAULT_MAX_GAP_FRAMES,
                        help="Ignore intra-track gaps larger than this (the "
                             "merger handles long absences).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    split_func(
        tracks_json=args.tracks_json,
        output_json=args.output_json,
        max_speed=args.max_speed,
        min_jump=args.min_jump,
        max_gap=args.max_gap,
    )


if __name__ == '__main__':
    main()
