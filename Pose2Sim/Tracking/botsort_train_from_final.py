#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################################
    ## Rebuild the active-learning classifier from an existing      ##
    ## final.json without re-running the manual review UI.          ##
    ##################################################################

    Use case: you reviewed camera 1 of a trial before the classifier
    existed (or before thumbnails were tracked). This utility reads the
    final.json (which already encodes your manual decisions via the
    id_to_final mapping) and trains a classifier from those crops, so
    the .pkl is available to bootstrap the review of the next camera.

    What it does:
      - Loads final.json -> resolves merged.json -> resolves tracks.json
      - For every final group:
          - Adds the founding window crops of the earliest track in that
            group (= the "founder" of this patient on this camera).
          - For every later track that the user merged into this group,
            adds the FIRST N frames of that track (= "fresh" crops the
            user effectively validated at confirmation time).
      - Skips the rest of the tracks (potentially polluted by mid-track
        ID swaps; same hygiene as the live review).
      - Trains the MLP and saves a .pkl + thumbnails-per-label.

    Usage:
      python botsort_train_from_final.py -f <final.json>
      python botsort_train_from_final.py -f <final.json> -o <out.pkl>
      from Pose2Sim.Tracking import botsort_train_from_final
      botsort_train_from_final.train_from_final(final_json=r'...')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

from Pose2Sim.Tracking.botsort_classifier import (
    FeatureExtractor, FounderClassifier,
)
from Pose2Sim.Tracking.botsort_merger import (
    STAFF_ID_OFFSET, extract_torso_hist, histogram_similarity,
)
import numpy as np
import cv2


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS (mirror merger_review's defaults so the training data is
## constructed the exact same way as during the live UI run)
FOUNDING_WINDOW_FRAMES = 60
FOUNDER_TRAIN_CROPS = 25
NEWCOMER_FRESH_CROPS = 15


## FUNCTIONS
def train_from_final(final_json, output=None,
                     reid_model='osnet_ain_x1_0_msmt17.pt',
                     device='cuda:0',
                     founding_window=FOUNDING_WINDOW_FRAMES,
                     founder_crops=FOUNDER_TRAIN_CROPS,
                     newcomer_crops=NEWCOMER_FRESH_CROPS):
    final_json = Path(final_json)
    with open(final_json, 'r') as f:
        final_data = json.load(f)

    merged_json = Path(final_data['source_merged_json'])
    with open(merged_json, 'r') as f:
        merged = json.load(f)

    tracks_json = Path(merged['source_tracks_json'])
    with open(tracks_json, 'r') as f:
        tracks_data = json.load(f)

    video_path = Path(tracks_data['video'])
    per_frame_dets = tracks_data['frames']
    id_to_final = {int(k): int(v) for k, v in final_data['id_to_final'].items()}

    # Detections per original track id
    by_id = defaultdict(list)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    for tid in by_id:
        by_id[tid].sort(key=lambda x: x[0])

    # Group by final id; sort tracks within each group by first_frame
    by_final = defaultdict(list)  # final_id -> [(tid, items), ...]
    for tid, items in by_id.items():
        final_id = id_to_final.get(tid, tid)
        by_final[final_id].append((tid, items))
    for final_id in by_final:
        by_final[final_id].sort(key=lambda gi: gi[1][0][0])

    logging.info(f"Loaded {final_json.name}: "
                 f"{len(id_to_final)} original tracks → "
                 f"{len(by_final)} final groups")

    # Build classifier
    extractor = FeatureExtractor(reid_weights=reid_model, device=device,
                                 half=False)
    classifier = FounderClassifier(extractor)

    for final_id in sorted(by_final.keys()):
        group_items = by_final[final_id]
        earliest_tid, earliest_items = group_items[0]
        first_frame_in_group = earliest_items[0][0]

        if first_frame_in_group < founding_window:
            # This group's earliest track is a true founder -> use founding window
            pairs = [(f, b) for f, b in earliest_items if f < founding_window]
            added = classifier.add_examples_from_frames(
                video_path, pairs, label=final_id, max_crops=founder_crops)
            logging.info(f"  P{final_id} (founder T{earliest_tid}): "
                         f"+{added} founding crops")
        else:
            # Group's first appearance is after the founding window: it was
            # introduced as a manual "new person" decision. Use its first N
            # frames as the trusted seed (same logic as live review).
            pairs = earliest_items[:newcomer_crops]
            added = classifier.add_examples_from_frames(
                video_path, pairs, label=final_id, max_crops=newcomer_crops)
            logging.info(f"  P{final_id} (new-person, T{earliest_tid}): "
                         f"+{added} fresh crops")

        # Every later track that the user merged into this group: take its
        # first N frames (= the bit the user validated by clicking).
        for tid, items in group_items[1:]:
            pairs = items[:newcomer_crops]
            added = classifier.add_examples_from_frames(
                video_path, pairs, label=final_id, max_crops=newcomer_crops)
            logging.info(f"     +T{tid} merged into P{final_id}: +{added} crops")

    counts = classifier.n_examples_per_label()
    logging.info(f"Total training set: {len(classifier.X)} examples "
                 f"on {len(counts)} labels")
    for lab in sorted(counts.keys()):
        logging.info(f"  P{lab}: {counts[lab]} crops")

    if not classifier.train():
        logging.warning("Classifier could not be trained (need >= 2 labels "
                        "with enough examples).")
        return None

    if output is None:
        output = str(final_json.with_name(
            final_json.stem + '_classifier.pkl'))
    classifier.save(output)
    logging.info(f"Classifier saved -> {output}")
    return output


MAX_CROPS_VALIDATED = 150       # cap per identity for full-sequence training
MIN_DETS_TO_TRAIN = 20          # identities with fewer detections are noise /
                                # tiny fragments -> not used for training
OUTLIER_SIM_THRESH = 0.40       # crops whose torso histogram is this far from
                                # the identity's median are dropped (= bbox on
                                # several people, garbage detection, etc.)


def _color_consistent_crops(video_path, items, sim_thresh=OUTLIER_SIM_THRESH):
    '''
    Given an identity's sampled (frame, bbox) crops, drop the ones whose
    torso colour histogram deviates from the identity's median — these are
    typically bboxes that merged several overlapping people, or otherwise
    garbage detections that would pollute training. Single sequential pass.
    '''
    if len(items) <= 5:
        return items  # too few to estimate a reliable median
    needed = {}
    for f, b in items:
        needed.setdefault(f, []).append(b)
    last = max(needed.keys())
    cap = cv2.VideoCapture(str(video_path))
    hists = {}
    f_idx = 0
    while f_idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx in needed:
            for b in needed[f_idx]:
                h = extract_torso_hist(frame, b)
                if h is not None:
                    hists[(f_idx, tuple(b))] = h
        f_idx += 1
    cap.release()
    valid = [(f, b, hists[(f, tuple(b))]) for f, b in items
             if (f, tuple(b)) in hists]
    if len(valid) < 5:
        return items
    median = np.median(np.stack([h for _, _, h in valid], axis=0), axis=0)
    s = float(np.sum(median))
    if s > 1e-9:
        median = median / s
    kept = [(f, b) for f, b, h in valid
            if histogram_similarity(h, median) >= sim_thresh]
    return kept if kept else items


def train_from_validated(validated_json, classifier_pkl=None,
                         max_crops_per_label=MAX_CROPS_VALIDATED,
                         backbone='osnet', dinov2_size='small',
                         reid_model='osnet_ain_x1_0_msmt17.pt',
                         device='cuda:0', head_path=None):
    '''
    Train (or extend) the classifier from a *validated* json (output of
    validate_review). Because every track is now verified clean, we use
    the FULL sequence of each identity (capped, evenly sampled) — not just
    founder-window + first frames. If `classifier_pkl` already exists, we
    LOAD it and ACCUMULATE this camera's crops into it (so the model gets
    richer with each validated camera), then save back.
    '''
    validated_json = Path(validated_json)
    with open(validated_json, 'r') as f:
        data = json.load(f)
    if not data.get('validated', False):
        logging.warning("JSON not marked validated=true; training anyway, "
                        "but quality is not guaranteed.")
    video_path = Path(data['video'])
    per_frame_dets = data['frames']
    excluded = set(data.get('excluded_from_training', []))
    if excluded:
        logging.info(f"Identities marked indéterminé (excluded): "
                     f"{sorted(excluded)}")

    by_id = defaultdict(list)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    for tid in by_id:
        by_id[tid].sort(key=lambda x: x[0])

    # Load existing classifier (accumulate) or start fresh
    if classifier_pkl and Path(classifier_pkl).is_file():
        classifier = FounderClassifier.load(classifier_pkl)
        logging.info(f"Loaded existing classifier ({len(classifier.X)} examples) "
                     f"to accumulate into.")
    else:
        extractor = FeatureExtractor(backbone=backbone, reid_weights=reid_model,
                                     dinov2_size=dinov2_size, device=device,
                                     half=False, head_path=head_path)
        classifier = FounderClassifier(extractor)

    added_total = 0
    for fid in sorted(by_id.keys()):
        # Skip orphan / unassigned temporaries (kept clear of P/S ranges)
        if fid >= STAFF_ID_OFFSET * 2:
            continue
        # Skip identities the user flagged as indeterminate
        if fid in excluded:
            logging.info(f"  {fid}: skipped (indéterminé)")
            continue
        # Skip tiny fragments / noise (too few detections to be a real person)
        if len(by_id[fid]) < MIN_DETS_TO_TRAIN:
            continue
        # Evenly sample, then drop colour-outlier crops (bbox on several
        # people / garbage) before adding to training.
        items = by_id[fid]
        if len(items) > max_crops_per_label:
            idx = np.linspace(0, len(items) - 1, max_crops_per_label).astype(int)
            sampled = [items[i] for i in idx]
        else:
            sampled = items
        n_before = len(sampled)
        sampled = _color_consistent_crops(video_path, sampled)
        dropped = n_before - len(sampled)
        added = classifier.add_examples_from_frames(
            video_path, sampled, label=fid, max_crops=max_crops_per_label)
        added_total += added
        logging.info(f"  {fid}: +{added} crops"
                     + (f" ({dropped} outliers couleur écartés)" if dropped else ""))

    logging.info(f"Added {added_total} crops total. Training...")
    if not classifier.train():
        logging.warning("Could not train (need >= 2 labels).")
        return None

    if classifier_pkl is None:
        classifier_pkl = str(validated_json.with_name('trial_classifier.pkl'))
    classifier.save(classifier_pkl)
    counts = classifier.n_examples_per_label()
    logging.info(f"Classifier saved -> {classifier_pkl}")
    logging.info(f"  {len(classifier.X)} examples on {len(counts)} labels: "
                 f"{ {k: counts[k] for k in sorted(counts)} }")
    return classifier_pkl


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--final_json', required=True,
                        help="Path to the _final.json produced by a previous run "
                             "of merger_review / botsort_pipeline.")
    parser.add_argument('-o', '--output', default=None,
                        help="Where to save the classifier .pkl "
                             "(default: alongside the final.json).")
    parser.add_argument('--reid_model', default='osnet_ain_x1_0_msmt17.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--founding_window', type=int,
                        default=FOUNDING_WINDOW_FRAMES)
    parser.add_argument('--founder_crops', type=int,
                        default=FOUNDER_TRAIN_CROPS)
    parser.add_argument('--newcomer_crops', type=int,
                        default=NEWCOMER_FRESH_CROPS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    train_from_final(
        final_json=args.final_json,
        output=args.output,
        reid_model=args.reid_model,
        device=args.device,
        founding_window=args.founding_window,
        founder_crops=args.founder_crops,
        newcomer_crops=args.newcomer_crops,
    )


if __name__ == '__main__':
    main()
