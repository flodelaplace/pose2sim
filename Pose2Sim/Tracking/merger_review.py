#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###############################################################
    ## Manual ranked re-ID review for the BoT-SORT merger output ##
    ###############################################################

    For each track that "ended" before the end of the video (= a person
    that disappeared from the camera), this tool shows:

      - The crop of that person's LAST visible frame on the left
      - All the candidate tracks that appeared LATER in the video, sorted
        by combined score (appearance + spatial proximity + edge match),
        each as a single crop of their first visible frame on the right
      - For each candidate, the frame number, gap to disappearance, and
        score breakdown

    The user picks the correct candidate with a number key (1, 2, 3 ...)
    or "0"/"N" to say no candidate matches (= the person did not come back
    in this camera). The chosen merge is applied to the running mapping
    and the next disappeared track is shown.

    Why not pairwise Y/N: pairwise prompts often forced you to compare two
    persons that obviously aren't the same (waste of a click); and showing
    several frames of one track at once turned out to be confusing when
    the underlying BoT-SORT track had a swap inside it.

    Usage:
      merger_review -m <merged.json>
      merger_review -m <merged.json> -n 8           (stop early at N persons)
      merger_review -m <merged.json> --max_candidates 6
      from Pose2Sim.Tracking import merger_review; merger_review.review_func(merged_json=r'<path>')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from Pose2Sim.Tracking.botsort_merger import (
    extract_torso_hist, histogram_similarity,
    edge_label, bbox_center, score_pair,
    OPPOSITE_EDGE, STAFF_ID_OFFSET, format_id,
)
from Pose2Sim.Tracking.botsort_classifier import (
    FeatureExtractor, FounderClassifier,
)


## CLASSIFIER CONSTANTS
FOUNDER_TRAIN_CROPS = 25       # crops per founder added at startup
NEWCOMER_FRESH_CROPS = 15      # first N crops of confirmed newcomer added
CLASSIFIER_CONF_THRESHOLD = 0.55  # below this, fall back to histogram ranking


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
DEFAULT_MAX_CANDIDATES = 6     # show at most this many candidates per popup
MIN_SHOW_SCORE = 0.0           # show every valid candidate ranked by score
CROP_PAD_FRAC = 0.10
FOUNDING_WINDOW_FRAMES = 60    # tracks starting within first N frames are
                               # "founders" (= the N persons of interest, P1, P2, ...)
                               # Every later track must be reassigned to a
                               # founder or explicitly marked as a new person.


def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 9973 + 17)
    return tuple(int(c) for c in rng.integers(60, 230, size=3))


def crop_from_video(cap, frame_idx, bbox, pad=CROP_PAD_FRAC):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    x2 = int(min(W, x2 + pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    y2 = int(min(H, y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def representative_score(bbox, width, height):
    '''
    How "showable" is a detection: prefer bboxes well inside the frame
    (away from edges) and with a reasonable size. Used to pick the single
    crop we show as "this is what this person looks like".
    '''
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Normalized distance to the nearest edge (0 = on edge, 0.5 = center)
    min_edge_dist = min(cx / width, (width - cx) / width,
                        cy / height, (height - cy) / height)
    area_frac = (bw * bh) / (width * height)
    # Reward area up to ~2% of the frame, then plateau
    area_term = min(area_frac * 50.0, 1.0)
    # Penalize aspect ratios that are too squashed (= sitting on the floor / occluded)
    aspect = bh / bw if bw > 0 else 0
    aspect_term = 1.0 if 1.5 <= aspect <= 4.0 else 0.6
    return min_edge_dist * area_term * aspect_term


def pick_representative_frame(items, width, height):
    '''Return (frame_idx, bbox) of the best-looking detection of a track.'''
    best = items[len(items) // 2]
    best_score = -1.0
    for f_idx, bbox, *_ in items:
        s = representative_score(bbox, width, height)
        if s > best_score:
            best_score = s
            best = (f_idx, bbox)
    return best


def build_group_summaries(per_frame_dets, id_to_final, video_path,
                          width, height, n_samples=6):
    '''
    Build per-final-group summaries from the auto-merged result. Each
    group inherits the union of detections of its member original tracks.

    Returns: dict {final_id: {first_frame, last_frame, first_bbox,
                              last_bbox, repr_frame, repr_bbox,
                              exit_edge, entry_edge,
                              embedding (pooled torso histogram)}}.
    '''
    # Pool detections by final id
    by_final = {}  # final_id -> list of (frame_idx, bbox, original_id)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            fid = id_to_final.get(r['id'], r['id'])
            by_final.setdefault(fid, []).append((f_idx, r['bbox'], r['id']))

    # Decide which (frame, fid, bbox) triples need a histogram and collect
    # them once, so we can read the video in a single sequential pass.
    needed_per_frame = {}  # frame_idx -> [(fid, bbox), ...]
    for fid, items in by_final.items():
        items.sort(key=lambda x: x[0])
        if len(items) <= n_samples:
            sample = items
        else:
            sample = [items[i] for i in
                      np.linspace(0, len(items) - 1, n_samples).astype(int)]
        for f_idx, bbox, _ in sample:
            needed_per_frame.setdefault(f_idx, []).append((fid, bbox))

    hist_per_fid = {fid: [] for fid in by_final}
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
                for fid, bbox in needed_per_frame[f_idx]:
                    h = extract_torso_hist(frame, bbox)
                    if h is not None:
                        hist_per_fid[fid].append(h)
            f_idx += 1
        cap.release()

    summaries = {}
    for fid, items in by_final.items():
        first_frame, first_bbox, _ = items[0]
        last_frame, last_bbox, _ = items[-1]
        first_center = bbox_center(first_bbox)
        last_center = bbox_center(last_bbox)
        repr_frame, repr_bbox = pick_representative_frame(items, width, height)
        hists = hist_per_fid.get(fid, [])
        embed = None
        if hists:
            embed = np.mean(np.stack(hists, axis=0), axis=0)
            s = float(np.sum(embed))
            if s > 1e-9:
                embed /= s
        summaries[fid] = {
            'tid': fid,
            'first_frame': first_frame, 'last_frame': last_frame,
            'first_center': first_center, 'last_center': last_center,
            'first_bbox': first_bbox, 'last_bbox': last_bbox,
            'repr_frame': repr_frame, 'repr_bbox': repr_bbox,
            'exit_edge': edge_label(*last_center, width, height),
            'entry_edge': edge_label(*first_center, width, height),
            'embedding': embed,
            'lifespan': last_frame - first_frame + 1,
            'n_detections': len(items),
            'original_ids': sorted({orig for _, _, orig in items}),
            # Full (frame, bbox) sequence, kept so the classifier can be
            # polled on N samples across the track and the scores
            # AVERAGED — robust to occasional bad crops (motion blur,
            # half-out-of-frame, occlusion) that would otherwise dominate
            # a single repr_frame prediction.
            'items': [(int(f), b) for f, b, _ in items],
        }
    return summaries


def predict_track_aggregated(classifier, items, video_path, cap=None,
                             n_samples=8):
    '''
    Aggregate the classifier's prediction over n_samples evenly-spaced
    detections of a track and return the score-averaged ranking.

    Why : single-frame prediction is brittle — a representative crop
    chosen by `pick_representative_frame` can still be a poor sample
    (motion blur, partial occlusion, edge of frame). Sampling 8 frames
    and averaging the cosine sims per label gives a much more robust
    confidence, so auto_founder / auto_newcomers don't reject 500-frame
    tracks just because their first or middle crop was unlucky.

    Items: iterable of (frame_idx, bbox) (extra fields ignored).
    Returns ranked [(label, mean_score), ...] desc, or None.
    '''
    if classifier is None or not getattr(classifier, 'trained', False):
        return None
    items = [(int(it[0]), it[1]) for it in items]
    items.sort(key=lambda x: x[0])
    if not items:
        return None
    if len(items) > n_samples:
        idx = np.linspace(0, len(items) - 1, n_samples).astype(int)
        sampled = [items[int(i)] for i in idx]
    else:
        sampled = items
    own_cap = False
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        own_cap = True
    accum = {}
    try:
        for f_idx, bbox in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            ranked = classifier.predict(frame, bbox)
            if ranked is None:
                continue
            for label, score in ranked:
                accum.setdefault(int(label), []).append(float(score))
    finally:
        if own_cap:
            cap.release()
    if not accum:
        return None
    means = [(lab, float(np.mean(sc))) for lab, sc in accum.items()]
    means.sort(key=lambda x: -x[1])
    return means


def disable_mpl_default_keys():
    matplotlib.rcParams['keymap.save'] = []
    matplotlib.rcParams['keymap.fullscreen'] = []
    matplotlib.rcParams['keymap.yscale'] = []
    matplotlib.rcParams['keymap.quit'] = []
    matplotlib.rcParams['keymap.home'] = []
    matplotlib.rcParams['keymap.back'] = []
    matplotlib.rcParams['keymap.forward'] = []
    matplotlib.rcParams['keymap.pan'] = []
    matplotlib.rcParams['keymap.zoom'] = []
    matplotlib.rcParams['keymap.grid'] = []


def render_founder_mapping_figure(founder, ranked, classifier,
                                  cap, n_frames_total, progress_str):
    '''
    Founder identification popup.
    Left  = the current cam's local founder, its representative crop.
    Right = top-K global labels predicted by the loaded classifier (each
            with the thumbnail saved during a previous review). When the
            classifier has no labels yet (= first camera), the right
            side is empty and the user marks each founder as a NEW
            patient (P key) or as staff (S/0 key).
    Keys:
      1..K : pick the K-th classifier candidate as the matching patient
      P    : declare this is a brand-new patient (assigns a fresh global ID)
      S, 0 : mark this track as STAFF / ignore (excluded from training)
      Q    : abort the founder phase, keep remaining as local
    '''
    n = len(ranked)
    # Always show at least the founder column; pad right with one note
    # column if there are no classifier candidates.
    ncols = max(2, 1 + n)
    fig_w = min(2.2 * ncols, 13.0)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2))
    if ncols == 1:
        axes = [axes]

    repr_frame = founder.get('repr_frame', founder['first_frame'])
    repr_bbox = founder.get('repr_bbox', founder['first_bbox'])
    crop = crop_from_video(cap, repr_frame, repr_bbox)
    ax = axes[0]
    if crop is not None:
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    color = tuple(c / 255.0 for c in color_for_id(founder['tid'])[::-1])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(4)
    ax.set_title(f"FOUNDER local\nT{founder['tid']}\nvue claire frame {repr_frame}",
                 fontsize=10, weight='bold')

    for i, (label, prob) in enumerate(ranked):
        ax = axes[1 + i]
        thumb = classifier.thumbnails.get(int(label))
        if thumb is not None:
            ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        else:
            ax.set_facecolor('lightgray')
            ax.text(0.5, 0.5, f"{format_id(label)}\n(pas de vignette)",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        c = tuple(v / 255.0 for v in color_for_id(label)[::-1])
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(3)
        ax.set_title(f"[{i+1}]  {format_id(label)}\nconfiance {prob:.2f}",
                     fontsize=10)
    # If no classifier candidates, fill the placeholder panel
    if n == 0:
        ax = axes[1]
        ax.set_facecolor('whitesmoke')
        ax.text(0.5, 0.5,
                "Aucune personne connue pour l'instant.\n\n"
                "P = nouveau patient (P1, P2, ...)\n"
                "T = nouveau staff (S1, S2, ...)\n"
                "S ou 0 = ignorer\n"
                "Q = quitter la phase",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"{progress_str}    "
        f"1..{n}=match  P=nouveau patient  T=nouveau staff  S/0=ignore  Q=quit",
        fontsize=11, weight='bold',
    )
    plt.subplots_adjust(top=0.74, bottom=0.03, left=0.02, right=0.99,
                        hspace=0.05, wspace=0.05)
    return fig


def remap_ids_inplace(per_frame_dets, summaries, mapping):
    '''Rewrite every detection id via mapping; rebuild summaries dict
    keys accordingly. Detections whose id is not in mapping are left
    untouched.'''
    for recs in per_frame_dets:
        for r in recs:
            if r['id'] in mapping:
                r['id'] = mapping[r['id']]
    new_summaries = {}
    for tid, summ in summaries.items():
        new_tid = mapping.get(tid, tid)
        summ['tid'] = new_tid
        if 'original_ids' in summ:
            summ['original_ids'] = [mapping.get(t, t) for t in summ['original_ids']]
        new_summaries[new_tid] = summ
    return new_summaries


def render_picker_figure(newcomer, candidates, cap, fps, n_frames_total,
                         progress_str):
    '''
    Layout: one wide row.
    Left column  = the newly-appearing track's FIRST visible crop.
    Right columns = each candidate's LAST visible crop (= the most recent
    look of a person who was already here and could be the newcomer
    coming back), sorted by score descending.
    '''
    n_cands = len(candidates)
    ncols = 1 + n_cands
    fig_w = min(2.2 * ncols, 13.0)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2))
    if ncols == 1:
        axes = [axes]

    # Left: newcomer (best in-frame view, with entry info in the title)
    nc_repr_frame = newcomer.get('repr_frame', newcomer['first_frame'])
    nc_repr_bbox = newcomer.get('repr_bbox', newcomer['first_bbox'])
    new_crop = crop_from_video(cap, nc_repr_frame, nc_repr_bbox)
    ax = axes[0]
    if new_crop is not None:
        ax.imshow(cv2.cvtColor(new_crop, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    border_color = tuple(c / 255.0 for c in color_for_id(newcomer['tid'])[::-1])
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color); spine.set_linewidth(4)
    ax.set_title(
        f"NOUVEAU  {format_id(newcomer['tid'])}\n"
        f"vue claire frame {nc_repr_frame}\n"
        f"entré frame {newcomer['first_frame']} ({newcomer['entry_edge']})",
        fontsize=9, weight='bold',
    )

    # Right: candidates (earlier tracks/groups, shown with their BEST
    # representative crop = clean centered view, not the partial exit view)
    for i, (score, B, detail) in enumerate(candidates):
        ax = axes[1 + i]
        is_virtual = B.get('is_virtual', False)
        if is_virtual:
            # Virtual candidate = known identity from a previous cam, no
            # local track yet. Use the classifier-saved thumbnail.
            thumb = B.get('thumb')
            if thumb is not None:
                ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            else:
                ax.set_facecolor('lightyellow')
                ax.text(0.5, 0.5,
                        f"{format_id(B['tid'])}\n(vignette indisponible)",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10)
            status = "JAMAIS VU sur cette cam (connu du classifier)"
            repr_info = "—"
        else:
            repr_frame = B.get('repr_frame', B['last_frame'])
            repr_bbox = B.get('repr_bbox', B['last_bbox'])
            crop = crop_from_video(cap, repr_frame, repr_bbox)
            if crop is not None:
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if B['last_frame'] < newcomer['first_frame']:
                gap_frames = newcomer['first_frame'] - B['last_frame']
                gap_sec = gap_frames / fps
                status = f"OK disparu depuis {gap_frames}f / {gap_sec:.1f}s"
            else:
                overlap = B['last_frame'] - newcomer['first_frame'] + 1
                status = f"!! encore tracké jusqu'à {B['last_frame']} (overlap {overlap}f)"
            repr_info = f"vue claire frame {repr_frame}"
        ax.set_xticks([]); ax.set_yticks([])
        cand_color = tuple(c / 255.0 for c in color_for_id(B['tid'])[::-1])
        for spine in ax.spines.values():
            spine.set_edgecolor(cand_color); spine.set_linewidth(3)
        ax.set_title(
            f"[{i+1}]  {format_id(B['tid'])}   score={score:.2f}\n"
            f"{repr_info}\n"
            f"{status}\n"
            f"sortie: {B['exit_edge']}  "
            f"app {detail['appearance']:.2f} sp {detail['spatial']:.2f}",
            fontsize=8,
        )

    fig.suptitle(
        f"{progress_str}    "
        f"1..{n_cands}=qui c'est   P=nouv.patient   T=nouv.staff   "
        f"0/N=indécis   S=skip   Q=quit",
        fontsize=11, weight='bold',
    )
    plt.subplots_adjust(top=0.74, bottom=0.03, left=0.02, right=0.99,
                        hspace=0.05, wspace=0.05)
    return fig


def review_func(merged_json, output_json=None, output_mp4=None,
                final_n_target=None, max_candidates=DEFAULT_MAX_CANDIDATES,
                founding_window=FOUNDING_WINDOW_FRAMES,
                founding_start=0,
                classifier_state=None, classifier_save_path=None,
                use_classifier=True, no_retrain=False,
                auto_newcomers_thresh=None, auto_margin=0.15,
                auto_founder_thresh=None,
                backbone='osnet', dinov2_size='small',
                head_path=None,
                reid_model='osnet_ain_x1_0_msmt17.pt',
                device='cuda:0'):
    '''
    Closed-world review with optional active-learning classifier.

    For each newcomer track, present the user with the founders / earlier
    groups that started BEFORE the newcomer, ranked by combined score
    (classifier confidence if trained, else histogram + spatial + edge).
    User picks the matching one or "new person". After each confirmation,
    the classifier is retrained with the fresh crops of the new match,
    becoming more accurate as the review progresses.

    `classifier_state`: optional path to a previously-saved classifier
    .pkl (e.g. from another camera of the same trial) to bootstrap
    re-identification on the new camera.
    '''
    merged_json = Path(merged_json)
    with open(merged_json, 'r') as f:
        merged = json.load(f)
    tracks_json_path = Path(merged['source_tracks_json'])
    with open(tracks_json_path, 'r') as f:
        tracks_data = json.load(f)

    video_path = Path(tracks_data['video'])
    per_frame_dets = tracks_data['frames']
    width, height = tracks_data['width'], tracks_data['height']
    fps = float(tracks_data['fps'])
    n_frames = tracks_data['n_frames']

    # Mapping starts from the auto-merger's output
    id_to_final = {int(k): int(v)
                   for k, v in merged.get('id_to_final', {}).items()}

    logging.info(f"Loaded merged.json: {len(set(id_to_final.values()))} "
                 f"groups after auto-merging.")

    # Build group summaries on the merged data
    logging.info("Building group summaries for review...")
    summaries = build_group_summaries(per_frame_dets, id_to_final,
                                      video_path, width, height)
    logging.info(f"  {len(summaries)} groups to consider")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    disable_mpl_default_keys()

    # Founders = tracks that are visible inside the founding window
    # [founding_start, founding_start + founding_window]. Use this window
    # to point at the moment where everyone is best on screen (not always
    # the very start of the video). Tracks NOT in that window are
    # "newcomers" that must each be reassigned to a confirmed patient or
    # marked as a new person / staff.
    f_lo = max(0, int(founding_start))
    f_hi = f_lo + int(founding_window)
    founders = [s for s in summaries.values()
                if s['first_frame'] < f_hi and s['last_frame'] >= f_lo]
    founder_ids = sorted(s['tid'] for s in founders)
    logging.info(f"Founding window: frames [{f_lo}, {f_hi}]")
    logging.info(f"Founders ({len(founder_ids)}): {founder_ids}")

    # The set of OFFICIAL identities = the people the user explicitly
    # established (founders P1.., S1.. + any new patient/staff created in
    # the newcomer review). Only these are valid merge targets; everything
    # else is a fragment to be assigned to one of them. Written to the
    # final.json so the validation tool proposes only these.
    official_ids = set()

    # ---- Init classifier (active learning) ----
    classifier = None
    classifier_was_loaded = False
    if use_classifier:
        try:
            extractor = FeatureExtractor(backbone=backbone,
                                         reid_weights=reid_model,
                                         dinov2_size=dinov2_size,
                                         device=device, half=False,
                                         head_path=head_path)
            if classifier_state:
                p = Path(classifier_state)
                if p.is_file():
                    classifier = FounderClassifier.load(classifier_state, extractor)
                    classifier_was_loaded = True
                    logging.info(f"Loaded classifier state from {classifier_state}: "
                                 f"{len(classifier.X)} examples on "
                                 f"{len(set(classifier.y))} labels "
                                 f"(trained={classifier.trained})")
                else:
                    logging.warning(f"--classifier_state given but file NOT FOUND: "
                                    f"{classifier_state}  -> starting from scratch")
                    classifier = FounderClassifier(extractor)
            else:
                logging.info("No --classifier_state given; starting fresh classifier.")
                classifier = FounderClassifier(extractor)

            # ---- Mandatory founder identification phase ----
            # Always run, even on the very first camera. The user labels
            # each candidate founder as:
            #   * an EXISTING known global patient (top-K classifier picks)
            #   * a brand-NEW patient (P key)        -> next_new_global
            #   * STAFF / IGNORE (S or 0 key)        -> offset, not a patient
            # On camera 1 the classifier is empty, so only P/S options
            # are useful; on later cameras the classifier proposes top-K
            # known patients ranked by confidence.
            classifier_labels = (sorted(set(int(l) for l in classifier.y))
                                 if classifier.trained else [])
            # Patients = labels < STAFF_ID_OFFSET, staff = labels >= STAFF_ID_OFFSET
            patient_labels = [l for l in classifier_labels if l < STAFF_ID_OFFSET]
            staff_labels = [l for l in classifier_labels if l >= STAFF_ID_OFFSET]
            next_new_patient = (max(patient_labels) + 1) if patient_labels else 1
            next_new_staff = (max(staff_labels) + 1) if staff_labels else STAFF_ID_OFFSET
            local_max = max(s['tid'] for s in summaries.values())
            safe_offset = (max(classifier_labels) if classifier_labels else 0)
            safe_offset = max(safe_offset, local_max, STAFF_ID_OFFSET) + 100000

            logging.info("=" * 60)
            if classifier_was_loaded and classifier.trained:
                logging.info(f"Founder identification (cross-cam): "
                             f"{len(founders)} local candidates")
                if patient_labels:
                    logging.info(f"  Known patients: "
                                 f"{[format_id(l) for l in patient_labels]}")
                if staff_labels:
                    logging.info(f"  Known staff: "
                                 f"{[format_id(l) for l in staff_labels]}")
            else:
                logging.info(f"Founder identification: {len(founders)} "
                             f"candidates, no prior identities known yet.")
            logging.info("  1..K = match this person   P = new patient   "
                         "T = new staff   S or 0 = ignore   Q = quit")
            logging.info("=" * 60)

            disable_mpl_default_keys()

            founder_picks = {}   # local_tid -> int (global label) or 'staff'
            for fi, f in enumerate(sorted(founders, key=lambda s: s['first_frame']),
                                   start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        f.get('repr_frame', f['first_frame']))
                ok, frame = cap.read()
                if not ok:
                    continue
                # Prediction AGGREGATED over 8 evenly-spaced frames of the
                # founder track — robust to a poor repr_frame (motion
                # blur, edge of frame, partial occlusion) that would
                # otherwise tank the top-1 score and block auto mode.
                ranked = (predict_track_aggregated(
                            classifier, f.get('items', []),
                            video_path, cap=cap, n_samples=8)
                          if classifier.trained else None)
                if ranked is None and classifier.trained:
                    # Fallback to single-frame on the very first cam (no
                    # items list yet would be unusual but be safe).
                    ranked = classifier.predict(
                        frame, f.get('repr_bbox', f['first_bbox']))
                # In cross-cam, surface ALL known classifier identities,
                # not just the top-K. Otherwise a quiet S1 (= coach who
                # rarely matches well at first glance) gets cut off and
                # the user can't pick it.
                effective_max = max(max_candidates, len(classifier_labels))
                top_K = ranked[:effective_max] if ranked else []

                # ---- Auto-founder mode ----
                # If --auto_founder is set AND the classifier's top-1
                # cosine sim is above the threshold AND clearly beats
                # top-2 (margin), accept the prediction without opening
                # the manual popup. Falls back to manual otherwise so the
                # ambiguous cases stay user-controlled.
                if (auto_founder_thresh is not None and top_K
                        and top_K[0][1] >= auto_founder_thresh
                        and (len(top_K) == 1
                             or top_K[0][1] - top_K[1][1] >= auto_margin)):
                    auto_label = int(top_K[0][0])
                    if auto_label not in (founder_picks.values()):
                        founder_picks[f['tid']] = auto_label
                        logging.info(
                            f"  Founder {fi}/{len(founders)} T{f['tid']} "
                            f"-> AUTO {format_id(auto_label)} "
                            f"(cosine {top_K[0][1]:.2f})")
                        continue
                progress = (f"Founder {fi}/{len(founders)}  local T{f['tid']}  "
                            f"frame {f['first_frame']}")
                fig = render_founder_mapping_figure(
                    f, top_K, classifier, cap, n_frames, progress)
                decision = {'key': None}

                def on_key(event, _decision=decision, _fig=fig):
                    k = (event.key or '').lower()
                    if (k in ('q', 's', 'p', 't')
                            or (k.isdigit() and k in '0123456789')):
                        _decision['key'] = k
                        plt.close(_fig)

                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()
                key = decision['key'] or 's'

                if key == 'q':
                    logging.info("Founder phase interrupted (Q). "
                                 "Remaining candidates left as local IDs.")
                    break
                if key in ('s', '0'):
                    founder_picks[f['tid']] = 'ignore'
                    continue
                if key == 'p':
                    founder_picks[f['tid']] = next_new_patient
                    next_new_patient += 1
                    continue
                if key == 't':
                    founder_picks[f['tid']] = next_new_staff
                    next_new_staff += 1
                    continue
                if key.isdigit():
                    pick_idx = int(key) - 1
                    if 0 <= pick_idx < len(top_K):
                        founder_picks[f['tid']] = int(top_K[pick_idx][0])

            # Build full remapping for every local TID
            full_remap = {}
            used_globals = set()
            for tid, summ in summaries.items():
                decision = founder_picks.get(tid)
                if decision is None or decision == 'ignore':
                    # Not assigned (skipped, ignored, or non-founder local track)
                    # -> offset to safe zone; will NOT be added to training.
                    full_remap[tid] = tid + safe_offset
                else:
                    if decision in used_globals:
                        logging.warning(
                            f"Local T{tid} mapped to already-used "
                            f"{format_id(decision)}; both will fuse into the "
                            f"same group.")
                    full_remap[tid] = int(decision)
                    used_globals.add(int(decision))

            summaries = remap_ids_inplace(per_frame_dets, summaries, full_remap)
            id_to_final = {s['tid']: s['tid'] for s in summaries.values()}
            # Real founders = the ones the user assigned to a patient OR staff
            # label. Both kinds participate in training (as separate classes).
            real_founder_globals = set(v for v in founder_picks.values()
                                       if v != 'ignore')
            official_ids |= real_founder_globals   # founders are official
            founders = [summaries[gid] for gid in real_founder_globals
                        if gid in summaries]
            ignore_picks = [k for k, v in founder_picks.items()
                            if v == 'ignore']
            n_patient = sum(1 for v in real_founder_globals if v < STAFF_ID_OFFSET)
            n_staff = sum(1 for v in real_founder_globals if v >= STAFF_ID_OFFSET)
            logging.info(
                f"Founders confirmed: {n_patient} patients + {n_staff} staff = "
                f"{[format_id(v) for v in sorted(real_founder_globals)]}")
            if ignore_picks:
                logging.info(f"Ignored (no class): {len(ignore_picks)} local tracks")

            # Seed founder examples from the founding window only (= safest
            # data). Skip entirely in --no_retrain mode (= use loaded
            # classifier as-is).
            if not no_retrain:
                seeded = 0
                for f in founders:
                    pairs = []
                    for f_idx, recs in enumerate(per_frame_dets):
                        if f_idx < f_lo:
                            continue
                        if f_idx >= f_hi:
                            break
                        for r in recs:
                            if r['id'] == f['tid']:
                                pairs.append((f_idx, r['bbox']))
                                break
                    added = classifier.add_examples_from_frames(
                        video_path, pairs, label=f['tid'],
                        max_crops=FOUNDER_TRAIN_CROPS)
                    seeded += added
                logging.info(f"Seeded classifier with {seeded} founder crops")
                if classifier.train():
                    logging.info(f"Classifier trained on "
                                 f"{len(classifier.X)} examples")
                else:
                    logging.warning("Classifier could not be trained "
                                    "(not enough examples).")
            else:
                logging.info("--no_retrain: skipping founder seeding + "
                             "classifier retraining for this cam.")
        except Exception as e:
            logging.warning(f"Classifier disabled: {e}")
            classifier = None

    # Newcomers = every track that is NOT a confirmed patient from the
    # founder phase (= founders list after remapping).
    founder_tids_now = {f['tid'] for f in founders}
    newcomers = sorted(
        [s for s in summaries.values() if s['tid'] not in founder_tids_now],
        key=lambda s: s['first_frame'],
    )
    logging.info(f"Newcomers to review: {len(newcomers)}")

    # Maintain dynamic group state, indexed by current root id.
    # last_frame / last_center extend as newcomers get absorbed.
    groups = {s['tid']: dict(s) for s in summaries.values()}

    manual_decisions = []
    quit_requested = False

    for idx, newcomer in enumerate(newcomers, start=1):
        # Skip if this newcomer has been merged by an earlier decision
        nc_root = id_to_final.get(newcomer['tid'], newcomer['tid'])
        if nc_root != newcomer['tid']:
            continue

        # Build candidates: every group that STARTED before this newcomer.
        # We deliberately keep founders that are still actively tracked at
        # the newcomer's first_frame: BoT-SORT may have mis-extended a
        # track onto someone else, and the user must be able to override
        # by saying "no, that 'still alive' P3 is wrong, this newcomer is
        # the real P3". A small status badge in the title makes the
        # ambiguity visible.
        candidates = []
        for g_id, g in groups.items():
            if g_id == newcomer['tid']:
                continue
            g_root = id_to_final.get(g_id, g_id)
            if g_root != g_id:
                continue
            if g['first_frame'] >= newcomer['first_frame']:
                continue  # candidate started AFTER the newcomer = wrong direction
            s = score_pair(g, newcomer) if g['last_frame'] < newcomer['first_frame'] \
                else {'total': 0.0, 'appearance': 0.0, 'spatial': 0.0,
                      'edge': 0.0, 'd_px': 0.0,
                      'a_exit': g['exit_edge'], 'b_entry': newcomer['entry_edge']}
            candidates.append({'group': g, 'histo_score': s['total'],
                               'detail': s, 'classifier_prob': None})

        # ---- Active-learning classifier prediction ----
        if classifier is not None and classifier.trained:
            # AGGREGATED prediction over 8 evenly-spaced frames of the
            # newcomer track. Single-frame predict on repr_frame was too
            # brittle: a 500-frame track with one poor sample crop could
            # produce a low top-1, block auto-newcomer, and force the
            # user to handle a re-entry that the classifier actually has
            # plenty of evidence for. Averaging the cosine sims across N
            # samples per label gives a stable signal.
            ranked = predict_track_aggregated(
                classifier, newcomer.get('items', []),
                video_path, cap=cap, n_samples=8)
            if ranked is None:
                # Fallback to legacy single-frame for safety.
                cap.set(cv2.CAP_PROP_POS_FRAMES, newcomer.get(
                    'repr_frame', newcomer['first_frame']))
                ok, frame = cap.read()
                if ok:
                    ranked = classifier.predict(
                        frame,
                        newcomer.get('repr_bbox', newcomer['first_bbox']))
            if ranked is not None:
                prob_by_label = dict(ranked)
                for c in candidates:
                    c['classifier_prob'] = prob_by_label.get(c['group']['tid'])
                # Cross-cam safety: surface every known classifier
                # identity as a "virtual" candidate, even if no local
                # track was mapped to it on this camera. Otherwise S1
                # (= coach known from a previous cam) is invisible
                # here and the user has no way to re-attach the
                # newcomer to it.
                existing_tids = {c['group']['tid'] for c in candidates}
                known_labels = {int(l) for l in classifier.y}
                for label in sorted(known_labels - existing_tids):
                    virtual = {
                        'tid': int(label),
                        'first_frame': -1,
                        'last_frame': newcomer['first_frame'] - 1,
                        'first_center': (0.0, 0.0),
                        'last_center': (0.0, 0.0),
                        'first_bbox': [0.0, 0.0, 0.0, 0.0],
                        'last_bbox': [0.0, 0.0, 0.0, 0.0],
                        'repr_frame': -1,
                        'repr_bbox': [0.0, 0.0, 0.0, 0.0],
                        'exit_edge': 'middle',
                        'entry_edge': 'middle',
                        'embedding': None,
                        'lifespan': 0,
                        'is_virtual': True,
                        'thumb': classifier.thumbnails.get(int(label)),
                    }
                    s = {'total': 0.0, 'appearance': 0.0,
                         'spatial': 0.0, 'edge': 0.0, 'd_px': 0.0,
                         'a_exit': 'middle',
                         'b_entry': newcomer['entry_edge']}
                    candidates.append({
                        'group': virtual,
                        'histo_score': 0.0,
                        'detail': s,
                        'classifier_prob': prob_by_label.get(int(label)),
                    })

        # Rank: classifier confidence first if available + above threshold,
        # else fallback to histogram score.
        def sort_key(c):
            p = c['classifier_prob']
            if p is not None and p >= CLASSIFIER_CONF_THRESHOLD:
                return (-1.0, -p, -c['histo_score'])
            return (0.0, -c['histo_score'], -(p or 0.0))
        candidates.sort(key=sort_key)
        # Always keep every ESTABLISHED person (patient + staff label, i.e.
        # tid < STAFF_ID_OFFSET*2) in the displayed set — otherwise e.g. a
        # staff member you already created gets cut off the top-K (low
        # appearance score) and you can't pick it. Fill remaining slots
        # with the best fragment candidates, then re-sort by score.
        n_known = len({int(l) for l in classifier.y}) if (classifier and classifier.trained) else 0
        real_person = [c for c in candidates
                       if c['group']['tid'] < STAFF_ID_OFFSET * 2]
        fragments = [c for c in candidates
                     if c['group']['tid'] >= STAFF_ID_OFFSET * 2]
        n_slots = max(max_candidates, n_known)
        keep = real_person + fragments[:max(0, n_slots - len(real_person))]
        keep.sort(key=sort_key)
        keep = keep[:9]   # hard cap (candidates are keyed 1..9)
        # Re-shape into the (score, group, detail) tuple the renderer expects
        candidates = [(c['classifier_prob'] if c['classifier_prob'] is not None
                       else c['histo_score'],
                       c['group'], c['detail']) for c in keep]

        if not candidates:
            manual_decisions.append({
                'newcomer': newcomer['tid'], 'pick': None,
                'reason': 'no-candidate', 'score': None,
            })
            continue

        # Early stop once we hit the target N
        if final_n_target is not None:
            cur_n = len({id_to_final.get(t, t) for t in summaries.keys()})
            if cur_n <= final_n_target:
                logging.info(f"Reached target N={final_n_target}, stopping review")
                break

        progress = (f"Newcomer {idx}/{len(newcomers)}  "
                    f"{format_id(newcomer['tid'])}@frame {newcomer['first_frame']}  "
                    f"({len(candidates)} cands)")

        # Auto-newcomer mode: skip the popup and pick the highest-ranked
        # NON-OVERLAPPING candidate iff (a) its score >= threshold AND
        # (b) it's clearly ahead of the next-best candidate by `auto_margin`.
        # The overlap filter prevents fusing two coexisting persons; the
        # margin guard prevents accepting an ambiguous classifier guess
        # (e.g., S1=0.65, P3=0.60 — the model is hesitating, don't force).
        if auto_newcomers_thresh is not None:
            pick_idx = None
            top_safe_score = None
            for i, (score, group, _) in enumerate(candidates):
                is_safe = (group.get('is_virtual', False)
                           or group.get('last_frame', -1) < newcomer['first_frame'])
                if is_safe:
                    top_safe_score = score
                    if score >= auto_newcomers_thresh:
                        pick_idx = i
                    break

            # Margin check: gap between top-1 (= the safe pick) and the
            # next candidate of a DIFFERENT identity. We compare against
            # any other candidate (safe or not) because if the classifier
            # ranks two different people closely, the choice is ambiguous.
            margin_ok = True
            margin_blocker = None
            if pick_idx is not None:
                picked_tid = candidates[pick_idx][1]['tid']
                for j, (other_score, other_group, _) in enumerate(candidates):
                    if j == pick_idx:
                        continue
                    if other_group['tid'] == picked_tid:
                        continue  # same identity (shouldn't happen, safety)
                    if top_safe_score - other_score < auto_margin:
                        margin_ok = False
                        margin_blocker = (other_group['tid'], other_score)
                    break  # only check against the next-best

            if pick_idx is not None and margin_ok:
                key = str(pick_idx + 1)
                chosen_tid_for_log = candidates[pick_idx][1]['tid']
                logging.info(f"  AUTO: {format_id(newcomer['tid'])} -> "
                             f"{format_id(chosen_tid_for_log)} "
                             f"(top {top_safe_score:.2f})")
            else:
                # Auto refused (low score or ambiguous). Previously we
                # silently set key='n' which left the newcomer as a fresh
                # fragment forever — so a 500-frame track for which the
                # classifier just couldn't pick CONFIDENTLY would never
                # get reattached to its real identity. Now we fall back
                # to the manual popup so the user gets a chance to
                # decide. They can still press N for "new".
                if pick_idx is None:
                    if top_safe_score is None:
                        reason = "no non-overlapping candidate"
                    else:
                        reason = (f"top safe {top_safe_score:.2f} < "
                                  f"threshold {auto_newcomers_thresh:.2f}")
                else:
                    other_tid, other_score = margin_blocker
                    reason = (f"ambiguous: top {top_safe_score:.2f} vs "
                              f"{format_id(other_tid)} {other_score:.2f} "
                              f"(margin {top_safe_score - other_score:.2f} < "
                              f"{auto_margin:.2f})")
                logging.info(f"  AUTO-REFUSE → manual popup pour "
                             f"{format_id(newcomer['tid'])} ({reason})")
                fig = render_picker_figure(newcomer, candidates, cap, fps,
                                           n_frames, progress)
                decision = {'key': None}

                def on_key(event, _decision=decision, _fig=fig):
                    k = (event.key or '').lower()
                    if (k in ('q', 's', 'n', 'p', 't')
                            or (k.isdigit() and k in '0123456789')):
                        _decision['key'] = k
                        plt.close(_fig)

                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()
                key = decision['key'] or 'n'
        else:
            fig = render_picker_figure(newcomer, candidates, cap, fps,
                                       n_frames, progress)
            decision = {'key': None}

            def on_key(event):
                k = (event.key or '').lower()
                if (k in ('q', 's', 'n', 'p', 't')
                        or (k.isdigit() and k in '0123456789')):
                    decision['key'] = k
                    plt.close(fig)

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()
            key = decision['key'] or 's'

        if key == 'q':
            quit_requested = True
            manual_decisions.append({
                'newcomer': newcomer['tid'], 'pick': None,
                'reason': 'quit', 'score': None,
            })
            break
        if key == 's':
            manual_decisions.append({
                'newcomer': newcomer['tid'], 'pick': None,
                'reason': 'skip', 'score': None,
            })
            continue
        if key in ('p', 't'):
            # New patient (P) or new staff (T): allocate a fresh global
            # label in the right numeric range and assign the newcomer to it.
            all_labels = set(id_to_final.values())
            all_labels |= {f['tid'] for f in founders}
            if classifier is not None and classifier.trained:
                all_labels |= {int(l) for l in classifier.y}
            if key == 'p':
                existing = [l for l in all_labels if l < STAFF_ID_OFFSET]
                new_label = (max(existing) + 1) if existing else 1
                kind = 'new-patient'
            else:
                existing = [l for l in all_labels
                            if STAFF_ID_OFFSET <= l < STAFF_ID_OFFSET * 2]
                new_label = (max(existing) + 1) if existing else STAFF_ID_OFFSET
                kind = 'new-staff'
            old = newcomer['tid']
            id_to_final[old] = new_label
            official_ids.add(new_label)   # newly created person is official
            if old in groups:
                g = groups.pop(old)
                g['tid'] = new_label
                groups[new_label] = g
            manual_decisions.append({
                'newcomer': old, 'pick': new_label, 'reason': kind,
                'score': None,
            })
            logging.info(f"  -> {format_id(old)} = {format_id(new_label)} "
                         f"({kind})")
            continue
        if key in ('0', 'n'):
            manual_decisions.append({
                'newcomer': newcomer['tid'], 'pick': None,
                'reason': 'new-person-undecided', 'score': None,
            })
            continue
        # Number key -> pick that candidate
        pick_idx = int(key) - 1
        if pick_idx < 0 or pick_idx >= len(candidates):
            manual_decisions.append({
                'newcomer': newcomer['tid'], 'pick': None,
                'reason': 'invalid-key', 'score': None,
            })
            continue
        chosen_score, chosen_B, _ = candidates[pick_idx]
        chosen_tid = chosen_B['tid']

        # If the picked candidate is a virtual classifier-only identity
        # (no group existed on this camera yet), create its group entry
        # using the newcomer's data so the merge logic below works.
        if chosen_B.get('is_virtual'):
            groups[chosen_tid] = {
                'tid': chosen_tid,
                'first_frame': newcomer['first_frame'],
                'last_frame': newcomer['last_frame'],
                'first_center': newcomer['first_center'],
                'last_center': newcomer['last_center'],
                'first_bbox': newcomer['first_bbox'],
                'last_bbox': newcomer['last_bbox'],
                'repr_frame': newcomer.get('repr_frame', newcomer['first_frame']),
                'repr_bbox': newcomer.get('repr_bbox', newcomer['first_bbox']),
                'exit_edge': newcomer['exit_edge'],
                'entry_edge': newcomer['entry_edge'],
                'embedding': newcomer.get('embedding'),
                'lifespan': newcomer['lifespan'],
                'n_detections': newcomer.get('n_detections', 0),
                'original_ids': [chosen_tid],
            }

        # Merge: newcomer becomes part of chosen_B's group.
        # New root = the smaller (= chronologically earlier founder) id.
        root_nc = id_to_final.get(newcomer['tid'], newcomer['tid'])
        root_pick = id_to_final.get(chosen_tid, chosen_tid)
        new_root = min(root_nc, root_pick)
        old_root = max(root_nc, root_pick)
        for k, v in list(id_to_final.items()):
            if v == old_root:
                id_to_final[k] = new_root
        id_to_final[newcomer['tid']] = new_root
        id_to_final[chosen_tid] = new_root

        # Extend the kept group's state with the newcomer's data so future
        # newcomers see the updated last_frame / last_center / embedding.
        kept = groups[new_root]
        if newcomer['last_frame'] > kept['last_frame']:
            kept['last_frame'] = newcomer['last_frame']
            kept['last_center'] = newcomer['last_center']
            kept['last_bbox'] = newcomer['last_bbox']
            kept['exit_edge'] = newcomer['exit_edge']
        if kept['embedding'] is not None and newcomer['embedding'] is not None:
            wA = max(1, kept['lifespan'])
            wB = max(1, newcomer['lifespan'])
            pooled = (kept['embedding'] * wA + newcomer['embedding'] * wB) / (wA + wB)
            s = float(np.sum(pooled))
            if s > 1e-9:
                pooled /= s
            kept['embedding'] = pooled
        elif kept['embedding'] is None:
            kept['embedding'] = newcomer['embedding']
        kept['lifespan'] += newcomer['lifespan']
        if old_root in groups:
            del groups[old_root]

        manual_decisions.append({
            'newcomer': newcomer['tid'], 'pick': chosen_tid,
            'score': chosen_score, 'reason': 'matched',
        })
        logging.info(f"  -> {format_id(newcomer['tid'])} = {format_id(chosen_tid)} "
                     f"(score {chosen_score:.2f})")

        # ---- Feed the manual confirmation back into the classifier ----
        if classifier is not None and not no_retrain:
            # Add only the first N frames of this newcomer = "fresh" crops,
            # guaranteed visually-validated by the user just now.
            fresh_pairs = []
            for f_idx, recs in enumerate(per_frame_dets):
                if len(fresh_pairs) >= NEWCOMER_FRESH_CROPS:
                    break
                if f_idx < newcomer['first_frame']:
                    continue
                for r in recs:
                    if r['id'] == newcomer['tid']:
                        fresh_pairs.append((f_idx, r['bbox']))
                        break
            added = classifier.add_examples_from_frames(
                video_path, fresh_pairs, label=new_root,
                max_crops=NEWCOMER_FRESH_CROPS)
            if added > 0 and classifier.train():
                logging.info(f"     classifier updated (+{added} crops "
                             f"for P{new_root}, total "
                             f"{len(classifier.X)} examples)")

    cap.release()

    final_count = len({id_to_final.get(t, t) for t in summaries.keys()})
    logging.info("")
    logging.info("==== Manual review done ====")
    logging.info(f"  matched:    {sum(1 for d in manual_decisions if d['reason'] == 'matched')}")
    logging.info(f"  new-person: {sum(1 for d in manual_decisions if d['reason'] == 'new-person')}")
    logging.info(f"  skipped:    {sum(1 for d in manual_decisions if d['reason'] == 'skip')}")
    logging.info(f"  no-cand:    {sum(1 for d in manual_decisions if d['reason'] == 'no-candidate')}")
    logging.info(f"Final distinct persons: {final_count}")

    # Output paths
    if output_json is None:
        output_json = str(merged_json.with_name(
            merged_json.stem.replace('_merged', '') + '_final.json'))
    if output_mp4 is None:
        output_mp4 = str(merged_json.with_name(
            merged_json.stem.replace('_merged', '') + '_final.mp4'))

    # ---- Detect simultaneous duplicates BEFORE saving ----
    # (= same final ID on >1 bbox in the same frame). Most often comes
    # from picking the same global label for two distinct local tracks
    # during founder mapping or manual newcomer review.
    duplicates_per_id = {}
    for f_idx, recs in enumerate(per_frame_dets):
        seen = {}
        for r in recs:
            final_id = id_to_final.get(r['id'], r['id'])
            seen.setdefault(final_id, []).append(r['id'])
        for fid, local_tids in seen.items():
            if len(local_tids) > 1:
                duplicates_per_id.setdefault(fid, {'frames': [],
                                                   'local_tids': set()})
                duplicates_per_id[fid]['frames'].append(f_idx)
                duplicates_per_id[fid]['local_tids'].update(local_tids)

    if duplicates_per_id:
        logging.warning("=" * 60)
        logging.warning("DUPLICATE detections detected (same ID on >1 bbox/frame)")
        for fid in sorted(duplicates_per_id.keys()):
            d = duplicates_per_id[fid]
            frames = sorted(d['frames'])
            # Compact to ranges
            ranges = []
            start = prev = frames[0]
            for f in frames[1:]:
                if f - prev > 1:
                    ranges.append((start, prev))
                    start = f
                prev = f
            ranges.append((start, prev))
            range_str = ", ".join(f"{s}-{e}" if s != e else str(s)
                                  for s, e in ranges[:5])
            if len(ranges) > 5:
                range_str += f", ... ({len(ranges)} ranges total)"
            logging.warning(f"  {format_id(fid)}: {len(frames)} affected "
                            f"frames; local tids contributing = "
                            f"{sorted(d['local_tids'])}; frames {range_str}")
        logging.warning("Cause possible: deux tracks locaux ont été mappés "
                        "au meme ID global. Verifie founder mapping + review.")
        logging.warning("=" * 60)

    # ---- Re-key id_to_final by RAW track id (the ids in source_tracks_json) ----
    # `per_frame_dets` is the SAME per-frame list loaded from tracks_json,
    # mutated in place: ids were overwritten by the merger mapping, the
    # founder remap and the manual newcomer reassignments, but detections
    # are never reordered, added or removed. So a positional zip against a
    # fresh read of the raw tracks recovers raw_id -> final label
    # unambiguously. Downstream (validate_review, train_from_final) bake
    # this directly onto source_tracks_json, so it MUST be keyed by raw id
    # — not by the post-remap in-memory ids (the previous bug: founders /
    # staff like S1 vanished because their post-remap ids were absent from
    # the raw tracks).
    with open(tracks_json_path, 'r') as f:
        _raw_frames = json.load(f)['frames']
    raw_to_final = {}
    for _raw_recs, _cur_recs in zip(_raw_frames, per_frame_dets):
        for _raw_r, _cur_r in zip(_raw_recs, _cur_recs):
            _final_label = int(id_to_final.get(_cur_r['id'], _cur_r['id']))
            raw_to_final[int(_raw_r['id'])] = _final_label
    logging.info(f"Re-keyed id_to_final by raw track id: {len(raw_to_final)} "
                 f"raw tracks -> {len(set(raw_to_final.values()))} final labels")

    with open(output_json, 'w') as f:
        json.dump({
            'source_merged_json': str(merged_json),
            'source_tracks_json': str(tracks_json_path),
            'video': str(video_path),
            'id_to_final': {str(k): int(v) for k, v in raw_to_final.items()},
            'official_ids': sorted(official_ids),
            'final_n_distinct': final_count,
            'manual_decisions': manual_decisions,
            'quit_early': quit_requested,
            'duplicates_detected': {
                str(fid): {
                    'n_frames': len(d['frames']),
                    'local_tids': sorted(d['local_tids']),
                    'first_frames': sorted(d['frames'])[:10],
                }
                for fid, d in duplicates_per_id.items()
            },
        }, f, indent=2)
    logging.info(f"Final JSON saved -> {output_json}")

    # ---- Save classifier state for re-use on other cameras ----
    if classifier is None:
        pass
    elif no_retrain:
        logging.info("--no_retrain: classifier file on disk left untouched.")
    elif not classifier.trained:
        logging.info("Classifier was never trained on this cam, nothing to save.")
    else:
        if classifier_save_path is None:
            classifier_save_path = str(Path(output_json).with_name(
                Path(output_json).stem + '_classifier.pkl'))
        try:
            classifier.save(classifier_save_path)
            logging.info(f"Classifier state saved -> {classifier_save_path}")
            logging.info(f"  {len(classifier.X)} training examples on "
                         f"{len(set(classifier.y))} labels (re-usable on next cam)")
        except Exception as e:
            logging.warning(f"Could not save classifier: {e}")

    logging.info(f"Rendering final MP4 -> {output_mp4}")
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))
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
            label = format_id(final_id)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 8),
                          (int(x1) + tw + 6, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)
        n_unique = len({id_to_final.get(r['id'], r['id'])
                        for r in per_frame_dets[f_idx]})
        hud = f"frame {f_idx+1}/{n_frames}   unique persons {n_unique}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    logging.info(f"Final MP4 saved -> {output_mp4}")

    return {
        'id_to_final': id_to_final,
        'manual_decisions': manual_decisions,
        'final_count': final_count,
        'output_json': output_json,
        'output_mp4': output_mp4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--merged_json', required=True)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--output_mp4', default=None)
    parser.add_argument('-n', '--final_n_target', type=int, default=None,
                        help="Stop the review once this many distinct persons remain")
    parser.add_argument('--max_candidates', type=int,
                        default=DEFAULT_MAX_CANDIDATES,
                        help="Max nb of candidates shown per newcomer popup")
    parser.add_argument('--founding_window', type=int,
                        default=FOUNDING_WINDOW_FRAMES,
                        help="Tracks starting within first N frames are 'founders' "
                             "(the N persons of interest). Every later track must be "
                             "matched to a founder or marked as a new person.")
    parser.add_argument('--classifier_state', default=None,
                        help="Path to a .pkl saved by a previous review (same trial / "
                             "same patients) to bootstrap re-ID on this camera.")
    parser.add_argument('--classifier_save_path', default=None,
                        help="Where to save the updated classifier state at the end "
                             "(default: next to the final.json).")
    parser.add_argument('--no_classifier', action='store_true',
                        help="Disable the active-learning classifier (= histogram-only ranking).")
    parser.add_argument('--no_retrain', action='store_true',
                        help="Use the loaded classifier for predictions but do NOT add "
                             "new crops or retrain. The .pkl on disk is left untouched.")
    parser.add_argument('--auto_newcomers', type=float, default=None,
                        nargs='?', const=0.5,
                        help="Skip newcomer popups: auto-pick the top-1 candidate when "
                             "its score >= THRESH, else mark as new person. "
                             "Default 0.5 if flag is given without value.")
    parser.add_argument('--auto_margin', type=float, default=0.15,
                        help="Min gap between top-1 and top-2 classifier scores for an "
                             "auto-pick. Below this, the prediction is treated as "
                             "ambiguous and marked NEW. Default 0.15.")
    parser.add_argument('--founding_start', type=int, default=0,
                        help="Frame index where the founders window starts (default 0).")
    parser.add_argument('--backbone', default='osnet',
                        choices=['osnet', 'dinov2'],
                        help="Feature backbone. 'osnet' = legacy (fast, lighter). "
                             "'dinov2' = stronger generalisation, esp. at far "
                             "mocap distance. Default 'osnet'.")
    parser.add_argument('--dinov2_size', default='small',
                        choices=['small', 'base', 'large'],
                        help="When backbone=dinov2: ViT size. small=384-dim (fast), "
                             "base=768, large=1024 (best, slower). Default small.")
    parser.add_argument('--reid_model', default='osnet_ain_x1_0_msmt17.pt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    review_func(
        merged_json=args.merged_json,
        output_json=args.output_json,
        output_mp4=args.output_mp4,
        final_n_target=args.final_n_target,
        max_candidates=args.max_candidates,
        founding_window=args.founding_window,
        founding_start=args.founding_start,
        classifier_state=args.classifier_state,
        classifier_save_path=args.classifier_save_path,
        use_classifier=not args.no_classifier,
        no_retrain=args.no_retrain,
        auto_newcomers_thresh=args.auto_newcomers,
        auto_margin=args.auto_margin,
        backbone=args.backbone,
        dinov2_size=args.dinov2_size,
        reid_model=args.reid_model,
        device=args.device,
    )


if __name__ == '__main__':
    main()
