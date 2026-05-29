#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #################################################################
    ## Frame-precise validation & correction of a reviewed camera  ##
    #################################################################

    After the normal review (merger_review) has assigned an identity to
    every track, this tool lets you VERIFY that each identity is correct
    throughout the whole sequence, and CORRECT any mid-track mistake. Only
    once a camera is fully validated do we trust its FULL sequence for
    training the recognition model (garbage-in-garbage-out is prevented by
    this gate).

    Workflow:
      1. For each final identity, open a frame-by-frame scrubber showing
         every detection's bbox crop. The whole track starts GREEN. The
         user paints RED the detections that are NOT this person — single
         frame (X) or [anchor..current] range (B / E).
      2. Pressing R opens the candidate picker (official people only) and
         reassigns every RED detection to the chosen target — in ONE call,
         no matter how many disjoint red ranges (handles swap-backs and
         repeated impostor flashes natively).
      3. Keys:
           ← / →           = ±1 FRAME  (intra-frame duplicates shown
                             side-by-side, not as extra arrow steps)
           PgUp / PgDn     = ±20 frames
           Home / End      = first / last frame
           Tab             = cycle focused slot when a frame has 2+ dets
           X               = toggle focused detection RED ↔ GREEN
           B / E           = set range anchor / paint [anchor..current]
                             RED on the focused slot
           U               = un-paint [anchor..current] back to GREEN
           A / C           = mark all RED / clear all to GREEN
           V               = validate entire identity (must have no reds)
           R               = open picker; all RED dets → picked target
                             (BBOX-precise: 2 dets on same frame, only
                             the red one moves)
           S               = indeterminate (kept, EXCLUDED from training)
           Q               = quit
      4. After every identity is processed, only the ones that were
         actually edited are re-scrubbed; otherwise validation completes.
      5. Output: `_validated.json` with `validated: true` and the corrected
         per-frame labels, ready for full-sequence training.

    Why green/red marking (not 10-crop filmstrip + auto-split): the old
    design suffered two failure modes — false positives in the
    histogram-derived swap boundary, and swap-back orphans absorbing late
    good frames — both producing infinite re-validation loops. Letting the
    user paint exactly the dets they reject removes all ambiguity.

    Usage:
      validate_review -f <final.json>
      from Pose2Sim.Tracking import validate_review; validate_review.validate_func(final_json=r'<path>')
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
import matplotlib
import matplotlib.pyplot as plt

from Pose2Sim.Tracking.botsort_merger import (
    extract_torso_hist, histogram_similarity, bbox_center, edge_label,
    score_pair, format_id, STAFF_ID_OFFSET,
)
from Pose2Sim.Tracking.merger_review import (
    crop_from_video, color_for_id, disable_mpl_default_keys,
)


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
FILMSTRIP_CROPS = 10            # crops shown per identity (keyable 1..9,0)
SUSPICIOUS_SIM_THRESH = 0.45    # window sim below this = highlighted crop
SUSP_WINDOW = 3                 # neighbours each side for the local mean
CROP_PAD = 0.08
SCRUB_MAX_W = 480               # pre-extracted crop max width  (display & RAM)
SCRUB_MAX_H = 720               # pre-extracted crop max height
SCRUB_JPEG_Q = 88               # in-RAM JPEG quality (good vs ~20-30 KB/crop)
SCRUB_PAGE_JUMP = 20            # PgUp/PgDn step in detections

# Per-detection marking states used by the scrubber.
MARK_OK = 0          # this det IS the current identity (green)
MARK_RED = 1         # this det is the WRONG person -> reassign on R
MARK_INDET = 2       # cannot tell who it is (occluded, half-out, ...) ->
                     # relabeled to INDETERMINATE_LABEL = excluded from
                     # training but kept in the per-frame output.
INDETERMINATE_LABEL = 8_888_888  # sentinel id; auto-added to
                                 # excluded_from_training in the validated
                                 # output so training skips these dets.


def bake_id_to_final(per_frame_dets, id_to_final):
    '''Rewrite each detection's id to its final label, in place. After
    this, det['id'] IS the final identity (no separate mapping needed),
    which lets us express frame-precise splits simply by relabelling.'''
    for recs in per_frame_dets:
        for r in recs:
            r['id'] = int(id_to_final.get(r['id'], r['id']))


def detections_by_final(per_frame_dets):
    by_id = defaultdict(list)
    for f_idx, recs in enumerate(per_frame_dets):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    return {tid: sorted(v, key=lambda x: x[0]) for tid, v in by_id.items()}


def _relabel_dets_precise(per_frame_dets, fid_items, indices, fid, target):
    '''
    Relabel the detections at the given indices of fid_items
    (BBOX-precise: matches by object id then by value, so 2 dets on the
    same frame with the same id can be moved independently). Returns the
    number of detections actually moved.
    '''
    by_frame = defaultdict(list)
    for j in indices:
        f, b = fid_items[j]
        by_frame[f].append(b)
    n_moved = 0
    for f_idx, want_bboxes in by_frame.items():
        want_ids = {id(b) for b in want_bboxes}
        want_vals = [tuple(b) for b in want_bboxes]
        for r in per_frame_dets[f_idx]:
            if r['id'] != fid:
                continue
            if (id(r['bbox']) in want_ids
                    or tuple(r['bbox']) in want_vals):
                r['id'] = target
                n_moved += 1
    return n_moved


def _crop_from_frame(frame, bbox, pad=CROP_PAD):
    '''In-memory crop (no seek), bbox padded by `pad` fraction.'''
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw)); x2 = int(min(W, x2 + pad * bw))
    y1 = int(max(0, y1 - pad * bh)); y2 = int(min(H, y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def precrop_identity(fid_items, video_path,
                     max_w=SCRUB_MAX_W, max_h=SCRUB_MAX_H,
                     jpeg_q=SCRUB_JPEG_Q):
    '''
    Sequential single-pass extraction of one identity's bbox crops, cached
    as JPEG bytes (~15-30 KB/crop) so the scrubber can step ±1 frame
    instantly. Random cv2 seeks per arrow press would otherwise stall the
    UI on long tracks. Returns a list parallel to `fid_items`, with None
    where the crop failed.

    Handles two real-world gotchas:
      - Multiple detections of the same id on the same frame (the
        `duplicates_detected` cases the merger flags): we keep a LIST of
        (idx, bbox) per frame so every entry of `fid_items` gets its crop,
        not just the last one (otherwise the earlier dupes stayed blank).
      - Intermittent decode failures mid-stream: `continue` past them
        instead of `break`, so a single bad frame doesn't leave the rest
        of the track blank.
    '''
    if not fid_items:
        return []
    first_f = fid_items[0][0]
    last_f = fid_items[-1][0]
    bbox_per_frame = defaultdict(list)
    for i, (f, b) in enumerate(fid_items):
        bbox_per_frame[f].append((i, b))
    crops = [None] * len(fid_items)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return crops
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_f)
    f = first_f
    try:
        while f <= last_f:
            ok, frame = cap.read()
            if not ok:
                # damaged / unsync frame -> skip but keep advancing so
                # later detections still get their crops.
                f += 1
                continue
            if f in bbox_per_frame:
                for idx, bbox in bbox_per_frame[f]:
                    crop = _crop_from_frame(frame, bbox)
                    if crop is None:
                        continue
                    h, w = crop.shape[:2]
                    scale = min(max_w / w, max_h / h, 1.0)
                    if scale < 1.0:
                        crop = cv2.resize(
                            crop,
                            (max(1, int(w * scale)), max(1, int(h * scale))))
                    ok2, jpg = cv2.imencode(
                        '.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                    if ok2:
                        crops[idx] = jpg.tobytes()
            f += 1
    finally:
        cap.release()
    return crops


def scrub_identity(fid, fid_items, crops_jpg, n_frames, progress=""):
    '''
    Frame-by-frame scrubber with green/red MARKING for one identity.

    The entire track starts GREEN (= belongs to this identity). The user
    paints RED the detections that are NOT this person — single det (X) or
    range [anchor..current] (B/E). On R, every RED detection is reassigned
    to the picked target.

    Frame-grouped view: when one frame carries SEVERAL detections of the
    same id (overlapping double-detection by the pose detector, or a real
    id collision = two distinct people merged to the same global), they
    are shown side-by-side as separate slots, and the user marks each
    independently. Tab cycles which slot is "focused". X / B / E act on
    the focused slot only; the relabel is BBOX-PRECISE so one slot can be
    moved while the other stays.

    Essentiel (affiché dans le titre) :
      ← / →     navigation par frame
      Tab       slot suivant (quand 2+ dets sur la même frame)
      X         rouge / annule rouge (sur le slot focus)
      I         gris (indéterminé) / annule gris
      V         valider l'identité
      R         déplacer les rouges via le picker
      Q         quitter

    Raccourcis avancés (pas affichés, voulus pour les longues plages) :
      PgUp/PgDn / ↑/↓   ± SCRUB_PAGE_JUMP frames
      Home / End        première / dernière frame
      B                 placer une ancre sur la frame courante
      E                 peindre [ancre..ici] rouge sur le slot focus
      U                 annuler [ancre..ici]
      A                 tout rouge (= identité entièrement fausse)
      C                 tout vert (efface toutes les marques)
      S                 identité entière indéterminée (ajoutée à
                        excluded_from_training en bloc)

    Returns a dict:
      {'action': 'v'|'r'|'s'|'q',
       'red_indices':   [int, ...]   # fid_items indices marqués rouges
       'indet_indices': [int, ...]}  # fid_items indices marqués gris
    '''
    n_dets = len(fid_items)
    # Group by frame -> list of (frame, [(fid_idx, bbox), ...]) preserving
    # ascending order. The "view index" iterates frames; "slot" iterates
    # the dets within the current frame.
    by_frame = defaultdict(list)
    for j, (f, b) in enumerate(fid_items):
        by_frame[f].append((j, b))
    frames_view = sorted(by_frame.items(), key=lambda x: x[0])
    n_view = len(frames_view)

    state = {'i': 0, 'slot': 0}
    marked = [MARK_OK] * n_dets     # per detection (0/1/2 tri-state)
    anchor = {'i': None, 'slot': 0}  # frame-view anchor + slot
    decision = {'action': None, 'red_indices': [], 'indet_indices': []}

    # Layout: large crop area on top, thin timeline strip below.
    fig = plt.figure(figsize=(7.5, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[22, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    try:
        fig.canvas.manager.set_window_title(f"Scrubber {format_id(fid)}")
    except Exception:
        pass
    ax_bar.set_xticks([]); ax_bar.set_yticks([])

    def _decode_crop(fid_idx):
        buf = crops_jpg[fid_idx] if fid_idx < len(crops_jpg) else None
        if buf is None:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def render():
        i = state['i']
        f, slots = frames_view[i]
        # Clamp focus to the current frame's slot count for DISPLAY only;
        # the user's intended slot (state['slot']) is preserved across
        # navigation so a range paint started on slot 1 keeps painting
        # slot 1 even after crossing frames that only have 1 slot.
        focus_slot = min(state['slot'], len(slots) - 1) if slots else 0

        # ----- crops side by side -----
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])

        crops = []
        for fid_idx, _b in slots:
            img = _decode_crop(fid_idx)
            if img is None:
                img = np.full((300, 200, 3), 60, dtype=np.uint8)
                cv2.putText(img, "crop?", (40, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            crops.append(img)

        # Pad all crops to the same height for hstack.
        max_h = max(c.shape[0] for c in crops)
        sep_w = 12
        padded = []
        for c in crops:
            if c.shape[0] != max_h:
                pad = np.zeros((max_h - c.shape[0], c.shape[1], 3),
                               dtype=np.uint8)
                c = np.vstack([c, pad])
            padded.append(c)
        # Build a list of (start_x, end_x) per slot for the focus rectangle.
        x_offsets = []
        composed = []
        x = 0
        for k, c in enumerate(padded):
            if k > 0:
                composed.append(np.full((max_h, sep_w, 3), 200,
                                        dtype=np.uint8))
                x += sep_w
            composed.append(c)
            x_offsets.append((x, x + c.shape[1]))
            x += c.shape[1]
        composed = np.hstack(composed)
        ax.imshow(composed)

        # Per-slot border rectangles. Colour reflects the slot's state:
        # red = wrong (reassign on R), grey = indeterminate (kept but
        # excluded from training), identity colour = green/OK.
        from matplotlib.patches import Rectangle
        for k, ((fid_idx, _b), (x0, x1)) in enumerate(zip(slots, x_offsets)):
            st = marked[fid_idx]
            if st == MARK_RED:
                colour = (0.85, 0.10, 0.10)
            elif st == MARK_INDET:
                colour = (0.55, 0.55, 0.55)
            else:
                colour = tuple(v / 255.0 for v in color_for_id(fid)[::-1])
            rect = Rectangle((x0, 0), x1 - x0, max_h,
                             linewidth=4, edgecolor=colour, facecolor='none')
            ax.add_patch(rect)
            if k == focus_slot:
                yellow = Rectangle((x0 - 2, -2), x1 - x0 + 4, max_h + 4,
                                   linewidth=2.5, edgecolor='gold',
                                   facecolor='none', linestyle='--')
                ax.add_patch(yellow)

        n_red = sum(1 for m in marked if m == MARK_RED)
        n_indet = sum(1 for m in marked if m == MARK_INDET)
        slot_str = (f"   slot {focus_slot+1}/{len(slots)}"
                    if len(slots) > 1 else "")
        ax.set_title(
            f"{progress}   {format_id(fid)}   frame {f}/{n_frames-1}"
            f"{slot_str}   ({n_red} rouges · {n_indet} gris / {n_dets})\n"
            f"← →  navigue   |   Tab  change de slot   |   "
            f"X=rouge   I=gris   V=valider   R=déplacer rouges   Q=quit",
            fontsize=11, weight='bold')

        # ----- timeline strip: tri-state per frame -----
        # green = all OK, red = all red, grey = all indeterminate,
        # orange = mixed (any combination of states).
        ax_bar.clear()
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        strip = np.zeros((1, n_view, 3), dtype=np.float32)
        for k, (_f, sl) in enumerate(frames_view):
            states = {marked[fi] for (fi, _b) in sl}
            if states == {MARK_OK}:
                strip[0, k] = (0.10, 0.70, 0.10)
            elif states == {MARK_RED}:
                strip[0, k] = (0.85, 0.10, 0.10)
            elif states == {MARK_INDET}:
                strip[0, k] = (0.55, 0.55, 0.55)
            else:
                strip[0, k] = (0.95, 0.60, 0.10)   # mixed
        ax_bar.imshow(strip, aspect='auto',
                      extent=[0, n_view, 0, 1], interpolation='nearest')
        ax_bar.set_xlim(0, n_view); ax_bar.set_ylim(0, 1)
        ax_bar.axvline(i + 0.5, color='royalblue', lw=2)
        if anchor['i'] is not None:
            ax_bar.axvline(anchor['i'] + 0.5, color='gold', lw=2)
        fig.canvas.draw_idle()

    def paint_range(value):
        '''Set the (desired-slot) det across [anchor..current] to `value`
        (one of MARK_OK / MARK_RED). The "desired slot" is whichever the
        user last Tabbed onto — preserved across navigation. For frames
        with fewer slots than that index, fall back to the last slot.'''
        i = state['i']; s = state['slot']
        a = anchor['i']
        if a is None:
            _f, slots = frames_view[i]
            if not slots:
                return
            idx = min(s, len(slots) - 1)
            marked[slots[idx][0]] = value
            return
        lo, hi = (a, i) if a <= i else (i, a)
        for k in range(lo, hi + 1):
            _f, slots = frames_view[k]
            if not slots:
                continue
            idx = min(s, len(slots) - 1)
            marked[slots[idx][0]] = value

    def toggle_state(target_state):
        '''X / I toggle: if already in `target_state` go back to OK, else
        switch into it. So X on a grey turns it red, and I on a red turns
        it grey.'''
        _f, slots = frames_view[state['i']]
        if not slots:
            return
        idx = min(state['slot'], len(slots) - 1)
        fid_idx = slots[idx][0]
        marked[fid_idx] = (MARK_OK if marked[fid_idx] == target_state
                           else target_state)

    def on_key(event):
        k = (event.key or '').lower()
        # Slot intent is PRESERVED across navigation (clamped at render).
        # That way a B-then-E range painting on slot 1 keeps painting slot
        # 1 across all the frames in the range — even after crossing
        # single-slot frames.
        if k == 'left':
            state['i'] = max(0, state['i'] - 1); render()
        elif k == 'right':
            state['i'] = min(n_view - 1, state['i'] + 1); render()
        elif k in ('pageup', 'up'):
            state['i'] = max(0, state['i'] - SCRUB_PAGE_JUMP); render()
        elif k in ('pagedown', 'down'):
            state['i'] = min(n_view - 1, state['i'] + SCRUB_PAGE_JUMP)
            render()
        elif k == 'home':
            state['i'] = 0; render()
        elif k == 'end':
            state['i'] = n_view - 1; render()
        elif k == 'tab':
            _f, slots = frames_view[state['i']]
            state['slot'] = (state['slot'] + 1) % max(1, len(slots))
            render()
        elif k == 'x':
            toggle_state(MARK_RED); render()
        elif k == 'i':
            toggle_state(MARK_INDET); render()
        elif k == 'b':
            anchor['i'] = state['i']; anchor['slot'] = state['slot']; render()
        elif k == 'e':
            paint_range(MARK_RED); render()
        elif k == 'u':
            paint_range(MARK_OK); render()
        elif k == 'a':
            for j in range(n_dets): marked[j] = MARK_RED
            render()
        elif k == 'c':
            for j in range(n_dets): marked[j] = MARK_OK
            anchor['i'] = None
            render()
        elif k in ('v', 'r', 's', 'q'):
            decision['action'] = k
            decision['red_indices'] = [j for j, m in enumerate(marked)
                                       if m == MARK_RED]
            decision['indet_indices'] = [j for j, m in enumerate(marked)
                                         if m == MARK_INDET]
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.subplots_adjust(top=0.84, bottom=0.05, left=0.04, right=0.99)
    render()
    plt.show()
    return decision


def sample_filmstrip(items, n=FILMSTRIP_CROPS):
    '''Evenly-spaced (frame_idx, bbox) samples across a track's lifespan.'''
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


def extract_histograms_single_pass(needed_per_frame, video_path):
    '''(frame -> [(key, bbox)]) -> {key: hist} in one sequential read.'''
    if not needed_per_frame:
        return {}
    last = max(needed_per_frame.keys())
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    f_idx = 0
    while f_idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx in needed_per_frame:
            for key, bbox in needed_per_frame[f_idx]:
                h = extract_torso_hist(frame, bbox)
                if h is not None:
                    out[key] = h
        f_idx += 1
    cap.release()
    return out


def find_suspicious_in_filmstrip(strip, hist_by_key, key_prefix):
    '''Return the set of filmstrip indices whose torso histogram deviates
    from the local rolling mean (= likely a different person there).'''
    hists = []
    for i, (f_idx, _bbox) in enumerate(strip):
        hists.append(hist_by_key.get((key_prefix, f_idx)))
    suspicious = set()
    n = len(hists)
    for i in range(n):
        if hists[i] is None:
            continue
        # local mean of the OTHER crops in a small window
        neigh = [hists[j] for j in range(max(0, i - SUSP_WINDOW),
                                         min(n, i + SUSP_WINDOW + 1))
                 if j != i and hists[j] is not None]
        if not neigh:
            continue
        mean = np.mean(np.stack(neigh, axis=0), axis=0)
        s = float(np.sum(mean))
        if s > 1e-9:
            mean = mean / s
        if histogram_similarity(hists[i], mean) < SUSPICIOUS_SIM_THRESH:
            suspicious.add(i)
    return suspicious


def find_swap_segment(interval_items, items_after, video_path,
                      flagged_frame_fallback=None):
    '''
    Locate the CONTIGUOUS impostor segment around a user-flagged crop.

    The interval [prev_frame..flagged_frame] goes (chronologically) from the
    GOOD person to a WRONG person. We need BOTH bounds of the impostor:

      - bad_start: first frame whose appearance diverges from the reference
                   (good person, built from the start of the interval).
      - bad_end:   last frame of the impostor run BEFORE the appearance
                   returns to the reference. If it never returns, bad_end is
                   the identity's last frame (= old "split to end" behavior).

    Returns (bad_start, bad_end), or (None, None) if too little data.

    Why both bounds: with only bad_start the orphan absorbed everything to
    end-of-track, which broke two real cases user hit on Demo_Seance:

      1. SWAP-BACK (A..A,B,A..A): the late good frames were dragged into the
         orphan, and the picker showed person A (the majority of the orphan)
         instead of B -> impossible to separate B.
      2. HISTOGRAMS TOO SIMILAR (similar dark clothing): bad_start fell back
         to flagged_frame, only that single sampled crop moved, the rest of
         the impostor stayed in the track -> next pass re-showed the same
         crop, infinite loop.

    `flagged_frame_fallback` (the user's clicked crop frame) is trusted as
    bad_start when no clear divergence is found in the interval, so at
    least the user's explicit signal moves a real segment out.
    '''
    if not interval_items:
        return (None, None)
    # Single-pass hist read covers the whole interval + the rest of the
    # identity, so we can also scan forward for the return-to-good frame.
    all_items = list(interval_items) + list(items_after)
    needed = defaultdict(list)
    for f_idx, bbox in all_items:
        needed[f_idx].append((('s', f_idx), bbox))
    hist = extract_histograms_single_pass(needed, video_path)
    interval_hists = [(f, hist.get(('s', f))) for f, _ in interval_items]
    interval_hists = [(f, h) for f, h in interval_hists if h is not None]
    if len(interval_hists) < 3:
        last = interval_items[-1][0]
        return (last, last)
    k = max(1, min(3, len(interval_hists) // 3))
    ref = np.mean(np.stack([h for _, h in interval_hists[:k]], axis=0), axis=0)
    s = float(np.sum(ref))
    if s > 1e-9:
        ref = ref / s

    # bad_start = first divergent frame in the interval; fall back to the
    # user-flagged crop if no clear divergence (similar clothing case).
    bad_start = None
    for f, h in interval_hists[k:]:
        if histogram_similarity(h, ref) < SUSPICIOUS_SIM_THRESH:
            bad_start = f
            break
    if bad_start is None:
        bad_start = (flagged_frame_fallback if flagged_frame_fallback is not None
                     else interval_items[-1][0])

    # bad_end: walk forward from bad_start across the identity; the segment
    # ends at the last frame BEFORE a return to the good reference.
    bad_end = bad_start
    for f, _ in all_items:
        if f < bad_start:
            continue
        h = hist.get(('s', f))
        if h is None:
            bad_end = f  # missing hist -> keep extending conservatively
            continue
        if histogram_similarity(h, ref) >= SUSPICIOUS_SIM_THRESH:
            break  # back to good person
        bad_end = f
    return (bad_start, bad_end)


def find_swap_boundary(interval_items, video_path):
    '''Back-compat shim returning only bad_start (no swap-back handling).'''
    bad_start, _ = find_swap_segment(interval_items, [], video_path)
    return bad_start


def rank_reassign_candidates(fid, fid_items, by_now, width, height, video_path,
                             official_ids=None):
    '''
    Rank candidate identities by torso-colour similarity to the current
    identity, so the visually-matching target (e.g. the coach S1) surfaces
    at the top. If `official_ids` is given, ONLY those established persons
    (founders P1.., S1.. + new patient/staff created in review) are
    proposed — fragments are never merge targets, only things to be merged
    INTO an official person. Returns list of (other_id, rep_(frame,bbox), overlaps).
    '''
    o_first, o_last = fid_items[0][0], fid_items[-1][0]
    if official_ids:
        cand_ids = [o for o in by_now
                    if o != fid and o in official_ids]
    else:
        cand_ids = [o for o in by_now
                    if o != fid and o < STAFF_ID_OFFSET * 2]
    if not cand_ids:
        return []
    cur_sample = sample_filmstrip(fid_items, 8)
    cand_rep = {o: max(by_now[o],
                       key=lambda it: representative_quality(it[1], width, height))
                for o in cand_ids}
    needed = defaultdict(list)
    for f, b in cur_sample:
        needed[f].append((('cur', f), b))
    for o, (f, b) in cand_rep.items():
        needed[f].append((('cand', o), b))
    hist = extract_histograms_single_pass(needed, video_path)
    cur_hists = [hist.get(('cur', f)) for f, _ in cur_sample]
    cur_hists = [h for h in cur_hists if h is not None]
    if cur_hists:
        cur_mean = np.mean(np.stack(cur_hists, axis=0), axis=0)
        s = float(np.sum(cur_mean))
        if s > 1e-9:
            cur_mean = cur_mean / s

        def sim(o):
            h = hist.get(('cand', o))
            return histogram_similarity(h, cur_mean) if h is not None else 0.0
        order = sorted(cand_ids, key=lambda o: -sim(o))
    else:
        order = sorted(cand_ids)
    # Staff identities (>= STAFF_ID_OFFSET) are few and important — always
    # surface them FIRST so they never get cut off by similar-looking
    # patient fragments.
    staff = [o for o in order if o >= STAFF_ID_OFFSET]
    non_staff = [o for o in order if o < STAFF_ID_OFFSET]
    order = staff + non_staff
    out = []
    for o in order:
        overlaps = any(o_first <= fi <= o_last for fi, _ in by_now[o])
        out.append((o, cand_rep[o], overlaps))
    return out


def render_filmstrip(final_id, strip, suspicious, cap, n_frames, progress):
    '''Grid of crops for one identity; suspicious crops get a red border.'''
    n = len(strip)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= n:
            ax.axis('off')
            continue
        f_idx, bbox = strip[i]
        crop = crop_from_video(cap, f_idx, bbox, pad=CROP_PAD)
        if crop is not None:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        edge = 'red' if i in suspicious else 'limegreen'
        lw = 4 if i in suspicious else 2
        for sp in ax.spines.values():
            sp.set_edgecolor(edge); sp.set_linewidth(lw)
        key_label = (i + 1) % 10  # 1..9 then 0
        flag = "  ⚠" if i in suspicious else ""
        ax.set_title(f"[{key_label}] f{f_idx}{flag}", fontsize=9)
    fig.suptitle(
        f"{progress}   identité {format_id(final_id)}   "
        f"({n} vues)\n"
        f"V=valider    1..9/0=1er crop FAUX (split)    "
        f"R=c'est en fait qqn d'autre (fusion)    "
        f"S=indéterminé (hors entraînement)    Q=quit",
        fontsize=10, weight='bold')
    plt.subplots_adjust(top=0.84, bottom=0.03, left=0.03, right=0.99,
                        hspace=0.25, wspace=0.05)
    return fig


def render_reassign(orphan_id, orphan_strip, candidates, cap, progress):
    '''Picker for a flagged (orphan) segment: left = the orphan crops,
    right = candidate identities (their representative crop). Returns the
    figure; keys handled by the caller.'''
    ncols = 1 + len(candidates)
    fig, axes = plt.subplots(1, max(2, ncols), figsize=(2.3 * max(2, ncols), 4.0))
    axes = np.atleast_1d(axes).ravel()
    # Left: a representative orphan crop
    mid = orphan_strip[len(orphan_strip) // 2]
    crop = crop_from_video(cap, mid[0], mid[1], pad=CROP_PAD)
    ax = axes[0]
    if crop is not None:
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('orange'); sp.set_linewidth(4)
    ax.set_title(f"SEGMENT à réassigner\nf{orphan_strip[0][0]}-{orphan_strip[-1][0]}",
                 fontsize=10, weight='bold')
    for i, (cand_id, cand_repr) in enumerate(candidates):
        ax = axes[1 + i]
        crop = crop_from_video(cap, cand_repr[0], cand_repr[1], pad=CROP_PAD)
        if crop is not None:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        c = tuple(v / 255.0 for v in color_for_id(cand_id)[::-1])
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(3)
        ax.set_title(f"[{i+1}]  {format_id(cand_id)}", fontsize=10)
    fig.suptitle(f"{progress}   1..{len(candidates)}=c'est cette personne   "
                 f"P=nouv.patient   T=nouv.staff   S=garder séparé   Q=quit",
                 fontsize=10, weight='bold')
    plt.subplots_adjust(top=0.80, bottom=0.03, left=0.02, right=0.99, wspace=0.05)
    return fig


def _wait_key(fig, allowed):
    decision = {'key': None}

    def on_key(event):
        k = (event.key or '').lower()
        if k in allowed:
            decision['key'] = k
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return decision['key']


MIN_DETS_TO_VALIDATE = 10       # identities shorter than this aren't worth
                                # a frame-precise check (too brief to swap)


def validate_func(final_json, output_json=None, device='cuda:0',
                  reid_model='osnet_ain_x1_0_msmt17.pt',
                  min_dets_to_validate=MIN_DETS_TO_VALIDATE):
    final_json = Path(final_json)
    with open(final_json, 'r') as f:
        final_data = json.load(f)
    tracks_json = Path(final_data['source_tracks_json'])
    with open(tracks_json, 'r') as f:
        tracks_data = json.load(f)

    video_path = Path(tracks_data['video'])
    per_frame_dets = tracks_data['frames']
    width, height = tracks_data['width'], tracks_data['height']
    fps = float(tracks_data['fps'])
    n_frames = tracks_data['n_frames']
    id_to_final = {int(k): int(v) for k, v in final_data['id_to_final'].items()}
    # Sanity: id_to_final MUST be keyed by raw track id (the ids present in
    # source_tracks_json). Old final.json files were keyed by post-remap
    # in-memory ids, so almost no key matched -> founders/staff vanished and
    # the sequence stayed fragmented. Detect and refuse to proceed on those.
    _raw_ids = {r['id'] for recs in per_frame_dets for r in recs}
    _match = sum(1 for k in id_to_final if k in _raw_ids)
    if id_to_final and _match < 0.5 * len(_raw_ids):
        raise SystemExit(
            f"final.json id_to_final n'est pas keyé par id de track brut "
            f"({_match}/{len(_raw_ids)} ids correspondent). C'est un ANCIEN "
            f"format produit avant le fix de re-keying. Relance la review "
            f"(stage 6) pour régénérer le final.json, puis la validation.")
    # The official people established at review (founders + new P/T). Only
    # these are valid merge targets in R. If absent (old final.json), we
    # fall back to "all real ids" and warn.
    official_ids = set(final_data.get('official_ids', []))
    if not official_ids:
        logging.warning("No 'official_ids' in this final.json (ancien format). "
                        "R proposera toutes les identités. Pour ne proposer que "
                        "les personnes initialisées au founder, refais un run "
                        "complet (review fraîche).")

    # Bake the mapping so detection ids ARE the final labels.
    bake_id_to_final(per_frame_dets, id_to_final)

    disable_mpl_default_keys()
    cap = cv2.VideoCapture(str(video_path))

    # ID allocation for orphan (flagged) segments
    next_orphan = max([r['id'] for recs in per_frame_dets for r in recs],
                      default=0) + 1
    next_orphan = max(next_orphan, STAFF_ID_OFFSET * 2)  # keep clear of P/S ranges

    quit_requested = False
    pass_num = 0
    # Identities the user already validated/skipped and that have NOT been
    # touched by a later correction -> skip them on subsequent passes so we
    # only re-review what actually changed.
    done_ids = set()
    # Identities marked indeterminate (S): kept in the output (still
    # rendered) but EXCLUDED from training — e.g. a bbox that sits on
    # several overlapping people, unusable to learn an identity from.
    indeterminate_ids = set()
    while not quit_requested:
        pass_num += 1
        by_final = detections_by_final(per_frame_dets)
        # Only validate "real" persons (P/S range), not orphan temporaries,
        # and only those long enough to be worth a frame-precise check.
        real_ids = sorted([t for t in by_final
                           if t < STAFF_ID_OFFSET * 2
                           and len(by_final[t]) >= min_dets_to_validate])
        if pass_num == 1:
            n_short = sum(1 for t in by_final
                          if t < STAFF_ID_OFFSET * 2
                          and len(by_final[t]) < min_dets_to_validate)
            if n_short:
                logging.info(f"  ({n_short} micro-tracks < {min_dets_to_validate} "
                             f"détections ignorés de la validation)")
        # Skip the ones already validated/skipped and untouched since
        ids_to_review = [fid for fid in real_ids if fid not in done_ids]

        if not ids_to_review:
            logging.info(f"Pass {pass_num}: nothing left to review -> done.")
            break

        corrections = 0
        for k, fid in enumerate(ids_to_review, start=1):
            fid_items = by_final[fid]
            progress = f"Pass {pass_num} · {k}/{len(ids_to_review)}"
            # Sequential single-pass crop extraction (instant scrubbing).
            logging.info(f"  {progress}: pre-extraction crops {format_id(fid)} "
                         f"({len(fid_items)} dets)...")
            crops_jpg = precrop_identity(fid_items, video_path)
            decision = scrub_identity(fid, fid_items, crops_jpg, n_frames,
                                      progress=progress)
            action = decision['action']
            indet_indices = decision.get('indet_indices', [])
            # Indeterminate dets (grey) get relabeled to the sentinel id
            # regardless of which top-level action ended the scrubber —
            # they're a per-det decision, not tied to V vs R vs Q.
            if indet_indices and action != 's':
                n_i = _relabel_dets_precise(
                    per_frame_dets, fid_items, indet_indices, fid,
                    INDETERMINATE_LABEL)
                if n_i:
                    corrections += 1
                    logging.info(f"  {format_id(fid)}: {n_i} dets gris -> "
                                 f"INDET (exclus du training)")
            if action == 'q':
                quit_requested = True
                break
            if action == 'v':
                done_ids.add(fid)
                indeterminate_ids.discard(fid)
                continue
            if action == 's':
                done_ids.add(fid)
                indeterminate_ids.add(fid)
                continue
            if action == 'r':
                # User painted some detections RED on the scrubber timeline.
                # Reassign ALL red detections (possibly disjoint ranges -
                # e.g. swap-back A,B,A,B,A) to a single picked target.
                red_indices = decision['red_indices']
                if not red_indices:
                    logging.info(f"  {format_id(fid)}: aucune frame marquée "
                                 f"en rouge — rien à ré-assigner.")
                    continue
                sub_items = [fid_items[j] for j in red_indices]
                red_frames = {f for f, _ in sub_items}
                by_now = detections_by_final(per_frame_dets)
                ranked = rank_reassign_candidates(
                    fid, sub_items, by_now, width, height, video_path,
                    official_ids=official_ids)
                if not ranked:
                    logging.info(f"  {format_id(fid)}: pas de candidat pour "
                                 f"les {len(red_indices)} frames rouges.")
                    continue
                cand = [(o, rep) for o, rep, _ov in ranked[:6]]
                o_strip = sample_filmstrip(sub_items, 6)
                fig2 = render_reassign(
                    fid, o_strip, cand, cap,
                    f"Ré-assigner {len(red_indices)} dets rouges de "
                    f"{format_id(fid)}")
                k2 = _wait_key(fig2, set('s q 0 n p t 1 2 3 4 5 6'.split()))
                if k2 == 'q':
                    quit_requested = True
                    break
                if k2 in ('s', '0', 'n', None):
                    continue  # cancelled -> identity untouched
                target = None
                if k2 in ('p', 't'):
                    # Create a NEW patient (P) or staff (T) label and assign
                    # the red dets to it (use when no existing candidate fits).
                    all_ids = {r['id'] for recs in per_frame_dets for r in recs}
                    if k2 == 'p':
                        pats = [i for i in all_ids if i < STAFF_ID_OFFSET]
                        target = (max(pats) + 1) if pats else 1
                    else:
                        stf = [i for i in all_ids
                               if STAFF_ID_OFFSET <= i < STAFF_ID_OFFSET * 2]
                        target = (max(stf) + 1) if stf else STAFF_ID_OFFSET
                else:
                    pick = int(k2) - 1
                    if 0 <= pick < len(cand):
                        target = cand[pick][0]
                if target is None:
                    continue
                if k2 in ('p', 't'):
                    official_ids.add(target)
                # BBOX-precise: only the actually-red bbox(es) move, even
                # if a frame has another det of the same id staying green.
                n_moved = _relabel_dets_precise(
                    per_frame_dets, fid_items, red_indices, fid, target)
                f_min = min(f for f, _ in (fid_items[j] for j in red_indices))
                f_max = max(f for f, _ in (fid_items[j] for j in red_indices))
                done_ids.discard(fid)      # remaining part changed
                done_ids.discard(target)   # target gained frames -> re-check
                corrections += 1
                logging.info(f"  {format_id(fid)}: {n_moved} dets rouges "
                             f"(frames {f_min}-{f_max}) -> {format_id(target)}")
                continue

        if quit_requested:
            break

        if corrections == 0:
            logging.info(f"Pass {pass_num}: no correction -> validation complete.")
            break
        else:
            logging.info(f"Pass {pass_num}: {corrections} corrections; "
                         f"re-verifying...")

    cap.release()

    # ---- Write validated output ----
    if output_json is None:
        output_json = str(final_json.with_name(
            final_json.stem.replace('_final', '') + '_validated.json'))
    # Build identity mapping (now baked into per_frame_dets)
    final_ids = sorted({r['id'] for recs in per_frame_dets for r in recs})
    excluded = set(indeterminate_ids)
    # Per-det indeterminacies carry the sentinel id; auto-add to excluded.
    if INDETERMINATE_LABEL in final_ids:
        excluded.add(INDETERMINATE_LABEL)
    excluded = sorted(excluded)
    with open(output_json, 'w') as f:
        json.dump({
            'source_final_json': str(final_json),
            'source_tracks_json': str(tracks_json),
            'video': str(video_path),
            'width': width, 'height': height, 'fps': fps,
            'n_frames': n_frames,
            'validated': not quit_requested,
            'final_ids': final_ids,
            'excluded_from_training': excluded,
            'frames': per_frame_dets,
        }, f)
    logging.info(f"Validated JSON saved -> {output_json}  "
                 f"(validated={not quit_requested}, {len(final_ids)} identities, "
                 f"{len(excluded)} indéterminées exclues de l'entraînement)")
    return {'output_json': output_json, 'validated': not quit_requested,
            'final_ids': final_ids, 'excluded_from_training': excluded}


def representative_quality(bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    edge = min(cx / width, (width - cx) / width, cy / height, (height - cy) / height)
    area = min((bw * bh) / (width * height) * 50.0, 1.0)
    aspect = bh / bw
    asp = 1.0 if 1.5 <= aspect <= 4.0 else 0.6
    return edge * area * asp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--final_json', required=True,
                        help="The _final.json produced by merger_review / pipeline.")
    parser.add_argument('-o', '--output_json', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reid_model', default='osnet_ain_x1_0_msmt17.pt')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    validate_func(final_json=args.final_json, output_json=args.output_json,
                  device=args.device, reid_model=args.reid_model)


if __name__ == '__main__':
    main()
