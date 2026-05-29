#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Debug: are LMBN embeddings actually discriminating between different
persons in the Demo_Seance footage? Pick two co-visible tracks at the
same frame (so guaranteed to be different physical persons) and compare
their cosine similarity. If it's ~0.99, LMBN is not discriminating on
this data and we need a better ReID model. If it's <0.6, it's our
appearance code that's buggy.'''
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
import numpy as np
import cv2
from pathlib import Path

TRACKS_JSON = r"C:\Users\fdela\Documents\These Flo\Stage M2\Demo_Seance_2024-02-05-09-09-06 - full\pose\22516499_botsort_tracks.json"

import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else 'hist'
if MODE == 'hist':
    print("==> Testing torso HSV color histogram\n")
    from Pose2Sim.Tracking.botsort_merger import extract_torso_hist, histogram_similarity
    reid = None
else:
    MODEL_NAME = MODE
    print(f"==> Testing ReID model: {MODEL_NAME}\n")
    from boxmot.reid.core.reid import ReID
    reid = ReID(weights=Path(MODEL_NAME), device='cuda:0', half=False).model

with open(TRACKS_JSON) as f:
    data = json.load(f)
video_path = data['video']
frames = data['frames']

cap = cv2.VideoCapture(video_path)

# Find frames where multiple tracks co-exist
co_visible_examples = []
for f_idx, recs in enumerate(frames):
    ids = sorted({r['id'] for r in recs})
    if len(ids) >= 4:  # at least 4 co-visible persons
        co_visible_examples.append((f_idx, ids[:6], recs))
        if len(co_visible_examples) >= 3:
            break

print(f"Found {len(co_visible_examples)} co-visibility examples\n")

def cos(a, b):
    return float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

for f_idx, ids, recs in co_visible_examples:
    print(f"--- Frame {f_idx}, {len(ids)} co-visible IDs: {ids} ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok, frame = cap.read()
    if not ok:
        continue
    by_id = {r['id']: r['bbox'] for r in recs}
    if MODE == 'hist':
        feats = [extract_torso_hist(frame, by_id[tid]) for tid in ids]
        compare = histogram_similarity
        good_sep_thresh = 0.7
    else:
        bboxes = np.array([by_id[tid] for tid in ids], dtype=np.float32)
        feats = reid.get_features(bboxes, frame)
        compare = cos
        good_sep_thresh = 0.7
        print(f"  embedding shape per detection: {feats[0].shape}, dtype={feats[0].dtype}")
        print(f"  embedding norm range: {[float(np.linalg.norm(f)) for f in feats]}")
    print(f"  pairwise similarities (should be < {good_sep_thresh} for distinct persons):")
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i < j:
                sim = compare(feats[i], feats[j])
                tag = "!!!" if sim > good_sep_thresh else "OK" if sim < 0.5 else "~"
                print(f"    ID {a:>2} vs ID {b:>2}: {sim:.4f}  {tag}")
    print()

cap.release()
