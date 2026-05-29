'''
One-shot rebuild of trial_classifier.pkl from every _validated.json in
the tracking folder.

Use --head <arcface_head.pt> to build the pkl in the LEARNED 256-D
ArcFace embedding space instead of raw DINOv2: prototypes become quasi
orthogonal between identities and cross-cam LOCO accuracy goes from
~55 % to ~99 %.

Usage:
  python _rebuild_classifier_dinov2.py
  python _rebuild_classifier_dinov2.py --head "<tracking>/arcface_head.pt"
'''
import argparse, sys, io, logging
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

from Pose2Sim.Tracking import botsort_train_from_final
from Pose2Sim.Tracking.botsort_classifier import FounderClassifier
import numpy as np

TRACKING = Path(r"C:\Users\fdela\Documents\These Flo\Stage M2"
                r"\Demo_Seance_2024-02-05-09-09-06 - full\tracking")
PKL = TRACKING / "trial_classifier.pkl"

ap = argparse.ArgumentParser()
ap.add_argument('--head', default=None,
                help="ArcFace head .pt — produces a pkl in 256-D learned "
                     "embedding space instead of raw 768-D DINOv2.")
args = ap.parse_args()

VALIDATEDS = sorted(TRACKING.glob("*_validated.json"))
if not VALIDATEDS:
    logging.error(f"Aucun _validated.json dans {TRACKING}")
    sys.exit(1)

if PKL.exists():
    logging.info(f"Suppression de l'ancien pkl : {PKL.name}")
    PKL.unlink()

mode = (f"DINOv2 base + ArcFace head ({Path(args.head).name})"
        if args.head else "DINOv2 base brut (pas de head)")
logging.info(f"Reconstruction depuis {len(VALIDATEDS)} cams validées · {mode}")

for vj in VALIDATEDS:
    logging.info(f"== Training depuis {vj.name} ==")
    botsort_train_from_final.train_from_validated(
        validated_json=str(vj),
        classifier_pkl=str(PKL),
        backbone='dinov2',
        dinov2_size='base',
        head_path=args.head,
    )

# Sanity check: inter-prototype cosine similarity. With DINOv2 base on
# distinct people in clinical garb, expect ~0.4-0.7 between different
# patients (low = separable). Anything > 0.9 across the board would mean
# the features still don't discriminate.
logging.info("== Diagnostic prototypes ==")
c = FounderClassifier.load(str(PKL))
labels = sorted(c.prototypes)
logging.info(f"backbone={c.feature_extractor.backbone} "
             f"dinov2_size={c.feature_extractor.dinov2_size} "
             f"backbone_dim={c.feature_extractor.backbone_dim}")
logging.info(f"labels={labels}  (counts={c.n_examples_per_label()})")
for i, a in enumerate(labels):
    for b in labels[i+1:]:
        sim = float(np.dot(c.prototypes[a], c.prototypes[b]))
        logging.info(f"  {a} vs {b}: cosine = {sim:.3f}")
