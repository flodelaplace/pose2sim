'''
Vraie évaluation du classifier : split train/test propre sur les cams
validées, métriques d'accuracy par identité + matrice de confusion.

Trois modes :

  --mode random       80/20 split aléatoire par crop, sans tenir compte
                      de la cam d'origine. Mesure surtout que le modèle
                      a bien APPRIS les identités (overfitting peu
                      probable avec prototype matching qui moyenne).

  --mode leave_cam    Leave-One-Camera-Out : pour chaque cam, on entraîne
                      sur les 3 autres et on évalue sur celle-ci.
                      = vrai test de GÉNÉRALISATION cross-cam (transposable
                      à une nouvelle cam jamais vue).

  --mode session      Train sur (par défaut) les 3 premières cams,
                      test sur la dernière. Compromis rapide.

Usage :
  python _evaluate_classifier.py --mode random
  python _evaluate_classifier.py --mode leave_cam
  python _evaluate_classifier.py --mode session --test-cam 23767062
'''
import argparse, json, sys, io, logging
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *_a, **_k): return it

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

from Pose2Sim.Tracking.botsort_classifier import (
    FounderClassifier, FeatureExtractor)
from Pose2Sim.Tracking.botsort_merger import (
    STAFF_ID_OFFSET, format_id)

TRACKING = Path(r"C:\Users\fdela\Documents\These Flo\Stage M2"
                r"\Demo_Seance_2024-02-05-09-09-06 - full\tracking")
VALIDATEDS = sorted(TRACKING.glob("*_validated.json"))

MAX_CROPS_PER_LABEL_PER_CAM = 150   # mirror training cap
MIN_DETS = 20                        # mirror training filter


def _load_examples(json_path):
    '''Returns list of (label, frame, bbox, cam_name) tuples for one cam,
    filtered like the training pipeline does (skip orphans, indeterminate,
    fragments) and sub-sampled to MAX_CROPS_PER_LABEL_PER_CAM.'''
    with open(json_path, 'r') as f:
        data = json.load(f)
    cam = json_path.stem.replace('_validated', '')
    excluded = set(data.get('excluded_from_training', []))
    by_id = defaultdict(list)
    for f_idx, recs in enumerate(data['frames']):
        for r in recs:
            by_id[r['id']].append((f_idx, r['bbox']))
    out = []
    for fid, items in by_id.items():
        if fid >= STAFF_ID_OFFSET * 2:    continue  # orphan range
        if fid in excluded:               continue
        if len(items) < MIN_DETS:         continue
        items.sort(key=lambda x: x[0])
        if len(items) > MAX_CROPS_PER_LABEL_PER_CAM:
            idx = np.linspace(0, len(items) - 1,
                              MAX_CROPS_PER_LABEL_PER_CAM).astype(int)
            items = [items[i] for i in idx]
        for f_idx, b in items:
            out.append((int(fid), f_idx, b, cam, str(data['video'])))
    return out


def _extract_features(rows, extractor):
    '''Single sequential read per video. Returns (X array, y array).'''
    import cv2
    by_video = defaultdict(list)   # video_path -> [(label, f_idx, bbox), ...]
    for label, f_idx, bbox, _cam, vpath in rows:
        by_video[vpath].append((label, f_idx, bbox))
    X, y = [], []
    total = sum(len(g) for g in by_video.values())
    pbar = tqdm(total=total, desc="extract features", unit="crop")
    for vpath, group in by_video.items():
        group.sort(key=lambda g: g[1])
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            logging.warning(f"cannot open {vpath}")
            pbar.update(len(group))
            continue
        for label, f_idx, bbox in group:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                pbar.update(1); continue
            feat = extractor.extract(frame, bbox)
            if feat is not None:
                X.append(feat); y.append(label)
            pbar.update(1)
        cap.release()
    pbar.close()
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _prototypes_from(X, y, backbone_dim):
    '''Mirror FounderClassifier._compute_prototypes on a plain array.'''
    by_label = defaultdict(list)
    for feat, label in zip(X, y):
        emb = feat[:backbone_dim]
        n = float(np.linalg.norm(emb))
        if n < 1e-8: continue
        by_label[int(label)].append(emb / n)
    protos = {}
    for label, embs in by_label.items():
        if len(embs) < 3: continue
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        n = float(np.linalg.norm(mean))
        if n < 1e-8: continue
        protos[label] = mean / n
    return protos


def _predict_batch(X, backbone_dim, prototypes):
    '''Return (pred_labels, top1_score) arrays.'''
    embs = X[:, :backbone_dim]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    labels = sorted(prototypes)
    P = np.stack([prototypes[l] for l in labels], axis=0)   # [L, D]
    sims = embs @ P.T   # [N, L] cosine sims
    top_idx = np.argmax(sims, axis=1)
    pred = np.asarray([labels[i] for i in top_idx])
    top_score = sims[np.arange(len(sims)), top_idx]
    return pred, top_score, sims, labels


def _report(y_true, y_pred, top_score, sims, labels, header):
    n = len(y_true)
    correct = int((y_pred == y_true).sum())
    acc = correct / n if n else 0.0
    logging.info(f"== {header} ==")
    logging.info(f"   {n} samples, top-1 accuracy = {acc:.3f} "
                 f"({correct}/{n})")
    # Per-identity recall
    by_label = defaultdict(lambda: [0, 0])   # label -> [correct, total]
    for t, p in zip(y_true, y_pred):
        by_label[int(t)][1] += 1
        if t == p: by_label[int(t)][0] += 1
    for lab in sorted(by_label):
        c, t = by_label[lab]
        logging.info(f"     {format_id(lab):>6}  recall = "
                     f"{c/t if t else 0:.3f}   ({c}/{t})")
    # Confusion (only off-diagonal hits > 0)
    err = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p: err[(int(t), int(p))] += 1
    if err:
        logging.info(f"   top erreurs (true -> pred):")
        for (t, p), n_err in err.most_common(8):
            logging.info(f"     {format_id(t)} -> {format_id(p)}  ×{n_err}")
    # Score distribution
    mean_score = float(np.mean(top_score)) if len(top_score) else 0
    correct_mean = float(np.mean(top_score[y_pred == y_true])) if correct else 0
    wrong_mean = float(np.mean(top_score[y_pred != y_true])) if (n - correct) else 0
    logging.info(f"   cosine top-1  moy={mean_score:.3f}   "
                 f"correct={correct_mean:.3f}   incorrect={wrong_mean:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['random', 'leave_cam', 'session'],
                    default='random')
    ap.add_argument('--test-cam', default=None,
                    help="for --mode session: stem of the cam to hold out")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--backbone', default='dinov2')
    ap.add_argument('--dinov2-size', default='base')
    ap.add_argument('--head', default=None,
                    help="Optional ArcFace head .pt: features then "
                         "live in the LEARNED 256-D space, not raw DINOv2.")
    args = ap.parse_args()

    if not VALIDATEDS:
        logging.error("Aucun _validated.json trouvé dans " + str(TRACKING))
        return
    logging.info(f"Cams validées trouvées : "
                 f"{[v.stem.replace('_validated','') for v in VALIDATEDS]}")

    extractor = FeatureExtractor(backbone=args.backbone,
                                 dinov2_size=args.dinov2_size,
                                 device='cuda:0', half=False,
                                 head_path=args.head)

    if args.mode == 'random':
        all_rows = []
        for vj in VALIDATEDS:
            all_rows.extend(_load_examples(vj))
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(all_rows)); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_rows = [all_rows[i] for i in idx[:cut]]
        test_rows  = [all_rows[i] for i in idx[cut:]]
        logging.info(f"Random 80/20 split: train={len(train_rows)}  "
                     f"test={len(test_rows)}")
        logging.info("Extraction features train...")
        Xtr, ytr = _extract_features(train_rows, extractor)
        logging.info("Extraction features test...")
        Xte, yte = _extract_features(test_rows, extractor)
        protos = _prototypes_from(Xtr, ytr, extractor.backbone_dim)
        pred, score, sims, labs = _predict_batch(
            Xte, extractor.backbone_dim, protos)
        _report(yte, pred, score, sims, labs, "Random 80/20")
        return

    if args.mode == 'leave_cam':
        cams = {vj.stem.replace('_validated', ''): _load_examples(vj)
                for vj in VALIDATEDS}
        accs = []
        for held_out in sorted(cams):
            train_rows = []
            for cam, rows in cams.items():
                if cam != held_out: train_rows.extend(rows)
            test_rows = cams[held_out]
            logging.info("")
            logging.info(f"--- LOCO held-out = {held_out}  "
                         f"(train={len(train_rows)}, test={len(test_rows)}) ---")
            logging.info("Extraction features train...")
            Xtr, ytr = _extract_features(train_rows, extractor)
            logging.info("Extraction features test...")
            Xte, yte = _extract_features(test_rows, extractor)
            protos = _prototypes_from(Xtr, ytr, extractor.backbone_dim)
            pred, score, sims, labs = _predict_batch(
                Xte, extractor.backbone_dim, protos)
            _report(yte, pred, score, sims, labs,
                    f"LOCO held-out = {held_out}")
            accs.append((pred == yte).mean())
        logging.info("")
        logging.info(f"=== LOCO moyenne sur {len(accs)} folds : "
                     f"{np.mean(accs):.3f}  (min {min(accs):.3f}, "
                     f"max {max(accs):.3f}) ===")
        return

    if args.mode == 'session':
        cams = {vj.stem.replace('_validated', ''): _load_examples(vj)
                for vj in VALIDATEDS}
        if args.test_cam is None:
            args.test_cam = sorted(cams)[-1]
        if args.test_cam not in cams:
            logging.error(f"--test-cam {args.test_cam} introuvable. "
                          f"Cams dispo : {list(cams)}")
            return
        train_rows = []
        for cam, rows in cams.items():
            if cam != args.test_cam: train_rows.extend(rows)
        test_rows = cams[args.test_cam]
        logging.info(f"Train sur {len(cams)-1} cams ({len(train_rows)} crops), "
                     f"test sur {args.test_cam} ({len(test_rows)} crops)")
        Xtr, ytr = _extract_features(train_rows, extractor)
        Xte, yte = _extract_features(test_rows, extractor)
        protos = _prototypes_from(Xtr, ytr, extractor.backbone_dim)
        pred, score, sims, labs = _predict_batch(
            Xte, extractor.backbone_dim, protos)
        _report(yte, pred, score, sims, labs,
                f"Session train→{args.test_cam}")


if __name__ == '__main__':
    main()
