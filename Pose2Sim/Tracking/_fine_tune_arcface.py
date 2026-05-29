'''
Fine-tune une tête ArcFace au-dessus de DINOv2 base sur les crops
validés (cams 1..N) pour rendre le re-ID réellement discriminant entre
identités spécifiques.

Pourquoi : DINOv2 zero-shot donne des cosines 0.84-0.94 entre tous tes
patients (LOCO accuracy 54 %) — il "voit" tous tes sujets en blouse
clinique comme la même personne ±5 %. ArcFace impose explicitement
"même personne -> embeddings proches, personnes différentes -> embeddings
loin", avec une marge angulaire sur la vraie classe. Standard ReID.

Pipeline :
  1. Charge tous les _validated.json du dossier tracking/.
  2. Extraction features DINOv2 base de chaque crop (cache .npz pour les
     re-runs ; suppression auto si --no-cache).
  3. Split 80/20 (stratifié par identité = chaque label dans val).
  4. Entraîne une projection head + ArcFace classifier.
  5. Évalue par cosine-prototype matching à chaque epoch et garde le
     meilleur checkpoint (val_accuracy).
  6. Sauve <tracking>/arcface_head.pt + diagnostic inter-prototypes.

Le résultat est intégré au FeatureExtractor : passer head_path à
FeatureExtractor charge la tête et fait extract() renvoyer le 256-D
ArcFace au lieu du 768-D DINOv2 brut.

Usage (depuis C:\\Users\\fdela\\Projects\\pose2sim) :
  python Pose2Sim\\Tracking\\_fine_tune_arcface.py --epochs 80
'''
import argparse, json, sys, io, logging, math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it=None, *a, **k):
        return it if it is not None else range(0)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

from Pose2Sim.Tracking.botsort_classifier import FeatureExtractor
from Pose2Sim.Tracking.botsort_merger import STAFF_ID_OFFSET, format_id

TRACKING = Path(r"C:\Users\fdela\Documents\These Flo\Stage M2"
                r"\Demo_Seance_2024-02-05-09-09-06 - full\tracking")
CACHE = TRACKING / "_arcface_features_cache.npz"
HEAD_OUT = TRACKING / "arcface_head.pt"

MAX_CROPS_PER_LABEL_PER_CAM = 150
MIN_DETS = 20
EMBED_DIM = 256
PROJ_HIDDEN = 512


def _load_examples(json_path):
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
        if fid >= STAFF_ID_OFFSET * 2 or fid in excluded or len(items) < MIN_DETS:
            continue
        items.sort(key=lambda x: x[0])
        if len(items) > MAX_CROPS_PER_LABEL_PER_CAM:
            idx = np.linspace(0, len(items) - 1,
                              MAX_CROPS_PER_LABEL_PER_CAM).astype(int)
            items = [items[i] for i in idx]
        for f_idx, b in items:
            out.append((int(fid), f_idx, b, cam, str(data['video'])))
    return out


def _extract_dinov2_features(rows, extractor):
    '''Returns X[backbone_dim], y, cam_ids — DINOv2 ONLY (no color/shape).'''
    import cv2
    by_video = defaultdict(list)
    for label, f_idx, bbox, cam, vpath in rows:
        by_video[vpath].append((label, f_idx, bbox, cam))
    Xs, ys, cams = [], [], []
    pbar = tqdm(total=sum(len(g) for g in by_video.values()),
                desc="DINOv2 forward", unit="crop")
    for vpath, group in by_video.items():
        group.sort(key=lambda g: g[1])
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            pbar.update(len(group)); continue
        for label, f_idx, bbox, cam in group:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                pbar.update(1); continue
            # We need only the BACKBONE (not concat with color/shape).
            extractor._ensure_loaded()
            backbone = (extractor._extract_dinov2(frame, bbox)
                        if extractor.backbone == 'dinov2'
                        else extractor._extract_osnet(frame, bbox))
            if backbone is not None:
                if extractor.backbone_dim is None:
                    extractor.backbone_dim = len(backbone)
                Xs.append(backbone); ys.append(label); cams.append(cam)
            pbar.update(1)
        cap.release()
    pbar.close()
    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(ys, dtype=np.int64),
            np.asarray(cams))


# ------------------------ Model ------------------------

def _build_model(in_dim, n_classes, embed_dim=EMBED_DIM, proj_hidden=PROJ_HIDDEN):
    import torch
    import torch.nn as nn

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(proj_hidden, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
        def forward(self, x):
            return self.net(x)

    class ArcFaceLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.randn(n_classes, embed_dim))
            nn.init.xavier_uniform_(self.W)
        def forward(self, emb, labels=None, margin=0.4, scale=30.0):
            import torch.nn.functional as F
            emb_n = F.normalize(emb, dim=1)
            W_n = F.normalize(self.W, dim=1)
            cos = emb_n @ W_n.t()
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            if labels is None:
                return emb_n, cos * scale
            theta = torch.acos(cos)
            onehot = F.one_hot(labels, num_classes=n_classes).float()
            cos_m = torch.cos(theta + margin * onehot)
            return emb_n, cos_m * scale

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = ProjectionHead()
            self.arc = ArcFaceLayer()
        def forward(self, x, labels=None):
            emb = self.head(x)
            return self.arc(emb, labels)

    return FullModel()


# ------------------------ Training ------------------------

def _stratified_split(y, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for lab in sorted(set(y.tolist())):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - val_frac)))
        train_idx.extend(idx[:cut]); val_idx.extend(idx[cut:])
    return np.asarray(train_idx), np.asarray(val_idx)


def _prototypes(X_emb, y):
    by = defaultdict(list)
    for emb, lab in zip(X_emb, y):
        n = np.linalg.norm(emb) + 1e-8
        by[int(lab)].append(emb / n)
    out = {}
    for lab, embs in by.items():
        mean = np.mean(np.stack(embs), axis=0)
        n = np.linalg.norm(mean) + 1e-8
        out[lab] = mean / n
    return out


def _eval_proto(X_emb, y, prototypes):
    labels = sorted(prototypes)
    P = np.stack([prototypes[l] for l in labels], axis=0)
    norms = np.linalg.norm(X_emb, axis=1, keepdims=True) + 1e-8
    embs = X_emb / norms
    sims = embs @ P.T
    pred_idx = np.argmax(sims, axis=1)
    pred = np.asarray([labels[i] for i in pred_idx])
    return float((pred == y).mean()), pred, sims


def train(epochs=80, batch=128, lr=1e-3, wd=1e-4, margin=0.4, scale=30.0,
          val_frac=0.2, seed=42, no_cache=False):
    import torch
    import torch.nn.functional as F

    # ----- 1. Load + extract (or restore from cache) -----
    if CACHE.exists() and not no_cache:
        logging.info(f"Cache trouvé : {CACHE.name} — chargement (rapide).")
        d = np.load(CACHE, allow_pickle=True)
        X, y, cams = d['X'], d['y'], d['cams']
        backbone_dim = int(d['backbone_dim'])
    else:
        validateds = sorted(TRACKING.glob("*_validated.json"))
        if not validateds:
            logging.error(f"Aucun _validated.json dans {TRACKING}")
            return
        logging.info(f"Cams validées : "
                     f"{[v.stem.replace('_validated','') for v in validateds]}")
        all_rows = []
        for vj in validateds:
            all_rows.extend(_load_examples(vj))
        logging.info(f"{len(all_rows)} crops à extraire au total")
        extractor = FeatureExtractor(backbone='dinov2', dinov2_size='base',
                                     device='cuda:0', half=False)
        X, y, cams = _extract_dinov2_features(all_rows, extractor)
        backbone_dim = extractor.backbone_dim
        np.savez(CACHE, X=X, y=y, cams=cams, backbone_dim=backbone_dim)
        logging.info(f"Cache écrit : {CACHE.name}  "
                     f"({X.shape[0]} crops × {backbone_dim}D)")

    # ----- 2. Stratified split -----
    labels = sorted(set(y.tolist()))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.asarray([label_to_idx[int(l)] for l in y], dtype=np.int64)
    train_idx, val_idx = _stratified_split(y_idx, val_frac=val_frac, seed=seed)
    logging.info(f"Train {len(train_idx)}  Val {len(val_idx)}  "
                 f"({len(labels)} classes : {[format_id(l) for l in labels]})")

    Xtr = torch.from_numpy(X[train_idx]).float()
    ytr = torch.from_numpy(y_idx[train_idx]).long()
    Xva = torch.from_numpy(X[val_idx]).float()
    yva = y_idx[val_idx]
    yva_real = y[val_idx]   # for prototype eval

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    Xtr = Xtr.to(device); ytr = ytr.to(device)
    Xva = Xva.to(device)

    # ----- 3. Train -----
    model = _build_model(backbone_dim, len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Xtr)
    best_val = -1.0
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        loss_acc, n_batches = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            if len(idx) < 2: continue
            xb, yb = Xtr[idx], ytr[idx]
            _emb, logits = model(xb, labels=yb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += float(loss.item()); n_batches += 1
        sched.step()

        # ----- val via prototypes built from TRAIN embeddings -----
        model.eval()
        with torch.no_grad():
            emb_tr = model.head(Xtr).cpu().numpy()
            emb_va = model.head(Xva).cpu().numpy()
        protos = _prototypes(emb_tr, y[train_idx])
        val_acc, _pred, _sims = _eval_proto(emb_va, yva_real, protos)
        history.append((ep, loss_acc / max(1, n_batches), val_acc))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        logging.info(f"Epoch {ep:3d}/{epochs}   "
                     f"loss={loss_acc / max(1, n_batches):.4f}   "
                     f"val_acc={val_acc:.3f}   "
                     f"(best={best_val:.3f})")

    # ----- 4. Save head -----
    import torch
    torch.save({
        'state_dict': best_state,
        'backbone': 'dinov2',
        'dinov2_size': 'base',
        'in_dim': backbone_dim,
        'embed_dim': EMBED_DIM,
        'proj_hidden': PROJ_HIDDEN,
        'labels': labels,
        'best_val_acc': best_val,
    }, HEAD_OUT)
    logging.info(f"Tête ArcFace sauvée -> {HEAD_OUT.name}   "
                 f"(meilleur val_acc = {best_val:.3f})")

    # ----- 5. Diagnostic inter-prototypes -----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb_all = model.head(torch.from_numpy(X).float().to(device)).cpu().numpy()
    protos = _prototypes(emb_all, y)
    logging.info("Cosine inter-prototypes (devrait baisser vs LMBN/DINOv2 brut) :")
    keys = sorted(protos)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            sim = float(np.dot(protos[a], protos[b]))
            logging.info(f"  {format_id(a):>6} vs {format_id(b):>6}  "
                         f"cosine = {sim:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--margin', type=float, default=0.4,
                    help="angular margin (0.3-0.5 typique)")
    ap.add_argument('--scale', type=float, default=30.0)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-cache', action='store_true',
                    help="re-extrait les features même si le cache existe")
    args = ap.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd,
          margin=args.margin, scale=args.scale, val_frac=args.val_frac,
          seed=args.seed, no_cache=args.no_cache)


if __name__ == '__main__':
    main()
