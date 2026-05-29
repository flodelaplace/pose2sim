#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ####################################################################
    ## Active-learning person classifier for the BoT-SORT merger UI   ##
    ####################################################################

    Feature extractor + small MLP classifier that learns who each patient
    is from the user's manual confirmations in the review UI.

    Training data hygiene:
      - Only "fresh" crops are added to the training set (founding window
        for founders, first N frames of confirmed newcomers). Later
        frames of any track are NOT used because they may be polluted by
        silent mid-track ID swaps that the upstream splitters did not
        catch. This is exactly the user's constraint: garbage in,
        garbage out.

    Features per crop (~1500-dim, concatenated):
      - OSNet/LMBN ReID embedding (semantic body features)
      - Torso HSV histogram (clothing colour)
      - Shape features (aspect ratio + area fraction)

    Classifier: sklearn MLP with two hidden layers, early stopping. Fast
    enough to retrain after each manual confirmation (~1-2 s on CPU).

    Persistence: pickle blob saved next to the final.json so it can be
    loaded for the next camera of the same trial (same physical
    patients) or for a follow-up session months later.
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import logging
import pickle
from pathlib import Path

import numpy as np
import cv2


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


## CONSTANTS
MAX_CROPS_PER_LABEL = 30           # cap to keep gallery balanced
MLP_HIDDEN_LAYERS = (256, 128)
MLP_MAX_ITER = 400
MLP_ALPHA = 1e-3
SHAPE_DIM = 4                       # [aspect, area_frac, cx_norm, cy_norm]
THUMB_MAX_DIM = 220                 # max edge of a saved thumbnail (px)


def _crop_quality(bbox, frame_shape):
    '''Score how showable a bbox is (centered, decent size, normal aspect).'''
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    edge_d = min(cx / w, (w - cx) / w, cy / h, (h - cy) / h)
    area_frac = (bw * bh) / (w * h)
    area_term = min(area_frac * 50.0, 1.0)
    aspect = bh / bw
    aspect_term = 1.0 if 1.5 <= aspect <= 4.0 else 0.6
    return edge_d * area_term * aspect_term


def _shrink_crop(crop, max_dim=THUMB_MAX_DIM):
    '''Resize a crop so its longest edge is at most max_dim, for storage.'''
    h, w = crop.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return crop.copy()
    ratio = max_dim / longest
    return cv2.resize(crop, (int(w * ratio), int(h * ratio)))


## FUNCTIONS
class FeatureExtractor:
    '''
    Extract per-crop multimodal features (backbone + torso color histogram
    + bbox shape). The backbone is selectable: 'osnet' (default, fast,
    legacy) or 'dinov2' (richer self-supervised features, better
    generalisation at multi-cam mocap distances).
    '''

    DINOV2_NAME = {
        'small': 'dinov2_vits14',   # 384-dim, fast
        'base':  'dinov2_vitb14',   # 768-dim
        'large': 'dinov2_vitl14',   # 1024-dim, slower
    }

    def __init__(self, backbone='osnet',
                 reid_weights='osnet_ain_x1_0_msmt17.pt',
                 dinov2_size='small',
                 device='cuda:0', half=False,
                 head_path=None):
        if backbone not in ('osnet', 'dinov2'):
            raise ValueError(f"Unknown backbone: {backbone}")
        self.backbone = backbone
        self.reid_weights = reid_weights
        self.dinov2_size = dinov2_size
        self.device = device
        self.half = half if device != 'cpu' else False
        self._reid_backend = None
        self._dinov2_model = None
        self._color_dim = None
        self.backbone_dim = None     # populated on first successful extract
                                     # (used by prototype matching to slice
                                     # the backbone embedding out of the
                                     # concatenated feature vector).
        # Optional ArcFace projection head trained on validated crops.
        # When set, the "backbone feature" returned by extract() becomes
        # the L2-normalised ArcFace embedding (= what the head was trained
        # to make discriminative for THIS trial's identities), not the
        # zero-shot DINOv2 output. backbone_dim then equals embed_dim.
        self.head_path = head_path
        self._head_model = None
        self._head_in_dim = None
        self._head_embed_dim = None

    # ---- ArcFace head loading (lazy) ----
    def _ensure_head_loaded(self):
        if self.head_path is None or self._head_model is not None:
            return
        import torch, torch.nn as nn
        ckpt = torch.load(self.head_path, map_location=self.device,
                          weights_only=False)
        self._head_in_dim = int(ckpt['in_dim'])
        self._head_embed_dim = int(ckpt['embed_dim'])
        proj_hidden = int(ckpt.get('proj_hidden', 512))
        in_dim, embed_dim = self._head_in_dim, self._head_embed_dim

        class Head(nn.Module):
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

        head = Head().to(self.device).eval()
        # The saved state_dict bundles head + ArcFace layer; load only what
        # belongs to the head (= keys starting with "head.").
        sd = ckpt['state_dict']
        head_sd = {k.removeprefix('head.'): v
                   for k, v in sd.items() if k.startswith('head.')}
        head.load_state_dict(head_sd, strict=True)
        if self.half:
            head = head.half()
        self._head_model = head
        import logging
        logging.info(
            f"ArcFace head loaded: {in_dim}D -> {embed_dim}D  "
            f"(best val_acc at train time: "
            f"{ckpt.get('best_val_acc', float('nan')):.3f})")

    # ---- Backbone loading (lazy) ----
    def _ensure_loaded(self):
        if self.backbone == 'osnet':
            if self._reid_backend is None:
                from boxmot.reid.core.reid import ReID
                self._reid_backend = ReID(
                    weights=Path(self.reid_weights),
                    device=self.device,
                    half=self.half,
                ).model
        else:   # dinov2
            if self._dinov2_model is None:
                import torch
                model_name = self.DINOV2_NAME.get(
                    self.dinov2_size, 'dinov2_vits14')
                self._dinov2_model = torch.hub.load(
                    'facebookresearch/dinov2', model_name,
                    trust_repo=True)
                self._dinov2_model = self._dinov2_model.to(self.device).eval()
                if self.half:
                    self._dinov2_model = self._dinov2_model.half()

    # ---- Backbone-specific extractors ----
    def _extract_osnet(self, frame, bbox):
        feats = self._reid_backend.get_features(
            np.asarray([list(bbox)], dtype=np.float32), frame)
        if feats is None or len(feats) == 0:
            return None
        return np.asarray(feats[0], dtype=np.float32).flatten()

    def _extract_dinov2(self, frame, bbox):
        import torch
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # DINOv2 wants square 14n × 14n input; 224 = 14 × 16, a safe default
        crop_resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        if self.half:
            tensor = tensor.half()
        with torch.no_grad():
            features = self._dinov2_model(tensor)
        return features.detach().cpu().float().numpy().flatten()

    def extract(self, frame, bbox):
        '''Returns the concatenated feature vector, or None if invalid.'''
        self._ensure_loaded()
        from Pose2Sim.Tracking.botsort_merger import extract_torso_hist

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5 / w
        cy = (y1 + y2) * 0.5 / h

        # Backbone (OSNet or DINOv2)
        if self.backbone == 'osnet':
            backbone_feat = self._extract_osnet(frame, bbox)
        else:
            backbone_feat = self._extract_dinov2(frame, bbox)
        if backbone_feat is None:
            return None

        # If an ArcFace head was trained on validated crops, project the
        # raw backbone embedding through it and L2-normalise -> the rest of
        # the system sees the LEARNED discriminative embedding instead of
        # the zero-shot DINOv2 one. Prototypes then live in a space where
        # P1 vs P2 is much further apart than 0.05 cosine.
        if self.head_path is not None:
            self._ensure_head_loaded()
            import torch
            with torch.no_grad():
                x = torch.from_numpy(backbone_feat).float().to(self.device)
                if self.half: x = x.half()
                x = x.unsqueeze(0)   # BN needs batch dim
                emb = self._head_model(x).squeeze(0)
                emb = emb / (emb.norm() + 1e-8)
                backbone_feat = emb.detach().cpu().float().numpy()

        if self.backbone_dim is None:
            self.backbone_dim = len(backbone_feat)

        # Torso histogram
        color_feat = extract_torso_hist(frame, bbox)
        if color_feat is None:
            color_feat = np.zeros(self._color_dim or 512, dtype=np.float32)
        if self._color_dim is None:
            self._color_dim = len(color_feat)

        # Shape
        aspect = bh / bw
        area_frac = (bw * bh) / max(1.0, w * h)
        shape_feat = np.asarray([aspect, area_frac, cx, cy], dtype=np.float32)

        return np.concatenate(
            [backbone_feat, color_feat, shape_feat]).astype(np.float32)


class FounderClassifier:
    '''
    Active-learning classifier: MLP on top of multimodal features.

    Add labeled crops with `add_track_examples` (sampled crops from a
    track) or `add_examples_from_frames` (specific (frame_idx, bbox)
    pairs). Call `train` after each batch. `predict` returns a ranked
    list of (label, probability) tuples for a new crop.
    '''

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        # Parallel arrays kept for cheap incremental updates
        self.X = []
        self.y = []
        self.scaler = None
        self.classifier = None
        self.trained = False
        # Per-label L2-normalized mean BACKBONE embedding. predict() uses
        # cosine similarity against these prototypes instead of the MLP
        # softmax, which saturated to 0/1 on 7 labels x ~150 crops and gave
        # confidently-wrong answers on a new camera viewpoint. Prototypes
        # are derived from self.X[:, :backbone_dim] so they cost nothing to
        # recompute and don't need a retrain.
        self.prototypes = {}        # label (int) -> 1D np.array (unit norm)
        # One representative thumbnail per label, used by the cross-cam
        # mapping UI so the user can SEE what each known patient looks
        # like when bootstrapping the next camera.
        self.thumbnails = {}        # label (int) -> BGR ndarray
        self._thumb_quality = {}    # label -> last best quality score

    def _compute_prototypes(self):
        '''Refresh self.prototypes from self.X / self.y.

        Each prototype is the L2-normalized mean of L2-normalized backbone
        embeddings for that label (= cosine-similarity centroid). Drops
        labels with too few samples to be reliable.
        '''
        self.prototypes = {}
        if not self.X:
            return
        backbone_dim = self.feature_extractor.backbone_dim
        if backbone_dim is None or backbone_dim <= 0:
            return
        by_label = {}
        for feat, label in zip(self.X, self.y):
            emb = np.asarray(feat[:backbone_dim], dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n < 1e-8:
                continue
            by_label.setdefault(int(label), []).append(emb / n)
        for label, embs in by_label.items():
            if len(embs) < 3:    # too few to estimate a stable mean
                continue
            mean = np.mean(np.stack(embs, axis=0), axis=0)
            n = float(np.linalg.norm(mean))
            if n < 1e-8:
                continue
            self.prototypes[label] = mean / n

    def n_examples_per_label(self):
        counts = {}
        for lab in self.y:
            counts[lab] = counts.get(lab, 0) + 1
        return counts

    def add_example(self, frame, bbox, label):
        feat = self.feature_extractor.extract(frame, bbox)
        if feat is None:
            return False
        self.X.append(feat)
        self.y.append(int(label))
        # Maintain a representative thumbnail per label
        q = _crop_quality(bbox, frame.shape)
        if q > self._thumb_quality.get(int(label), -1.0):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                self.thumbnails[int(label)] = _shrink_crop(frame[y1:y2, x1:x2])
                self._thumb_quality[int(label)] = q
        return True

    def add_examples_from_frames(self, video_path, frame_bbox_pairs, label,
                                 max_crops=MAX_CROPS_PER_LABEL):
        '''
        Sample up to `max_crops` from `frame_bbox_pairs`, extract features
        for each, and add them with the given label.
        '''
        pairs = list(frame_bbox_pairs)
        if not pairs:
            return 0
        if len(pairs) > max_crops:
            idx = np.linspace(0, len(pairs) - 1, max_crops).astype(int)
            pairs = [pairs[i] for i in idx]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        added = 0
        for f_idx, bbox in pairs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            if self.add_example(frame, bbox, label):
                added += 1
        cap.release()
        return added

    def train(self):
        '''Fit/refit the MLP on the current (X, y). Returns True if trained.'''
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        labels = set(self.y)
        if len(labels) < 2 or len(self.X) < 2 * len(labels):
            self.trained = False
            return False

        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # validation_fraction needs at least 1 sample per class on each
        # split; disable early stopping when data is tiny.
        early_stop = len(self.X) >= 3 * len(labels) * 4
        self.classifier = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            alpha=MLP_ALPHA,
            random_state=42,
            early_stopping=early_stop,
            validation_fraction=0.1 if early_stop else 0.0,
        )
        self.classifier.fit(X_scaled, y)
        self.trained = True
        # Refresh cosine-similarity prototypes from the same data.
        self._compute_prototypes()
        return True

    def predict(self, frame, bbox):
        '''Rank labels by cosine similarity to per-label prototypes.

        Returns [(label, sim_in_[0,1]), ...] descending, or None if the
        classifier isn't ready. Cosine sim is mapped to [0, 1] by clamping
        negative values (= "definitely not this person") to 0 — this is
        what the UI displays as 'confiance'. With DINOv2 backbone, same
        person across cams typically scores 0.6-0.9; different people in
        the same scene score < 0.5.

        Falls back to the MLP softmax if no prototypes are available
        (old pkl, or backbone_dim unknown).
        '''
        feat = self.feature_extractor.extract(frame, bbox)
        if feat is None:
            return None
        backbone_dim = self.feature_extractor.backbone_dim
        if self.prototypes and backbone_dim:
            emb = np.asarray(feat[:backbone_dim], dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n < 1e-8:
                return None
            emb = emb / n
            out = []
            for label, proto in self.prototypes.items():
                sim = float(np.dot(emb, proto))   # cosine, in [-1, 1]
                out.append((int(label), max(0.0, sim)))
            out.sort(key=lambda x: -x[1])
            return out
        # Legacy fallback (kept so old pkls keep working until refresh).
        if not self.trained:
            return None
        X_scaled = self.scaler.transform([feat])
        probs = self.classifier.predict_proba(X_scaled)[0]
        return sorted(zip(self.classifier.classes_.tolist(), probs.tolist()),
                      key=lambda x: -x[1])

    # ---- Persistence ----
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'X': self.X, 'y': self.y,
                'scaler': self.scaler, 'classifier': self.classifier,
                'trained': self.trained,
                'backbone': self.feature_extractor.backbone,
                'reid_weights': self.feature_extractor.reid_weights,
                'dinov2_size': self.feature_extractor.dinov2_size,
                'head_path': str(self.feature_extractor.head_path)
                             if self.feature_extractor.head_path else None,
                'thumbnails': self.thumbnails,
                'thumb_quality': self._thumb_quality,
            }, f)

    @classmethod
    def load(cls, path, feature_extractor=None):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        saved_backbone = state.get('backbone', 'osnet')
        saved_dinov2 = state.get('dinov2_size', 'small')
        saved_reid = state.get('reid_weights', 'osnet_ain_x1_0_msmt17.pt')
        saved_head = state.get('head_path', None)
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(
                backbone=saved_backbone,
                reid_weights=saved_reid,
                dinov2_size=saved_dinov2,
                head_path=saved_head,
            )
        elif saved_head and not feature_extractor.head_path:
            # Caller didn't ask for a head but the pkl was built with one
            # -> inherit (would otherwise produce mismatching feature dims).
            feature_extractor.head_path = saved_head
        elif (feature_extractor.backbone != saved_backbone
              or (saved_backbone == 'dinov2'
                  and feature_extractor.dinov2_size != saved_dinov2)):
            # Force-align with the saved config to keep feature dimensions
            # consistent. The user can delete trial_classifier.pkl if they
            # truly want to switch backbones.
            import logging
            logging.warning(
                f"Classifier was trained with backbone={saved_backbone}"
                + (f" (dinov2_size={saved_dinov2})"
                   if saved_backbone == 'dinov2' else "")
                + f"; forcing the FeatureExtractor to match instead of "
                + f"the requested {feature_extractor.backbone}.")
            feature_extractor.backbone = saved_backbone
            feature_extractor.dinov2_size = saved_dinov2
            feature_extractor.reid_weights = saved_reid
            feature_extractor._reid_backend = None
            feature_extractor._dinov2_model = None
        obj = cls(feature_extractor)
        obj.X = state['X']
        obj.y = state['y']
        obj.scaler = state['scaler']
        obj.classifier = state['classifier']
        obj.trained = state['trained']
        obj.thumbnails = state.get('thumbnails', {})
        obj._thumb_quality = state.get('thumb_quality', {})
        # Old pkls don't have backbone_dim cached on the extractor; infer it
        # from the very first stored feature so prototypes can be computed
        # without doing an extra extract pass.
        if obj.X and feature_extractor.backbone_dim is None:
            # The concat layout is [backbone | color | shape(4)]. Color dim
            # is what extract_torso_hist returns; reconstruct it by reading
            # one row and subtracting the known shape (4) + a probe of the
            # extractor's color_dim (if already cached) or by trusting the
            # backbone size lookup table.
            from collections import Counter as _C
            row_len = len(obj.X[0])
            # backbone size is deterministic per (backbone, size) pair
            BACKBONE_SIZES = {
                ('dinov2', 'small'): 384,
                ('dinov2', 'base'):  768,
                ('dinov2', 'large'): 1024,
            }
            guess = BACKBONE_SIZES.get(
                (feature_extractor.backbone, feature_extractor.dinov2_size))
            if guess is None and feature_extractor.backbone == 'osnet':
                # OSNet ain_x1_0 outputs 512-D, lmbn_n_market 2048-D
                guess = 2048 if 'lmbn' in (feature_extractor.reid_weights or '') \
                    else 512
            if guess is not None and 0 < guess < row_len:
                feature_extractor.backbone_dim = guess
        obj._compute_prototypes()
        return obj
