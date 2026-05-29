# Pose2Sim · Tracking — re-ID multi-caméra assisté par ML

Pipeline **offline** de raffinement de tracking par caméra, centré sur
le problème de re-ID long-terme : **quand une personne sort du champ et
revient, lui redonner le bon ID** — pas avec un système 100 % automatique
peu fiable, mais avec une **boucle interactive humain-en-tête** assistée
d'un modèle ML qui apprend de tes confirmations et devient bon
spécifiquement pour la session en cours.

---

## 0. Pour la stagiaire — où regarder en priorité

Si tu reprends ce code, lis dans cet ordre :

1. **§1** (vue d'ensemble) — comprends les 8 stages.
2. **§4** (ML technique) — l'architecture re-ID et pourquoi elle est ce
   qu'elle est. Comme tu fais du stats/ML, tu y verras les arbitrages.
3. **§5 et §6** (les deux GUI : founder+newcomer review, puis scrubber
   de validation) — c'est là que se passe l'interaction humaine.
4. **§7** — déroulé complet d'une session (8 cams), ordre des commandes,
   et le fine-tune ArcFace qui est le saut qualitatif principal.
5. **§9** (référence fichier par fichier) en cas de doute sur où vit
   telle fonction.

Le contexte clinique : **Demo_Seance** = 8 caméras Sony synchronisées,
~2760 frames, ~6 patients en blouse identique + 1-4 staff (coach,
physio, soignants). La difficulté ML majeure est que les patients
visuellement très similaires se ressemblent à 90 % en cosine DINOv2
zero-shot — le pivot ArcFace (§4.3) règle ça.

---

## 1. Vue d'ensemble — pipeline en 8 stages

```
Pose2Sim poseEstimation (existant)  ─►  pose/<cam>_json/   (détections 2D par frame, RTMPose)
                                        videos/<cam>.mp4    (vidéo réelle sync'd)
                │
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  botsort_pipeline.py   (orchestrateur, 1 commande par caméra)         │
│                                                                        │
│  1. botsort_poc           BoT-SORT (boxmot) → tracks bruts            │
│  2. botsort_split_jumps   coupe les sauts de bbox physiquement impossibles │
│  3. botsort_split_appearance coupe les swaps d'ID silencieux (couleur)  │
│  4. botsort_swap_fixer    corrige les inversions d'ID aux frames noires │
│  5. botsort_merger        score appearance + edge + temporel  →  ranking │
│  6. merger_review         GUI 1 : founder mapping + newcomer review (manuel/auto) │
│  7. validate_review       GUI 2 : SCRUBBER frame-précis (vert/rouge/gris) │
│  8. train_from_validated  ENTRAÎNEMENT prototype matching sur séquence validée │
└────────────────────────────────────────────────────────────────────────┘
                │
                ▼
        tracking/<cam>_final.mp4        (vidéo annotée après stage 6)
        tracking/<cam>_final.json       (mapping raw_id → final_label, official_ids)
        tracking/<cam>_validated.json   (per-frame labels après stage 7, source du training)
        tracking/trial_classifier.pkl   (modèle accumulant les cams, src d'inférence stages 6 suivants)
        tracking/arcface_head.pt        (tête de projection ArcFace, produite hors pipeline)
```

**Règle d'or invariable :** training (stage 8) **toujours après**
validation (stage 7). Le classifier n'apprend que sur des labels
frame-par-frame approuvés par l'humain. Sans cette règle, un swap de
tracking non détecté pollue les prototypes et amplifie ses erreurs
sur la suite.

**Idée directrice du tracking lui-même :** BoT-SORT track bien tant
qu'une personne est visible (on garde un `track_buffer` court pour
qu'il ne tente PAS de re-ID hasardeux lui-même). Toute disparition
crée un fragment. Stages 2-5 nettoient, l'humain tranche au stage 6,
le stage 7 met la dernière main, et le classifier apprend des décisions
humaines pour assister les cams suivantes.

---

## 2. Installation et environnement

```bash
conda activate Pose2Sim_new
pip install boxmot tqdm
```

**ATTENTION env :** sur Windows ce projet **doit** être lancé avec le
Python du conda env `Pose2Sim_new`, pas le `python` du shell par défaut.
Si tu utilises le shell par défaut tu tomberas sur la copie pip
installée de Pose2Sim (potentiellement périmée) et tes modifs ne
prendront pas. Une commande type :

```powershell
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" Pose2Sim\Tracking\botsort_pipeline.py [...]
```

(le `&` est le call operator PowerShell, nécessaire quand le premier
token est une string quotée.)

**Poids ReID :**
- **DINOv2** (`--backbone dinov2 --dinov2_size base`, défaut
  **recommandé**) : se télécharge automatiquement via `torch.hub`
  (facebookresearch/dinov2). Base = 768-D, le bon compromis.
- **OSNet** (`--backbone osnet`) : legacy ; à éviter sauf nostalgie. Les
  features Market-1501 ne séparent pas les patients cliniques.

Le conflit OpenMP Windows (`libiomp5md.dll`) est géré
(`KMP_DUPLICATE_LIB_OK=TRUE` posé en tête de chaque module).

---

## 3. Structure des sorties d'une session

```
<trial>/tracking/
├── trial_classifier.pkl              # modèle accumulant les cams (cosine prototypes)
├── arcface_head.pt                   # tête de projection 768→256 (fine-tuned)
├── _arcface_features_cache.npz       # cache des features DINOv2 (régénérable)
├── <cam>_final.mp4                   # vidéo annotée stage 6
├── <cam>_final.json                  # mapping + official_ids + log
├── <cam>_validated.json              # per-frame labels stage 7 (vérité-terrain)
└── intermediates/<cam>/              # fichiers intermédiaires (debug, supprimables)
    ├── botsort.mp4
    ├── tracks.json
    ├── tracks_split.json
    ├── tracks_appsplit.json
    ├── tracks_fixed.json
    ├── merged.mp4
    └── merged.json
```

`<cam>_validated.json` est **la** source de vérité : c'est ce que le
training stage 8 consomme et c'est ce qu'on enverrait à un partenaire
si on devait redistribuer les labels.

---

## 4. Le côté ML — l'architecture re-ID

### 4.1 Pourquoi un MLP softmax ne suffit pas

L'approche naïve (et celle implémentée initialement) : extraire un
embedding par crop (DINOv2 768-D + histogramme couleur torse 30-D +
forme bbox 4-D), entraîner un MLP cross-entropy sur les ~150 crops/identité
des founders. Résultat : softmax saturé à 0/1 sur les exemples
d'entraînement, et **prédictions confidemment fausses** sur une nouvelle
caméra. Pourquoi :

- Le MLP a 256·128 paramètres → capacité largement supérieure au
  nombre d'exemples → mémorisation totale.
- Les features DINOv2 zero-shot pour 5 patients en blouse identique
  donnent des **cosines inter-classes 0.84-0.94** : le signal pour
  discriminer est dans 5-15 % de l'espace, et le MLP fitte ça
  parfaitement mais sans généraliser cross-cam.

LOCO test (leave-one-camera-out) : **54 % accuracy moyenne**. Pour les
besoins d'auto-mode il faut > 90 %.

### 4.2 Prototype matching (étape intermédiaire)

Premier vrai fix : remplacer le MLP par un **prototype par identité** =
moyenne L2-normalisée des embeddings backbone L2-normalisés. La
prédiction = argmax cosine sim entre nouveau crop et chaque prototype.
C'est la méthode canonique de la littérature ReID (cf. *FastReID*,
*TransReID*). Avantages :

- Score continu naturellement borné dans [-1, 1].
- Pas de capacité de mémorisation (juste une moyenne).
- Inférence triviale (1 produit scalaire par classe).

Mais : les prototypes inter-classes restaient à **0.84-0.94 cosine**
dans l'espace DINOv2 zero-shot. La structure géométrique du backbone
n'a tout simplement pas la résolution pour séparer ces patients. Donc
prototype-matching seul = même limitation que le MLP.

### 4.3 Le pivot — fine-tune ArcFace de la tête de projection

L'ingrédient manquant : apprendre **explicitement** au modèle à
discriminer les identités de la session. Schéma :

```
crop ─► DINOv2 base (frozen, 768-D)
           │
           ▼
        Projection head (trainable, ~660K params)
        Linear 768 → 512  + BN + ReLU + Dropout 0.3
        Linear 512 → 256  + BN
           │
           ▼ L2-norm
        embedding 256-D  (vit sur la sphère unité)
           │
           ▼ ArcFace cosine softmax
        logits = scale × cos(θ + margin × y_onehot)
        loss   = cross_entropy(logits, y)
```

**Pourquoi ArcFace (vs softmax classique) :**
- Force les embeddings d'une même classe à se regrouper sur une calotte
  sphérique étroite, et **ajoute une marge angulaire** sur la vraie
  classe (cos(θ + m) au lieu de cos(θ)). Cela "pénalise" le modèle
  pour les prédictions confidemment proches de l'autre côté de la
  frontière → marge accrue à l'inférence.
- Standard du re-ID moderne. Plus stable que triplet (pas besoin de
  mining), plus discriminant que softmax classique (margin angulaire).
- Hyperparamètres `margin=0.4`, `scale=30` sont des valeurs canoniques
  qui marchent généralement.

**Pourquoi DINOv2 reste gelé :**
- ~21M paramètres DINOv2 vs ~660K paramètres de la tête → fine-tuner
  l'ensemble surfitterait gravement sur 4200 crops.
- DINOv2 est pretraité self-supervised sur 142M d'images naturelles ;
  ses features intermédiaires sont déjà très informatives. La tête
  réoriente l'embedding pour la tâche spécifique, ce qui suffit.
- Training de la tête : ~30 sec sur GPU (les features sont
  pré-extraites et cachées, on n'itère que sur des tenseurs).

**Résultats sur Demo_Seance** (9 identités) :

| Métrique | v1 (4 cams) | v2 (8 cams) |
|---|---|---|
| Val accuracy (random 80/20) | 0.985 | 0.984 |
| **LOCO moyen** (vrai test cross-cam) | **0.942** | **0.997** |
| LOCO min sur un fold | 0.776 | 0.994 |
| Cosines inter-prototypes | -0.4 à +0.16 | -0.4 à +0.3 |
| Marge cosine correct vs incorrect | 0.92 vs 0.5-0.7 | 0.94 vs 0-0.85 |

À 8 cams le LOCO atteint 99.7 % en moyenne sur ~8400 samples cumulés
(~30 erreurs totales, 0.3 % de taux d'erreur). Les erreurs résiduelles
sont presque toutes des confusions entre patients en blouse identique
(P1↔P3, P2↔P5) — limite physiologique du re-ID visuel sur cette
population. Un seuil auto à 0.70 absorbe tous ces cas sans faux
positif notable.

### 4.4 Le classifier au runtime

À l'inférence (founder mapping, newcomer review, eval) :

1. `FeatureExtractor` charge DINOv2 base + (optionnel) la tête ArcFace.
2. Pour chaque crop : DINOv2 forward → projection head → L2-norm → 256-D.
3. `FounderClassifier` maintient `prototypes[label]` (mean L2-norm).
4. `predict()` retourne `[(label, cosine_sim), ...]` trié, scores
   continus dans [0, 1] après clamp des négatifs.

Le `trial_classifier.pkl` sauvegarde `X, y, head_path, backbone, ...` :
une fois trained, le pkl est self-contained — recharger suffit.

### 4.5 Évaluation (`_evaluate_classifier.py`)

Trois modes :

| Mode | Train | Test | Mesure |
|---|---|---|---|
| `random` | 80 % aléatoire | 20 % restants | Apprentissage in-distribution |
| `leave_cam` | 3 cams | la 4ème jamais vue | **Vraie généralisation cross-cam** |
| `session` | N-1 cams | 1 cam choisie | Fold unique, plus rapide |

Métriques reportées : top-1 accuracy, recall par identité, top
confusions (true → pred), score cosine top-1 moyen sur corrects vs
incorrects (= calibration).

```bash
python Pose2Sim/Tracking/_evaluate_classifier.py --mode leave_cam \
       --head "<trial>/tracking/arcface_head.pt"
```

---

## 5. GUI 1 — Stage 6 : founder + newcomer review (`merger_review`)

### 5.1 Phase founder mapping (manuelle à froid, assistée ensuite)

Au début de chaque cam, pour chaque track candidat founder (= visible
dans la fenêtre `[founding_start, founding_start + 60]`) :

| Touche | Action |
|---|---|
| `1`..`9` | Confirme : ce track = candidat n°X (proposé par le classifier) |
| `P` | Nouveau **patient** (jamais vu) → P1, P2, ... |
| `T` | Nouveau **staff** (coach, physio...) → S1, S2, ... |
| `S` ou `0` | Ignorer ce track (passant bref, faux positif) |
| `Q` | Quitter la phase (les founders restants gardent leur id local) |

À froid (1ère cam, pas de classifier) le panneau de droite est vide,
tu ne peux que P/T/S. Aux cams suivantes le classifier propose les
top-K candidats (avec leurs vignettes représentatives) — généralement
1-clic.

**Auto-mode** : `--auto_founder 0.80` saute le popup quand la top-1
prédiction a cosine ≥ 0.80 ET marge ≥ 0.15 avec top-2. Fallback
manuel pour les cas ambigus.

### 5.2 Phase newcomer review

Pour chaque track qui apparaît après la fenêtre founder (= une
ré-entrée d'un founder, OU une vraie nouvelle personne) :

| Touche | Action |
|---|---|
| `1`..`6` | Ce newcomer = ce candidat (proposé par appearance + classifier) |
| `P` | Nouveau patient (cas rare : personne entrée tard) |
| `T` | Nouveau staff |
| `S`, `0`, `N` | Garder séparé (ne pas merger) |
| `Q` | Quitter |

`--auto_newcomers 0.80` : même logique que auto_founder pour cette
phase.

### 5.3 Sortie

`<cam>_final.json` contient :
- `id_to_final` : mapping **raw_track_id → final_label**. Le re-keying
  par id brut est critique (un bug antérieur le keyait par id
  in-memory post-remap → tout cassait).
- `official_ids` : la liste des identités établies au founder
  (P1..Pn, S1..Sm). Sert au stage 7 pour ne proposer QUE ces
  identités comme cible de fusion.
- `duplicates_detected` : log des cas "même id sur plusieurs bbox
  d'une même frame" — généralement bénin (overlap detector) mais
  parfois signal d'une collision id à corriger.

---

## 6. GUI 2 — Stage 7 : scrubber de validation (`validate_review`)

C'est **la** vérité-terrain. Un humain valide frame par frame que
chaque identité contient bien la bonne personne et rien que la bonne
personne.

### 6.1 Workflow par identité

- Pré-extraction séquentielle de tous les crops bbox en JPEG RAM
  (~20 KB/crop, ~150 MB max pour un track de 2000 frames). Scrubbing
  instantané, pas de seek cv2 par frame.
- Affichage **frame par frame** d'un grand crop (avec bordure colorée
  selon l'état).
- Si plusieurs détections d'une même identité existent sur la même
  frame (overlap du détecteur OU collision d'id du merger), elles
  s'affichent **côte-à-côte** comme slots séparés ; Tab cycle le focus
  entre slots.
- Une **timeline tri-state en bas** montre d'un coup d'œil quelles
  frames sont vert/rouge/gris/mixte.

### 6.2 Touches essentielles

| Touche | Action |
|---|---|
| `←` / `→` | ±1 frame (skip les duplicates intra-frame) |
| `Tab` | Slot suivant (quand 2+ dets sur la même frame) |
| `X` | Toggle current vert ↔ rouge (= "pas cette personne") |
| `I` | Toggle current vert ↔ gris (= "indéterminé, exclu du training") |
| `V` | Valider l'identité (les gris partent au sentinelle exclu) |
| `R` | Ouvre le picker : tous les rouges → identité choisie |
| `S` | Identité entière indéterminée |
| `Q` | Quitter |

### 6.3 Raccourcis avancés (longues plages d'impostor)

| Touche | Action |
|---|---|
| `PgUp` / `PgDn` ou `↑` / `↓` | ±20 frames |
| `Home` / `End` | Première / dernière frame |
| `B` | Placer une ancre sur la frame courante |
| `E` | Peindre **[ancre..ici]** en rouge sur le slot focus |
| `U` | Annuler **[ancre..ici]** (re-vert) |
| `A` / `C` | Tout rouge / tout vert |

Le slot focus est **préservé** à travers la navigation (clampé au max
slot count de la frame courante) — donc tu peux Tab sur le slot 1, B
au début d'un run d'impostor, naviguer 200 frames en avant, E, et tout
le slot-1 de la plage est marqué rouge. Le slot 0 reste intact.

### 6.4 Mécanique R bbox-précise

Quand tu fais R, on relabelise **exactement les bboxes rouges**
(match par `id(bbox)` Python d'abord, valeur en fallback). Donc 2
détections d'une même id sur une même frame peuvent partir
indépendamment : une vers S1, l'autre reste P1. Différent d'un relabel
par frame qui déplacerait les deux.

Le picker (touche R) ne propose **QUE** les `official_ids` issus du
stage 6 + les P/T créés en review (jamais les fragments). C'est ce
qui garantit que tu ne disperses pas une identité dans des fragments.

### 6.5 Sentinelle indéterminé

Les dets marquées gris sont relabelisées à
`INDETERMINATE_LABEL = 8_888_888`, et cet id est automatiquement
ajouté à `excluded_from_training` du `_validated.json`. Le training
stage 8 voit ce label dans la liste exclude et passe son chemin.

---

## 7. Workflow complet d'une session

Pour 8 caméras, voici l'ordre et les commandes (les `<...>` à
remplacer).

### Étape 0 — préparation
- Tu as déjà `<trial>/pose/<cam>_json/` (Pose2Sim poseEstimation) et
  `<trial>/videos/<cam>.mp4` synchronisés pour les 8 cams.
- Identifie un `--founding_start` par caméra : la frame où le maximum
  de personnes est simultanément visible (ouvre la vidéo dans VLC,
  scrubble, repère).

### Étape 1 — première caméra à froid (manuel total)

```powershell
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\botsort_pipeline.py `
  -j "<trial>\pose\<cam1>_json" `
  -i "<trial>\videos\<cam1>.mp4" `
  -n 8 --backbone dinov2 --dinov2_size base `
  --founding_start <frameN>
```

- Stages 1-5 automatiques (~quelques minutes).
- Stage 6 : tu fais le founder mapping à la main (P/T pour chaque
  personne d'intérêt, S pour les passants). Puis newcomer review pour
  les ré-entrées.
- Stage 7 : tu vérifies chaque identité au scrubber et corriges (R
  les rouges, I les ambigus).
- Stage 8 : entraînement prototype matching incrémental sur le
  `<cam1>_validated.json`. `trial_classifier.pkl` créé.

### Étapes 2 et 3 — caméras 2 et 3 (assistées par le classifier zero-shot)

Mêmes commandes, `--founding_start` adapté. Le classifier de la cam 1
suggère des top-1 à confirmer en 1 clic, mais avec DINOv2 zero-shot la
discrimination patient/patient reste faible (~55 % LOCO) donc beaucoup
de cas ambigus à trancher manuellement.

### Étape clé — fine-tune ArcFace une fois qu'on a 3-4 cams validées

```powershell
# 1. Fine-tune la tête de projection (~5 min total)
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\_fine_tune_arcface.py --epochs 80

# 2. Sanity check par LOCO (vise > 0.85 average)
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\_evaluate_classifier.py --mode leave_cam `
  --head "<trial>\tracking\arcface_head.pt"

# 3. Si LOCO ≥ 0.85 : reconstruit le pkl en 256-D ArcFace
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\_rebuild_classifier_dinov2.py `
  --head "<trial>\tracking\arcface_head.pt"
```

### Étapes 5-8 — caméras restantes en quasi-auto

```powershell
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\botsort_pipeline.py `
  -j "<trial>\pose\<camN>_json" -i "<trial>\videos\<camN>.mp4" `
  -n 8 --backbone dinov2 --dinov2_size base `
  --head "<trial>\tracking\arcface_head.pt" `
  --founding_start <frameN> `
  --auto_founder 0.80 --auto_newcomers 0.80
```

- `--auto_founder 0.80` : top-1 founder accepté auto si cosine ≥ 0.80
  ET marge ≥ 0.15.
- `--auto_newcomers 0.80` : pareil pour les ré-entrées.
- Le scrubber reste actif pour confirmation rapide. Si tout est vert
  dès la première passe (typique vers cam 6-7), tu peux ajouter
  `--skip_validate` pour rendre le run 100 % unattended.

### Si une cam introduit un nouveau staff inconnu

Le scrubber le verra comme une identité dont les top candidats ne
sont jamais > 0.6. Tu presses **R**, dans le picker tu fais **T**
(nouveau staff) → il est ajouté à `official_ids` et trainé après
validation. Idéalement, refais le fine-tune ArcFace (étape clé)
après cette cam pour intégrer le nouveau staff dans la tête.

---

## 8. Référence des flags principaux

| Flag | Défaut | Rôle |
|---|---|---|
| `-j` / `--json_folder` | (requis) | Dossier des JSON de pose Pose2Sim |
| `-i` / `--input` | (requis) | Vidéo de la cam (la vraie, pas `_pose.mp4`) |
| `-n` / `--n_target` | 8 | Nombre attendu de personnes distinctes |
| `--founding_start` | 0 | Frame de début de la fenêtre founder |
| `--founding_window` | 60 | Largeur (frames) de cette fenêtre (= 2 s @ 30 fps) |
| `--backbone` | osnet | `osnet` ou `dinov2` (RECOMMANDÉ `dinov2`) |
| `--dinov2_size` | small | `small` / `base` / `large`. **`base` est le bon défaut.** |
| `--head` | none | Chemin de `arcface_head.pt` (fine-tuned) — saut qualitatif |
| `--auto_founder` | off | Seuil cosine pour auto-accepter top-1 founder |
| `--auto_newcomers` | off | Idem pour ré-entrées |
| `--auto_margin` | 0.15 | Marge top-1 / top-2 minimale pour l'auto-pick |
| `--skip_validate` | off | Saute stages 7-8 (validation + training) |
| `--resume_validate` | off | Saute 1-6, ne fait que validation + training |
| `--classifier_state` | auto | Chemin du `.pkl` (auto-détecté si absent) |
| `--no_classifier` | off | Désactive le ML (ranking histogramme seul, debug) |
| `--no_retrain` | off | Utilise le classifier sans le modifier |
| `--track_buffer` | 30 | Mémoire BoT-SORT (frames). Court = pas de re-ID auto |
| `--skip_poc` / `--skip_merger` / ... | off | Réutilise les intermédiaires |
| `--max_frames` | none | Cap de frames (smoke test) |

---

## 9. Référence fichier par fichier

| Fichier | Rôle | Lignes critiques |
|---|---|---|
| `botsort_pipeline.py` | Orchestrateur, point d'entrée principal | `run_pipeline()` (~ligne 95) |
| `botsort_poc.py` | Stage 1 : BoT-SORT, dump `tracks.json` + MP4 | |
| `botsort_split_jumps.py` | Stage 2 : split aux sauts physiques | |
| `botsort_split_appearance.py` | Stage 3 : split aux discontinuités couleur | |
| `botsort_swap_fixer.py` | Stage 4 : correction des swaps aux frames noires | |
| `botsort_merger.py` | Stage 5 : scoring + ranking des candidats ; constantes `STAFF_ID_OFFSET`, `format_id` | |
| `merger_review.py` | Stage 6 GUI : founder + newcomer review + cross-cam + auto modes | `review_func()` (~l. 422), `render_founder_mapping_figure` (~l. 235), persistance `raw_to_final` à l'écriture (~l. 1112) |
| `validate_review.py` | Stage 7 GUI : scrubber tri-state | `scrub_identity()` (~l. 240), `precrop_identity()` (~l. 165), `_relabel_dets_precise()` (~l. 110) |
| `botsort_classifier.py` | `FeatureExtractor` (DINOv2 / OSNet + tête ArcFace) + `FounderClassifier` (prototypes cosine, persistence) | `_compute_prototypes()`, `predict()`, `_ensure_head_loaded()` |
| `botsort_train_from_final.py` | Stage 8 : `train_from_validated` (séquence validée complète, accumule cam après cam) | `train_from_validated()` (~l. 218) |
| `reid_debug.py` | Diagnostic : pouvoir discriminant d'un backbone sur un dataset | |
| `_fine_tune_arcface.py` | Fine-tune de la tête ArcFace (768→256) avec margin angulaire | `train()` (~l. 250), `_build_model()` (~l. 95) |
| `_evaluate_classifier.py` | Eval : random / LOCO / session, recall + confusion + cosine margin | |
| `_rebuild_classifier_dinov2.py` | Reconstruction du `trial_classifier.pkl` depuis tous les `_validated.json` | |

Tous les fichiers principaux sont lançables seul (`python <fichier>.py --help`)
pour debug, ou via le pipeline.

---

## 10. Limites connues et pistes futures

- **Identités quasi indistinguables** (ex. 3 patients en blouse noire
  identique, vus de loin). Après ArcFace fine-tune sur 4 cams le LOCO
  est ~99 % sur les identités vues, mais une **nouvelle session** avec
  des patients jamais vus partira nécessairement à froid. Solution :
  augmenter les crops (augmentation : flip horizontal, color jitter,
  small crops) AVANT le fine-tune ArcFace pour gagner en robustesse.

- **S3, S4 (staff rares apparus tard) sur 1 seule cam** : impossible à
  évaluer en LOCO (pas de train pour ces fold-là), et possiblement
  fragile si on les croise sur une cam jamais vue. Workaround : forcer
  l'utilisateur à les introduire au stage 6 via T quand le classifier
  donne une cosine < 0.65 sur toutes les options.

- **Multi-cam orchestrator** : actuellement on lance le pipeline 1 cam
  à la fois. Un `botsort_trial.py` qui enchaîne les 8 cams en passant
  le classifier de l'une à l'autre, avec un seul founding_start
  partagé, serait un confort. **TODO open.**

- **Fusion CoMotion** : à terme remplacer stages 1-4 (BoT-SORT +
  splitters + swap_fixer) par les tracks CoMotion (le tracker online de
  la stagiaire). Le merger + GUIs + classifier sont agnostiques au
  tracker amont — ils ne dépendent que de `tracks_fixed.json` (frames,
  bboxes, ids locaux).

- **Cross-trial transfer** : pour le moment chaque session repart de
  zéro (classifier reset, prototypes refaits). Une piste : conserver
  les têtes ArcFace de sessions passées et fine-tuner par-dessus pour
  des sessions futures avec les mêmes patients (suivi longitudinal de
  rééducation, p.ex.).

- **Active learning par incertitude** : actuellement la GUI montre
  toutes les ré-entrées. On pourrait n'afficher au scrubber que les
  identités où la marge cosine top-1 / top-2 est < 0.3 (= incertaines).
  Gain de temps massif quand le classifier est mûr.

---

## 11. Reproduire les résultats Demo_Seance

Pour vérifier que le code fonctionne sur Demo_Seance (8 cams,
~6 patients + 4 staff, 2760 frames chacune) après reprise :

```powershell
# 1. État de référence : 4 cams validées
ls "<trial>\tracking\*_validated.json"
# 26461602, 29813646, 25024839, 23767062 → 4 fichiers attendus

# 2. Fine-tune la tête (régénérable, ~5 min)
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\_fine_tune_arcface.py --epochs 80
# Doit donner val_acc final ~0.98

# 3. LOCO eval
& "C:\ProgramData\anaconda3\envs\Pose2Sim_new\python.exe" `
  Pose2Sim\Tracking\_evaluate_classifier.py --mode leave_cam `
  --head "<trial>\tracking\arcface_head.pt"
# Avec 8 cams validées : LOCO moyenne ~0.997 (min ~0.99)
# Avec 4 cams : LOCO moyenne ~0.94 (min 0.77 — fold dégénéré car S3/S4
# n'existent que dans la cam hold-out)
```

Les 4 cams restantes (22516499, 23859316, 23880904, 24710321) sont à
traiter avec la commande quasi-auto de §7.
