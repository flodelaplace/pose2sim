#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #######################################################
    ## One-shot pipeline: BoT-SORT + merger + manual UI  ##
    #######################################################

    Runs the three stages of the per-camera tracking refinement on a
    single video in one command:

      1. botsort_poc      -> raw BoT-SORT tracks + annotated MP4 + tracks.json
      2. botsort_merger   -> closed-world merging with torso histogram,
                             spatial proximity and entry/exit edge priors
                             -> merged.json + merged.mp4
      3. merger_review    -> matplotlib popup for borderline pairs
                             -> final.json + final.mp4

    The pipeline expects the pose JSONs to already exist (produced by
    `Pose2Sim.poseEstimation()`); it does not re-run pose estimation.

    Outputs all sit next to each other in <output_dir>, defaulting to the
    parent of the JSON folder:
      <basename>.mp4           # stage 1 (raw)
      <basename>_tracks.json   # stage 1
      <basename>_merged.mp4    # stage 2 (auto-merged)
      <basename>_merged.json   # stage 2
      <basename>_final.mp4     # stage 3 (after your Y/N decisions)
      <basename>_final.json    # stage 3

    Usage:
      botsort_pipeline -j <json_folder> -i <video.mp4>
      botsort_pipeline -j <json_folder> -i <video.mp4> -n 8 --skip_review
      botsort_pipeline -j <json_folder> -i <video.mp4> --auto_thresh 0.50
      from Pose2Sim.Tracking import botsort_pipeline
      botsort_pipeline.run_pipeline(json_folder=r'<path>', input=r'<video>')
'''


## INIT
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import logging
import argparse
from pathlib import Path

from Pose2Sim.Tracking import (
    botsort_poc, botsort_split_jumps, botsort_split_appearance,
    botsort_swap_fixer, botsort_merger, merger_review,
    validate_review, botsort_train_from_final,
)


## AUTHORSHIP INFORMATION
__author__ = "Florian Delaplace"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["Florian Delaplace", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')


def derive_tracking_paths(json_folder, output_dir_override=None):
    '''
    Compute the trial-wide tracking layout:
      <trial>/tracking/
        trial_classifier.pkl
        <cam>_final.mp4 + <cam>_final.json    (top-level for easy access)
        intermediates/<cam>/                  (per-cam pipeline artifacts)
    json_folder is expected to be `<trial>/pose/<cam>_json`, so trial_dir
    is two levels up. If `output_dir_override` is given, it replaces the
    auto-derived `<trial>/tracking/` location.
    '''
    json_folder = Path(json_folder)
    cam_basename = json_folder.name.replace('_json', '')
    pose_dir = json_folder.parent
    trial_dir = pose_dir.parent
    if output_dir_override:
        tracking_dir = Path(output_dir_override)
    else:
        tracking_dir = trial_dir / 'tracking'
    intermediates_dir = tracking_dir / 'intermediates' / cam_basename
    tracking_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir, intermediates_dir, cam_basename


def run_pipeline(json_folder, input, output_dir=None, n_target=8,
                 auto_thresh=1.5, review_thresh=0.20,
                 reid_model='lmbn_n_market.pt',  # unused now (color hist),
                                                 # kept for CLI compat
                 device='cuda:0', half=True,
                 track_buffer=30, cmc_method='ecc',
                 min_track_length=5,
                 skip_review=False, skip_poc=False, skip_merger=False,
                 skip_swap_fixer=False, skip_split_jumps=False,
                 skip_split_appearance=False,
                 dropped_json=None,
                 split_max_speed=40.0, split_min_jump=60.0,
                 appearance_sim_threshold=0.40,
                 classifier_state=None, no_classifier=False,
                 no_retrain=False,
                 auto_newcomers_thresh=None, auto_margin=0.15,
                 auto_founder_thresh=None,
                 backbone='osnet', dinov2_size='small',
                 head_path=None,
                 founding_start=0, founding_window=60,
                 skip_validate=False, resume_validate=False,
                 max_frames=None):
    '''Run the pipeline stages on a single camera. See module docstring.'''
    json_folder = Path(json_folder)
    video_path = Path(input)

    tracking_dir, intermediates_dir, basename = derive_tracking_paths(
        json_folder, output_dir)

    # Intermediate artifacts in <tracking>/intermediates/<cam>/
    raw_mp4 = intermediates_dir / "botsort.mp4"
    tracks_json = intermediates_dir / "tracks.json"
    split_tracks_json = intermediates_dir / "tracks_split.json"
    appsplit_tracks_json = intermediates_dir / "tracks_appsplit.json"
    fixed_tracks_json = intermediates_dir / "tracks_fixed.json"
    merged_mp4 = intermediates_dir / "merged.mp4"
    merged_json = intermediates_dir / "merged.json"

    # Final outputs land directly in <tracking>/ (one per camera)
    final_mp4 = tracking_dir / f"{basename}_final.mp4"
    final_json = tracking_dir / f"{basename}_final.json"
    validated_json = tracking_dir / f"{basename}_validated.json"

    # Trial-wide classifier: auto-detected if no --classifier_state given
    trial_classifier_path = tracking_dir / "trial_classifier.pkl"
    if classifier_state is None and trial_classifier_path.is_file():
        classifier_state = str(trial_classifier_path)
        logging.info(f"Auto-loading trial classifier: {trial_classifier_path}")
    elif classifier_state is None:
        logging.info(f"No trial_classifier.pkl at {tracking_dir} yet "
                     f"(this looks like the first cam of the trial).")

    # ---- Resume mode: skip stages 1-6, re-run only validation + training ----
    # Use when the review (stage 6) is already done and saved to final.json,
    # and you only want to (re)do the frame-precise validation + training.
    if resume_validate:
        if not final_json.exists():
            raise FileNotFoundError(
                f"--resume_validate needs an existing final.json: {final_json}")
        logging.info("=" * 60)
        logging.info(f"RESUME : validation + training on {final_json.name} "
                     f"(stages 1-6 skipped)")
        logging.info("    V=valider | 1..9/0=1er crop faux | S=skip | Q=quit")
        logging.info("=" * 60)
        vres = validate_review.validate_func(
            final_json=str(final_json),
            output_json=str(validated_json),
            device=device,
            reid_model=reid_model,
        )
        if vres.get('validated'):
            logging.info("Full-sequence training on validated data...")
            botsort_train_from_final.train_from_validated(
                validated_json=str(validated_json),
                classifier_pkl=str(trial_classifier_path),
                backbone=backbone,
                dinov2_size=dinov2_size,
                reid_model=reid_model,
                device=device,
                head_path=head_path,
            )
        else:
            logging.info("Validation not completed (quit) -> training skipped.")
        return {'validated_json': str(validated_json),
                'validated': vres.get('validated')}

    # ---- Stage 1: BoT-SORT ----
    if skip_poc and tracks_json.exists():
        logging.info(f"Stage 1 skipped (using existing {tracks_json.name})")
    else:
        logging.info("=" * 60)
        logging.info("Stage 1/3 : BoT-SORT")
        logging.info("=" * 60)
        botsort_poc.botsort_poc_func(
            json_folder=str(json_folder),
            input=str(video_path),
            output=str(raw_mp4),
            tracks_json_path=str(tracks_json),
            device=device,
            half=half,
            track_buffer=track_buffer,
            cmc_method=cmc_method,
            max_frames=max_frames,
        )

    # ---- Stage 2: split jumps (cut tracks at physically impossible bbox jumps) ----
    if skip_split_jumps:
        logging.info("Stage 2/6 skipped (--skip_split_jumps)")
        post_split_tracks = tracks_json
    else:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 2/6 : split impossible bbox jumps")
        logging.info("=" * 60)
        botsort_split_jumps.split_func(
            tracks_json=str(tracks_json),
            output_json=str(split_tracks_json),
            max_speed=split_max_speed,
            min_jump=split_min_jump,
        )
        post_split_tracks = split_tracks_json

    # ---- Stage 3: split appearance (cut tracks at silent ID swaps mid-track) ----
    if skip_split_appearance:
        logging.info("Stage 3/6 skipped (--skip_split_appearance)")
        post_appsplit_tracks = post_split_tracks
    else:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 3/6 : split appearance discontinuities")
        logging.info("=" * 60)
        botsort_split_appearance.split_appearance(
            tracks_json=str(post_split_tracks),
            output_json=str(appsplit_tracks_json),
            sim_threshold=appearance_sim_threshold,
        )
        post_appsplit_tracks = appsplit_tracks_json

    # ---- Stage 4: swap fixer (correct ID inversions at crossings) ----
    if skip_swap_fixer:
        logging.info("Stage 4/6 skipped (--skip_swap_fixer)")
        merger_input_tracks = post_appsplit_tracks
    elif fixed_tracks_json.exists() and skip_poc and skip_split_jumps and skip_split_appearance:
        logging.info(f"Stage 4/6 skipped (using existing {fixed_tracks_json.name})")
        merger_input_tracks = fixed_tracks_json
    else:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 4/6 : swap fixer (ID inversions at crossings)")
        logging.info("=" * 60)
        botsort_swap_fixer.fix_swaps(
            tracks_json=str(post_appsplit_tracks),
            output_json=str(fixed_tracks_json),
            dropped_json=dropped_json,
        )
        merger_input_tracks = fixed_tracks_json

    # ---- Stage 5: post-hoc merger ----
    if skip_merger and merged_json.exists():
        logging.info(f"Stage 5/6 skipped (using existing {merged_json.name})")
    else:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 5/6 : post-hoc merging (3 priors + closed-world)")
        logging.info("=" * 60)
        botsort_merger.merge_func(
            tracks_json=str(merger_input_tracks),
            output_mp4=str(merged_mp4),
            output_json=str(merged_json),
            n_target=n_target,
            auto_thresh=auto_thresh,
            review_thresh=review_thresh,
            min_track_length=min_track_length,
        )

    # ---- Stage 6: manual review of re-entries ----
    if skip_review:
        logging.info("Stage 6/6 skipped (--skip_review).")
        logging.info(f"  Auto-merged result available at: {merged_mp4}")
        return {'raw': str(raw_mp4), 'merged_json': str(merged_json),
                'merged_mp4': str(merged_mp4)}

    do_validate = not skip_validate

    logging.info("")
    logging.info("=" * 60)
    logging.info("Stage 6/8 : manual re-entry review (who is who)")
    logging.info("    [1..N] = match candidate | 0/N = new person | S = skip | Q = quit")
    logging.info("=" * 60)
    # When we are going to validate + train on the FULL validated sequence
    # (stages 7-8), the review must NOT train on this (still unvalidated)
    # cam: training is deferred until after validation. So force the
    # deferred mode (= no_retrain) during review in that case.
    review_no_retrain = no_retrain or do_validate
    save_path = None if review_no_retrain else str(trial_classifier_path)
    merger_review.review_func(
        merged_json=str(merged_json),
        output_json=str(final_json),
        output_mp4=str(final_mp4),
        final_n_target=n_target,
        classifier_state=classifier_state,
        classifier_save_path=save_path,
        use_classifier=not no_classifier,
        no_retrain=review_no_retrain,
        auto_newcomers_thresh=auto_newcomers_thresh,
        auto_founder_thresh=auto_founder_thresh,
        auto_margin=auto_margin,
        backbone=backbone,
        dinov2_size=dinov2_size,
        head_path=head_path,
        founding_window=founding_window,
        founding_start=founding_start,
        reid_model=reid_model,
        device=device,
    )

    # ---- Stage 7: frame-precise validation & correction ----
    validated = False
    if do_validate:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 7/8 : frame-precise validation & correction")
        logging.info("    V=valider | 1..9/0=1er crop faux | S=skip | Q=quit")
        logging.info("=" * 60)
        vres = validate_review.validate_func(
            final_json=str(final_json),
            output_json=str(validated_json),
            device=device,
            reid_model=reid_model,
        )
        validated = bool(vres.get('validated'))

    # ---- Stage 8: full-sequence training on the validated data ----
    if do_validate and validated:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Stage 8/8 : full-sequence training on validated data")
        logging.info("=" * 60)
        botsort_train_from_final.train_from_validated(
            validated_json=str(validated_json),
            classifier_pkl=str(trial_classifier_path),
            backbone=backbone,
            dinov2_size=dinov2_size,
            reid_model=reid_model,
            device=device,
        )
    elif do_validate and not validated:
        logging.info("Validation not completed (quit early) -> training skipped. "
                     "The classifier on disk is unchanged.")

    logging.info("")
    logging.info("=" * 60)
    logging.info("Pipeline done.")
    logging.info(f"  Final MP4       : {final_mp4}")
    if do_validate:
        logging.info(f"  Validated JSON  : {validated_json} (validated={validated})")
    logging.info("=" * 60)

    return {
        'raw_mp4': str(raw_mp4),
        'tracks_json': str(tracks_json),
        'merged_mp4': str(merged_mp4),
        'merged_json': str(merged_json),
        'final_mp4': str(final_mp4),
        'final_json': str(final_json),
        'validated_json': str(validated_json) if do_validate else None,
        'validated': validated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-j', '--json_folder', required=True,
                        help="Pose2Sim per-frame JSONs (one folder per camera).")
    parser.add_argument('-i', '--input', required=True,
                        help="Camera video file (mp4).")
    parser.add_argument('-o', '--output_dir', default=None,
                        help="Where to write outputs (defaults to JSON folder's parent).")
    parser.add_argument('-n', '--n_target', type=int, default=8,
                        help="Expected distinct persons (closed-world cap).")
    parser.add_argument('--auto_thresh', type=float, default=1.5,
                        help="Auto-merge threshold. Default 1.5 = no auto-merge (every re-entry asked manually).")
    parser.add_argument('--review_thresh', type=float, default=0.20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-half', action='store_true')
    parser.add_argument('--track_buffer', type=int, default=30,
                        help="BoT-SORT's lost-track memory (frames). 30=1s only bridges short occlusions, "
                             "longer gaps create new tracks for the merger + manual review.")
    parser.add_argument('--cmc_method', default='ecc')
    parser.add_argument('--min_track_length', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--skip_review', action='store_true',
                        help="Stop after stage 3 (no popup, useful for batch).")
    parser.add_argument('--skip_poc', action='store_true',
                        help="Reuse existing tracks.json if present.")
    parser.add_argument('--skip_merger', action='store_true',
                        help="Reuse existing merged.json if present.")
    parser.add_argument('--skip_swap_fixer', action='store_true',
                        help="Skip the swap fixer stage (don't try to correct ID inversions at crossings).")
    parser.add_argument('--skip_split_jumps', action='store_true',
                        help="Skip the jump splitter stage.")
    parser.add_argument('--skip_split_appearance', action='store_true',
                        help="Skip the appearance-discontinuity splitter stage.")
    parser.add_argument('--split_max_speed', type=float, default=40.0,
                        help="Max px/frame speed; bigger = physically impossible jump.")
    parser.add_argument('--split_min_jump', type=float, default=60.0,
                        help="Don't split if absolute jump in px is below this (= noise).")
    parser.add_argument('--appearance_sim_threshold', type=float, default=0.40,
                        help="Sliding-window histogram similarity below which an "
                             "appearance shift inside a track triggers a split.")
    parser.add_argument('--classifier_state', default=None,
                        help="Path to a .pkl classifier. If omitted, the pipeline "
                             "auto-detects <trial>/tracking/trial_classifier.pkl. "
                             "Saved at the end (unless --no_retrain) for the next cam.")
    parser.add_argument('--no_classifier', action='store_true',
                        help="Disable the active-learning classifier (histogram-only ranking).")
    parser.add_argument('--no_retrain', action='store_true',
                        help="Use the loaded classifier for predictions but DO NOT add "
                             "new crops or retrain it; the trial_classifier.pkl on disk "
                             "is left untouched. Useful once the classifier is solid.")
    parser.add_argument('--auto_founder', type=float, default=None,
                        metavar='THRESH',
                        help="Auto-accept the classifier's top-1 prediction "
                             "at the founder step if cosine sim >= THRESH "
                             "(typical 0.65). Falls back to manual otherwise. "
                             "Use to test full-auto mode on a new cam.")
    parser.add_argument('--auto_newcomers', type=float, default=None,
                        nargs='?', const=0.5,
                        help="Skip newcomer popups: auto-pick top-1 candidate when its "
                             "score >= THRESH, else mark as new person. Default 0.5 "
                             "if flag given without value. Founder phase stays manual.")
    parser.add_argument('--auto_margin', type=float, default=0.15,
                        help="Min gap between top-1 and top-2 classifier scores for an "
                             "auto-pick. Below this the prediction is ambiguous -> NEW.")
    parser.add_argument('--backbone', default='osnet',
                        choices=['osnet', 'dinov2'],
                        help="Feature backbone for the classifier. 'osnet' = fast, "
                             "legacy. 'dinov2' = richer features, better "
                             "generalisation at multi-cam mocap distance. "
                             "WARNING: switching backbones invalidates an existing "
                             "trial_classifier.pkl (delete it to start fresh).")
    parser.add_argument('--dinov2_size', default='small',
                        choices=['small', 'base', 'large'],
                        help="When --backbone dinov2: ViT size. small=384-dim, "
                             "base=768, large=1024. Default small.")
    parser.add_argument('--head', default=None,
                        metavar='PATH',
                        help="Optional ArcFace projection head .pt produced "
                             "by _fine_tune_arcface.py. When set, the "
                             "backbone output is projected through the head "
                             "(L2-normalised 256-D embedding) before "
                             "prototype matching, giving genuinely "
                             "discriminative re-ID for the trial's "
                             "specific identities.")
    parser.add_argument('--skip_validate', action='store_true',
                        help="Skip stage 7-8 (frame-precise validation + full-sequence "
                             "training). Falls back to incremental training during review.")
    parser.add_argument('--resume_validate', action='store_true',
                        help="Skip stages 1-6 and (re)run ONLY validation + training on "
                             "the already-existing final.json. Use to resume/redo the "
                             "validation step without redoing tracking + review.")
    parser.add_argument('--founding_start', type=int, default=0,
                        help="Frame index where the 'founders' window starts. Default 0 "
                             "(= beginning of video). Set this if everyone is best "
                             "visible later, e.g., 300 to start at frame 300.")
    parser.add_argument('--founding_window', type=int, default=60,
                        help="Length in frames of the founders window. Default 60 "
                             "(= 2s @30fps). Tracks visible inside [founding_start, "
                             "founding_start+founding_window] are the candidate founders.")
    parser.add_argument('--dropped_json', default=None,
                        help="Path to .dropped.json sidecar (auto-detected from video name if omitted)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S')

    run_pipeline(
        json_folder=args.json_folder,
        input=args.input,
        output_dir=args.output_dir,
        n_target=args.n_target,
        auto_thresh=args.auto_thresh,
        review_thresh=args.review_thresh,
        device=args.device,
        half=not args.no_half,
        track_buffer=args.track_buffer,
        cmc_method=args.cmc_method,
        min_track_length=args.min_track_length,
        max_frames=args.max_frames,
        skip_review=args.skip_review,
        skip_poc=args.skip_poc,
        skip_merger=args.skip_merger,
        skip_swap_fixer=args.skip_swap_fixer,
        skip_split_jumps=args.skip_split_jumps,
        skip_split_appearance=args.skip_split_appearance,
        split_max_speed=args.split_max_speed,
        split_min_jump=args.split_min_jump,
        appearance_sim_threshold=args.appearance_sim_threshold,
        dropped_json=args.dropped_json,
        classifier_state=args.classifier_state,
        no_classifier=args.no_classifier,
        no_retrain=args.no_retrain,
        auto_newcomers_thresh=args.auto_newcomers,
        auto_founder_thresh=args.auto_founder,
        auto_margin=args.auto_margin,
        backbone=args.backbone,
        dinov2_size=args.dinov2_size,
        head_path=args.head,
        founding_start=args.founding_start,
        founding_window=args.founding_window,
        skip_validate=args.skip_validate,
        resume_validate=args.resume_validate,
    )


if __name__ == '__main__':
    main()
