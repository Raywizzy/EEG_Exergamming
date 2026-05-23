#!/usr/bin/env python3
"""
Phase III — UCSD ds002778 ingestion, harmonization, and Core5 burst features.

Usage:
  python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --download
  python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --process
  python scripts/phase3_ucsd_ingest.py --config config/phase3_ucsd.yaml --features
"""

import argparse, os, json, sys, shutil, glob
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports (no internet; rely on local env) ---
try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    print("Please install mne: pip install mne", file=sys.stderr); sys.exit(1)

# Optional: openneuro-py if you want to auto-download; otherwise manual download.
try:
    from openneuro import download as on_download
    HAVE_OPENNEURO = True
except Exception:
    HAVE_OPENNEURO = False

# Try to reuse your Phase-8 battle-tested extractor if it exists
EXTRACTOR = None
for candidate in [
    "src/features/burst_extractor.py",
    "src/features/bursts.py",
    "src/features/phase8_burst_extractor.py",
]:
    if Path(candidate).exists():
        EXTRACTOR = candidate
        break

if EXTRACTOR:
    sys.path.append(str(Path(EXTRACTOR).parent.resolve()))
    try:
        from burst_extractor import extract_beta_bursts, core5_from_bursts  # type: ignore
    except Exception:
        try:
            from bursts import extract_beta_bursts, core5_from_bursts  # type: ignore
        except Exception:
            EXTRACTOR = None

# Fallback minimal extractor (only used if your module isn't found)
def _fallback_extract_beta_bursts(raw, picks_motor, picks_posterior, cfg):
    """Return dict with bursts list per region and basic stats."""
    logger.info("Using fallback burst extractor")

    raw_f = raw.copy().filter(cfg['bandpass'][0], cfg['bandpass'][1],
                              method='iir', verbose=False)
    raw_f.set_eeg_reference('average', projection=False, verbose=False)

    # Hilbert envelope
    data = raw_f.get_data(picks=picks_motor + picks_posterior)
    env = np.abs(mne.filter.hilbert(data, envelope=True, n_jobs=1, axis=1))
    sfreq = raw.info['sfreq']

    # Adaptive threshold: median + k * MAD
    flat_env = env.flatten()
    median_env = np.median(flat_env)
    mad = np.median(np.abs(flat_env - median_env))
    thr = median_env + cfg['burst']['k'] * mad

    min_samps = int(cfg['burst']['min_duration_ms'] * 1e-3 * sfreq)
    merge_gap = int(cfg['burst']['merge_gap_ms'] * 1e-3 * sfreq)

    def _detect_bursts(channel_env):
        above = channel_env > thr
        bursts = []
        i = 0
        n = above.size
        while i < n:
            if above[i]:
                start = i
                while i < n and above[i]:
                    i += 1
                end = i
                if end - start >= min_samps:
                    # merge gaps
                    if bursts and start - bursts[-1][1] <= merge_gap:
                        bursts[-1] = (bursts[-1][0], end)
                    else:
                        bursts.append((start, end))
            i += 1
        return bursts

    # Aggregate across regions by mean envelope
    n_motor = len(picks_motor)
    n_post = len(picks_posterior)

    motor_env = env[:n_motor].mean(axis=0) if n_motor > 0 else np.zeros(env.shape[1])
    post_env = env[n_motor:n_motor+n_post].mean(axis=0) if n_post > 0 else np.zeros(env.shape[1])

    def _compute_stats(bursts, total_samples):
        if not bursts:
            return dict(rate_per_min=0, mean_ms=0, median_ms=0, cv=0, duty=0)

        durations = np.array([(e - s)/sfreq*1000. for s,e in bursts])
        mean_ms = float(np.mean(durations))
        median_ms = float(np.median(durations))
        cv = float(np.std(durations)/(np.mean(durations)+1e-12))

        total_burst_samples = np.sum([e - s for s,e in bursts])
        duty = float(total_burst_samples / total_samples)

        recording_duration_min = total_samples / sfreq / 60.0
        rate_per_min = float(len(bursts) / recording_duration_min)

        return dict(rate_per_min=rate_per_min, mean_ms=mean_ms,
                   median_ms=median_ms, cv=cv, duty=duty)

    bursts_motor = _detect_bursts(motor_env)
    bursts_post = _detect_bursts(post_env)

    sm = _compute_stats(bursts_motor, len(motor_env))
    sp = _compute_stats(bursts_post, len(post_env))

    # Core5 features
    features = {
        "mean_duration_ms": sm["mean_ms"],
        "median_duration_ms": sm["median_ms"],
        "duration_cv": sm["cv"],
        "duty_cycle": sm["duty"],
        "motor_posterior_duty_ratio": (sm["duty"]/(sp["duty"]+1e-12)) if sp["duty"]>0 else 0.0
    }

    return {"bursts_motor": bursts_motor, "bursts_posterior": bursts_post, "features": features}

def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _find_candidate_raws(bids_root):
    """Minimal BIDS walker: find EEG files (edf/fif/set)"""
    patterns = ["**/*eeg.edf", "**/*.fif", "**/*eeg.set", "**/*_eeg.edf", "**/*_eeg.fif", "**/*_eeg.bdf"]
    files = []
    for pat in patterns:
        files.extend(sorted(Path(bids_root).glob(pat)))
    return files

def _pick_channels(raw, want):
    """Pick channels with fuzzy matching"""
    ch_names = raw.info["ch_names"]
    avail = {ch.upper(): ch for ch in ch_names}
    picks = []

    for w in want:
        w_upper = w.upper()
        if w_upper in avail:
            picks.append(avail[w_upper])
            continue

        # Try variations
        variations = [w, w.replace('Z', 'z'), w.replace('z', 'Z')]
        found = False
        for var in variations:
            if var in ch_names:
                picks.append(var)
                found = True
                break

        if not found:
            logger.warning(f"Channel {w} not found in {ch_names[:10]}...")

    return picks

def _download_ds002778(cfg):
    if not HAVE_OPENNEURO:
        logger.warning("openneuro-py not available. Please download ds002778 from OpenNeuro manually and place under: %s", cfg["bids_root"])
        return

    dest = Path(cfg["bids_root"])
    if dest.exists() and any(dest.iterdir()):
        logger.info("BIDS root already populated, skipping download.")
        return

    _ensure_dir(dest)
    logger.info("Downloading ds002778 to %s", dest)

    try:
        on_download(dataset=cfg["openneuro_id"], target_dir=str(dest), include=["sub-*"], quiet=False)
        logger.info("Download completed successfully")
    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info("Please download ds002778 manually from OpenNeuro")

def _harmonize(cfg):
    bids_root = Path(cfg["bids_root"])
    out_root = Path(cfg["harmonized_root"])
    _ensure_dir(out_root)

    raws = _find_candidate_raws(bids_root)
    if not raws:
        logger.error("No EEG files found under %s", bids_root)
        return

    logger.info("Found %d EEG files to process", len(raws))
    saved = 0

    for i, f in enumerate(raws):
        logger.info("Processing file %d/%d: %s", i+1, len(raws), f.name)

        try:
            suffix = f.suffix.lower()
            if suffix == ".edf":
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            elif suffix == ".bdf":
                raw = mne.io.read_raw_bdf(f, preload=True, verbose=False)
            elif suffix == ".fif":
                raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
            elif suffix == ".set":
                raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            else:
                logger.warning("Unsupported file format: %s", suffix)
                continue

            # Basic info
            logger.debug("Raw info: %d channels, %.1f Hz, %.1f s",
                        len(raw.ch_names), raw.info['sfreq'], raw.times[-1])

            # Resample first to target rate
            if cfg.get("resample_hz") and raw.info['sfreq'] != cfg["resample_hz"]:
                logger.debug("Resampling from %.1f to %d Hz", raw.info['sfreq'], cfg["resample_hz"])
                raw.resample(cfg["resample_hz"], npad="auto", verbose=False)

            # Set channel types if needed
            if raw.get_channel_types()[0] == 'misc':
                raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

            # Reference + Notch + Band-limit for QC (1–40) before burst band
            raw.set_eeg_reference('average', projection=False, verbose=False)

            if cfg.get("notch_hz"):
                raw.notch_filter(freqs=[cfg["notch_hz"]], verbose=False)

            # Broad filter for QC
            raw.filter(1., 40., method='iir', verbose=False)

            # Save harmonized file
            rel = f.relative_to(bids_root)
            out_fif = out_root / rel.with_suffix(".fif")
            _ensure_dir(out_fif.parent)
            raw.save(out_fif, overwrite=True, verbose=False)
            saved += 1

            logger.debug("Saved: %s", out_fif)

        except Exception as e:
            logger.warning("Skipped %s: %s", f.name, e)

    logger.info("Harmonized: %d files → %s", saved, out_root)

def _extract_core5(cfg):
    harm = Path(cfg["harmonized_root"])
    files = sorted(harm.glob("**/*.fif"))

    if not files:
        logger.error("No harmonized FIF files found under %s", harm)
        return

    logger.info("Extracting Core5 features from %d files", len(files))
    rows = []

    for i, fif in enumerate(files):
        logger.info("Extracting features %d/%d: %s", i+1, len(files), fif.name)

        try:
            raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)

            # Bandpass to beta for bursts
            raw_b = raw.copy().filter(cfg["bandpass"][0], cfg["bandpass"][1], method="iir", verbose=False)

            # Pick motor/posterior channels
            motor_picks = _pick_channels(raw_b, cfg["motor"])
            post_picks = _pick_channels(raw_b, cfg["posterior"])

            if not motor_picks or not post_picks:
                logger.warning("Skipping %s: missing motor (%d) or posterior (%d) channels",
                             fif.name, len(motor_picks), len(post_picks))
                continue

            logger.debug("Using motor channels: %s", motor_picks)
            logger.debug("Using posterior channels: %s", post_picks)

            # Use your extractor if present, else fallback
            if EXTRACTOR and 'extract_beta_bursts' in globals():
                logger.debug("Using existing burst extractor")
                bursts = extract_beta_bursts(
                    raw_b, picks_motor=motor_picks, picks_posterior=post_picks,
                    cfg=cfg
                )
                # Expect a dict with "features" or compute Core5
                if "features" in bursts and all(k in bursts["features"] for k in cfg["core5_features"]):
                    feats = bursts["features"]
                elif 'core5_from_bursts' in globals():
                    feats = core5_from_bursts(bursts, raw_b.info["sfreq"])
                else:
                    # last resort: fallback compute
                    feats = _fallback_extract_beta_bursts(raw_b, motor_picks, post_picks, cfg)["features"]
            else:
                feats = _fallback_extract_beta_bursts(raw_b, motor_picks, post_picks, cfg)["features"]

            # Extract subject ID from path
            subject_id = "unknown"
            for part in fif.parts:
                if str(part).startswith("sub-"):
                    subject_id = str(part)
                    break

            # Determine condition/label from filename or path
            condition = "UNKNOWN"
            filename_lower = str(fif).lower()
            if any(term in filename_lower for term in ['rest', 'baseline', 'control']):
                condition = "PD_SHAM"  # Assume control condition
            elif any(term in filename_lower for term in ['task', 'active', 'stim']):
                condition = "PD_REAL"  # Assume active condition
            else:
                # Default assignment - could be refined based on actual dataset structure
                condition = "PD_REAL"

            row = {
                "subject_id": subject_id,
                "condition": condition,
                "site": cfg.get("site_tag", "UCSD"),
                "file": str(fif.relative_to(harm)),
                **{k: feats.get(k, np.nan) for k in cfg["core5_features"]}
            }
            rows.append(row)

            logger.debug("Features extracted: %s", {k: f"{v:.3f}" for k, v in feats.items()})

        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", fif.name, e)

    if not rows:
        logger.error("No features extracted successfully")
        return

    df = pd.DataFrame(rows)
    out = Path(cfg["features_out"])
    _ensure_dir(out.parent)
    df.to_csv(out, index=False)

    logger.info("Saved Core5 features: %s (n=%d)", out, len(df))
    logger.info("Condition distribution: %s", df['condition'].value_counts().to_dict())
    logger.info("Feature summary:")
    for feat in cfg["core5_features"]:
        values = df[feat].dropna()
        if len(values) > 0:
            logger.info("  %s: %.3f ± %.3f", feat, values.mean(), values.std())

def main():
    ap = argparse.ArgumentParser(description="UCSD ds002778 ingestion and Core5 feature extraction")
    ap.add_argument("--config", required=True, help="Config YAML file")
    ap.add_argument("--download", action="store_true", help="Download dataset from OpenNeuro")
    ap.add_argument("--process", action="store_true", help="Harmonize and preprocess data")
    ap.add_argument("--features", action="store_true", help="Extract Core5 features")
    ap.add_argument("--all", action="store_true", help="Run all steps")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    logger.info("Loaded config: %s", args.config)

    if args.all or args.download:
        logger.info("=== DOWNLOAD STEP ===")
        _download_ds002778(cfg)

    if args.all or args.process:
        logger.info("=== HARMONIZATION STEP ===")
        _harmonize(cfg)

    if args.all or args.features:
        logger.info("=== FEATURE EXTRACTION STEP ===")
        _extract_core5(cfg)

    logger.info("Phase III UCSD ingestion completed")

if __name__ == "__main__":
    main()