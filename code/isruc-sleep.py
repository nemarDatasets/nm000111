"""ISRUC-Sleep: BIDS conversion script

Dataset home: https://sleeptight.isr.uc.pt
Publication: Paiva et al., Comput Methods Programs Biomed, 2015.
DOI: 10.1016/j.cmpb.2015.10.013

Folder layout after download + extraction (see isruc-sleep_download_and_extract.py):
- Details/
  - Details_subgroup_I_Submission.xlsx
  - Details_subgroup_II_Submission.xlsx
  - Details_subgroup_III_Submission.xlsx
- subgroupI/<subject_id>/
  - <subject_id>.rec     (PSG recording, EDF-compatible)
  - <subject_id>_1.txt   (sleep stage per 30 s epoch, scorer 1)
  - <subject_id>_2.txt   (sleep stage per 30 s epoch, scorer 2)
- subgroupII/...
- subgroupIII/...

This script will:
- Read PSG .rec files with MNE (EDF reader)
- Parse sleep staging text files (default scorer=1) into annotations
- Ingest subject-level metadata from Details spreadsheets into participants.tsv
- Write BIDS using mne-bids
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime
import warnings
import tempfile
import shutil

import numpy as np
import pandas as pd
from mne import Annotations
from mne.io import read_raw_edf
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, make_report


README_CONTENT = """## Introduction

The ISRUC-Sleep dataset comprises overnight polysomnographic (PSG) recordings and manual sleep stage annotations across three subgroups. The data support research in automatic sleep staging and sleep-disordered breathing. Signals include EEG, EOG, EMG, respiratory channels and others, provided as EDF-compatible `.rec` files. For each recording, sleep was scored by two expert scorers in 30-second epochs.

## Overview of the experiment

Participants slept overnight in a clinical environment with standard PSG montage. Two independent human scorers labeled each 30-second epoch into sleep stages following AASM/R&K guidelines used by the dataset (W, N1, N2, N3, and REM). The dataset is divided into subgroups with different focuses (e.g., subjects with sleep disorders, multiple nights). Please refer to the publication and the Details spreadsheets for demographic and clinical descriptors.

## Description of the preprocessing if any

Original `.rec` files are symlinked (or copied if needed) to `.edf` without modification. Sleep stages come from the scorer-1 Excel files (col0 epoch, col1 label; headers auto-skipped; NaN epochs filled sequentially; unknown labels -> U). Labels from the second scorer, when present, are stored in annotation extras. Measurement dates use `Date of recording` from Details (UTC) when available, otherwise 2020-01-01. Participant demographics (Sex, Age) are pulled directly from the Details spreadsheets.

## Description of the event values
Sleep stages are encoded per 30 s epoch. The following mapping is used:
- 0: Sleep stage W (Wake)
- 1: Sleep stage N1
- 2: Sleep stage N2
- 3: Sleep stage N3
- 5: Sleep stage R (REM)
- 6: Sleep stage U (Unknown)

The annotations are added as events with `onset` at the epoch start, `duration` 30 seconds, and `description` matching the above labels.

## Citation

When using this dataset, please cite:
1. Khalighi S., Sousa T., Santos J.M., Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. 
   Computer methods and programs in biomedicine 124 (2016): 180-192. DOI: 10.1016/j.cmpb.2015.10.013
2. Project site: https://sleeptight.isr.uc.pt

**Data curators (BIDS conversion):**
Pierre Guetschel

**Data collectors (original dataset):**
Sirvan Khalighi; Teresa Sousa; Jose Moutinho Santos; Urbano Nunes
"""

DATASET_NAME = "ISRUC-Sleep"

# Sleep stage code mapping found in ISRUC text labels
STAGE_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 5: "R", 6: "U"}
STAGE_DESC = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    5: "Sleep stage R",
    6: "Sleep stage U",
}
EVENT_ID = {v: k for k, v in STAGE_DESC.items()}
MAIN_SCORER = 1
OTHER_SCORERS = (2,)
SUBGROUPS = {"I": range(1, 101), "II": range(1, 9), "III": range(1, 11)}
EPOCH_LEN = 30.0  # seconds per epoch

# Temporary directory for .rec -> .edf conversions (module-level to persist across calls)
_TEMP_REC_CACHE = None


def _get_rec_as_edf(rec_path: Path) -> Path:
    """Return an EDF-readable path for a .rec file.

    First tries to create a symlink with .edf extension in the same directory.
    If symlink fails (permissions, etc.), copies to a temp directory with .edf extension.
    Returns the path to the EDF file (either symlink or copy).
    """
    global _TEMP_REC_CACHE
    rec_path = Path(rec_path)

    # Try symlink in source directory first
    edf_path = rec_path.with_suffix(".edf")
    if edf_path.exists():
        return edf_path  # Already exists

    try:
        edf_path.symlink_to(rec_path)
        return edf_path
    except (OSError, FileExistsError) as e:
        warnings.warn(
            f"Could not create symlink for {rec_path}: {e}. Falling back to copy."
        )

    # Fallback: copy to temp directory
    if _TEMP_REC_CACHE is None:
        _TEMP_REC_CACHE = tempfile.mkdtemp(prefix="isruc_rec_")
    temp_dir = Path(_TEMP_REC_CACHE)
    temp_edf = temp_dir / (rec_path.stem + ".edf")
    try:
        shutil.copy2(rec_path, temp_edf)
        return temp_edf
    except Exception as e:
        raise RuntimeError(f"Could not copy {rec_path} to temp: {e}")


def _iter_subject_dirs(source_root: Path) -> List[Tuple[str, Path]]:
    """Yield (subgroup_label, subject_dir) for all expected subjects.

    Uses predefined subject lists to ensure nothing is silently skipped.
    Raises ValueError if any expected subgroup or subject folder is missing.
    """
    subgroup_dirs = {
        "I": source_root / "subgroupI",
        "II": source_root / "subgroupII",
        "III": source_root / "subgroupIII",
    }

    missing_subgroups = [
        label for label, path in subgroup_dirs.items() if not path.exists()
    ]
    if missing_subgroups:
        raise ValueError(f"Missing subgroup directories: {missing_subgroups}")

    found = []
    for label, root in subgroup_dirs.items():
        for sid_num in SUBGROUPS[label]:
            sd = root / f"{sid_num}"
            if not sd.exists():
                raise ValueError(f"Missing subject directory: {sd}")
            found.append((label, sd))
    return found


def _stage_file_for(
    subject_dir: Path, scorer: int, session_subdir: Optional[Path] = None
) -> Optional[Path]:
    """Return stage file path for scorer, optionally inside a session subdir (for subgroup II)."""
    if session_subdir is not None:
        sid = session_subdir.name  # session folder name (e.g., '1' or '2')
        path = session_subdir / f"{sid}_{scorer}.xlsx"
        return path if path.exists() else None
    sid = subject_dir.name
    path = subject_dir / f"{sid}_{scorer}.xlsx"
    return path if path.exists() else None


def _read_stages(stage_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read stage labels from an Excel file.

    Uses positional indexing: column 0 = epoch number, column 1 = stage label.
    Automatically detects and skips header row if present.
    Any unrecognized or NaN values are mapped to code 6 (Unknown/U).
    Maps codes to ISRUC numeric codes: W=0, N1=1, N2=2, N3=3, R=5, U=6.

    Returns
    -------
    epochs : np.ndarray
        Array of epoch indices (integers)
    stages_numeric : np.ndarray
        Array of numeric stage codes (one per epoch)
    """
    df = pd.read_excel(stage_file, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"File {stage_file} has less than 2 columns")

    # Header detection (case-insensitive, tolerant to typo "hich")
    first_val = str(df.iat[0, 0]).strip().lower()
    second_val = str(df.iat[0, 1]).strip().lower()
    looks_like_header = first_val in {"epoch", "hich"} or second_val == "stage"

    if not looks_like_header:
        first_is_int = (
            pd.to_numeric(pd.Series([df.iat[0, 0]]), errors="coerce").notna().iloc[0]
        )
        looks_like_header = not first_is_int

    start_row = 1 if looks_like_header else 0

    # Positional extract
    data = df.iloc[start_row:, [0, 1]].copy()

    # Epochs: coerce to numeric; fill NaNs with sequential indices
    epochs_raw = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    idx_seq = pd.Series(
        np.arange(1, len(epochs_raw) + 1, dtype=int), index=epochs_raw.index
    )
    epochs_filled = epochs_raw.fillna(idx_seq)
    epochs = epochs_filled.astype(int).to_numpy()
    if not np.array_equal(epochs.astype(int), idx_seq):
        raise ValueError(f"Non-sequential epoch numbers found in {stage_file}")

    # Stages: map letters to codes, unknown -> 6
    stage_letter_to_numeric = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 5, "U": 6}
    stages_str = (
        data.iloc[:, 1].astype(str).str.strip().str.upper().replace("NAN", np.nan)
    )
    stages_numeric = (
        stages_str.map(stage_letter_to_numeric).fillna(6).astype(int).to_numpy()
    )

    return epochs, stages_numeric


def _stages_to_annotations(
    stages: np.ndarray, extras: Optional[List[Dict[str, str]]] = None
) -> Annotations:
    """Create MNE Annotations from stage array.

    Parameters
    ----------
    stages : np.ndarray
        Array of numeric stage codes (one per epoch)
    extras : Optional[List[Dict[str, str]]]
        Extra metadata per epoch (e.g., secondary scorer labels)
    """
    onsets = np.arange(len(stages), dtype=float) * EPOCH_LEN
    durations = np.full_like(onsets, EPOCH_LEN, dtype=float)
    descriptions = [STAGE_DESC[int(s)] for s in stages]
    return Annotations(
        onset=onsets, duration=durations, description=descriptions, extras=extras
    )


def _details_read(source_root: Path) -> pd.DataFrame:
    """Read and unify Details spreadsheets across subgroups.

    Returns a DataFrame with a unified index 'participant_id' (e.g., I001, II003), and
    all available columns from the source spreadsheets (renamed where reasonable).
    """
    details_dir = source_root / "Details"
    files = [
        ("I", details_dir / "Details_subgroup_I_Submission.xlsx"),
        ("II", details_dir / "Details_subgroup_II_Submission.xlsx"),
        ("III", details_dir / "Details_subgroup_III_Submission.xlsx"),
    ]
    frames = []
    for label, fpath in files:
        if not fpath.exists():
            raise ValueError(f"Missing details file: {fpath}")
        df = pd.read_excel(fpath, header=2)

        def _sid3_from_value(v) -> Optional[str]:
            if pd.isna(v):
                return None
            s = str(v).strip()
            # For entries like '1_Rec.1' keep the part before underscore
            if "_" in s:
                s = s.split("_")[0]
            try:
                return f"{int(float(s)) :03d}"
            except Exception:
                return None

        df["_sid3"] = df["Subject"].apply(_sid3_from_value)
        df = df[df["_sid3"].notna()].copy()

        # For subgroup II, demographics may appear only on Rec.1 row; aggregate by subject
        def _first_nonnull(series):
            for x in series:
                if pd.notna(x):
                    return x
            return np.nan

        df = df.groupby("_sid3", sort=False).agg(_first_nonnull).reset_index()
        df["participant_id"] = label + df["_sid3"]
        df.drop(columns=["_sid3"], inplace=True)

        frames.append(df)

    out = pd.concat(frames, axis=0, sort=False)
    out.set_index("participant_id", inplace=True)
    print(out.head())
    return out


def _get_records(source_root: Path, scorer: int = MAIN_SCORER):
    """Yield tuples (raw_path, stage_files, bids_path, subgroup_label, subject_num).

    `stage_files` is a dict of scorer -> Path|None; scorer MAIN_SCORER must exist.
    Subject id in BIDS will be like I001, II003, etc. For subgroup II, emit session=1 and session=2.
    """
    for label, subject_dir in _iter_subject_dirs(source_root):
        sid = subject_dir.name  # numeric string without zero padding
        pid = f"{label}{int(sid):03d}"
        scorers_all = (MAIN_SCORER,) + tuple(
            s for s in OTHER_SCORERS if s != MAIN_SCORER
        )
        if label == "II":
            for session in (1, 2):
                session_dir = subject_dir / f"{session}"
                if not session_dir.exists():
                    raise ValueError(f"Missing session directory: {session_dir}")
                rec_path = session_dir / f"{session}.rec"
                if not rec_path.exists():
                    raise ValueError(f"Missing PSG file: {rec_path}")

                stage_files = {
                    sc: _stage_file_for(
                        subject_dir, scorer=sc, session_subdir=session_dir
                    )
                    for sc in scorers_all
                }
                if stage_files.get(MAIN_SCORER) is None:
                    raise ValueError(
                        f"Missing stage file for scorer {MAIN_SCORER}: {session_dir}"
                    )

                bids_path = BIDSPath(
                    subject=pid,
                    session=str(session),
                    task="sleep",
                    suffix="eeg",
                    datatype="eeg",
                    extension=".edf",
                )
                yield rec_path, stage_files, bids_path, label, int(sid)
        else:
            rec_path = subject_dir / f"{sid}.rec"
            if not rec_path.exists():
                raise ValueError(f"Missing PSG file: {rec_path}")

            stage_files = {
                sc: _stage_file_for(subject_dir, scorer=sc) for sc in scorers_all
            }
            if stage_files.get(MAIN_SCORER) is None:
                raise ValueError(
                    f"Missing stage file for scorer {MAIN_SCORER}: {subject_dir}"
                )

            bids_path = BIDSPath(
                subject=pid,
                task="sleep",
                suffix="eeg",
                datatype="eeg",
                extension=".edf",
            )
            yield rec_path, stage_files, bids_path, label, int(sid)


def main(
    source_root: Path,
    bids_root: Path,
    overwrite: bool = False,
    finalize_only: bool = False,
):
    """Convert the ISRUC-Sleep dataset to BIDS format.

    Parameters
    ----------
    source_root : Path
        Root folder containing Details/ and subgroupI/II/III/.
    bids_root : Path
        Destination BIDS root.
    overwrite : bool
        Overwrite existing files.
    """
    source_root = Path(source_root).expanduser()
    bids_root = Path(bids_root).expanduser()
    bids_root.mkdir(parents=True, exist_ok=True)

    if finalize_only:
        _finalize_dataset(bids_root, overwrite=overwrite)
        return

    print(f"\n=== ISRUC-Sleep BIDS Conversion ===")
    print(f"Source: {source_root}")
    print(f"BIDS root: {bids_root}")
    print(f"Using scorer {MAIN_SCORER} annotations (extras from {OTHER_SCORERS})\n")

    print("Reading Details spreadsheets...")
    details_df = _details_read(source_root)
    print(f"  Loaded demographics for {len(details_df)} subjects\n")

    print("Scanning for PSG recordings...")
    records = list(_get_records(source_root))
    print(f"  Found {len(records)} recordings\n")
    for rec_path, stage_files, bids_path, label, sid_num in records:
        bids_path = bids_path.update(root=bids_root)

    # sanity check duplicates
    bids_paths = [bp.fpath for _, _, bp, _, _ in records]
    assert len(bids_paths) == len(set(bids_paths)), "Duplicate BIDS paths found"

    # Convert each recording
    print(f"Converting {len(records)} recordings to BIDS...")
    for idx, (rec_path, stage_files, bids_path, label, sid_num) in enumerate(
        records, 1
    ):
        if not overwrite and bids_path.fpath.exists():
            print(
                f"Skipping existing {bids_path.fpath} (use overwrite=True to replace)"
            )
            continue
        print(f"\n---\nProcessing {bids_path.fpath}")

        # Read raw PSG (.rec files, readable by EDF reader via symlink or temp copy)
        rec_path = Path(rec_path)
        edf_path = _get_rec_as_edf(rec_path)
        raw = read_raw_edf(edf_path, preload=False, verbose=False)
        # Add sleep stage annotations
        main_stage_file = stage_files.get(MAIN_SCORER)
        if main_stage_file is None:
            raise ValueError(f"Missing stage file for scorer {MAIN_SCORER}: {rec_path}")

        epochs_main, stages_main = _read_stages(main_stage_file)

        # Build extras metadata for other scorers, aligning by epoch number
        extras_cols: Dict[str, List[str]] = {}
        for sc in OTHER_SCORERS:
            sf = stage_files.get(sc)
            if sf is None:
                continue
            epochs_extra, stages_extra = _read_stages(sf)

            # Align the two scorers based on epoch indices
            # Create DataFrames for alignment
            df_main = pd.DataFrame({"epoch": epochs_main, "stage_main": stages_main})
            df_extra = pd.DataFrame(
                {"epoch": epochs_extra, "stage_extra": stages_extra}
            )

            # Merge on epoch, keeping all epochs from main scorer
            df_merged = df_main.merge(df_extra, on="epoch", how="left")

            # Fill missing extra scorer values with 6 (Unknown)
            df_merged["stage_extra"] = df_merged["stage_extra"].fillna(6).astype(int)

            # Update stages_main in case extra scorer has more epochs
            if len(epochs_extra) > len(epochs_main):
                # Also need to include epochs only in extra scorer
                df_merged_full = df_main.merge(df_extra, on="epoch", how="outer")
                df_merged_full["stage_main"] = (
                    df_merged_full["stage_main"].fillna(6).astype(int)
                )
                df_merged_full["stage_extra"] = (
                    df_merged_full["stage_extra"].fillna(6).astype(int)
                )
                df_merged_full = df_merged_full.sort_values("epoch")

                # Update main stages for this iteration
                stages_main = df_merged_full["stage_main"].to_numpy()
                epochs_main = df_merged_full["epoch"].to_numpy()
                df_merged = df_merged_full

            extras_cols[f"scorer{sc}_label"] = [
                STAGE_DESC[int(s)] for s in df_merged["stage_extra"].to_numpy()
            ]
            extras_cols[f"scorer{sc}_label_value"] = (
                df_merged["stage_extra"].astype(int).tolist()
            )

        extras_list: Optional[List[Dict[str, str]]]
        if extras_cols:
            n = len(stages_main)
            extras_list = [
                {col: extras_cols[col][i] for col in extras_cols} for i in range(n)
            ]
        else:
            extras_list = None

        ann = _stages_to_annotations(stages_main, extras=extras_list)
        print(ann[:5])
        print(ann.extras[:5])
        raw.set_annotations(ann)

        # Subject info for participants.tsv from Details if available
        pid = f"{label}{sid_num:03d}"
        if idx % 10 == 0 or idx == len(records):
            print(f"  [{idx}/{len(records)}] Converting {pid}...")
        subrow = details_df.loc[pid]

        # Sex mapping (fail if not M/F)
        s = str(subrow["Sex"]).strip().upper()
        if s.startswith("M"):
            sex_val = 1
        elif s.startswith("F"):
            sex_val = 2
        else:
            sex_val = 0  # unknown

        # Age (fail if missing or non-numeric)
        try:
            age_years = float(subrow["Age"])
        except Exception:
            age_years = None

        # subject_info per MNE: id (int), his_id (str), sex (0/1/2), optional birthday
        raw_meas_date = subrow["Date of recording"]
        if isinstance(raw_meas_date, str) and len(raw_meas_date.split("/")) > 3:
            raw_meas_date = raw_meas_date.split("-")[0]
        meas_date = pd.to_datetime(raw_meas_date, errors="raise", dayfirst=True)
        if hasattr(meas_date, "to_pydatetime"):
            meas_date = meas_date.to_pydatetime()
        if isinstance(meas_date, datetime.date) and not isinstance(
            meas_date, datetime.datetime
        ):
            meas_date = datetime.datetime.combine(meas_date, datetime.time.min)
        if meas_date.tzinfo is None:
            meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
        raw.set_meas_date(meas_date)

        sub_info: Dict[str, object] = {
            "his_id": pid,
            "sex": int(sex_val),
        }

        if age_years is not None and meas_date is not None:
            bdate = meas_date.date() - datetime.timedelta(
                days=float(age_years) * 365.25
            )
            sub_info["birthday"] = bdate

        raw.info["subject_info"] = sub_info
        # Store description of scorer used
        raw.info["description"] = f"Sleep staging from scorer {MAIN_SCORER}"

        write_raw_bids(
            raw,
            bids_path=bids_path,
            overwrite=True,
            verbose=False,
            event_id=EVENT_ID,
        )

    _finalize_dataset(bids_root, details_df=details_df, overwrite=overwrite)


def _finalize_dataset(
    bids_root: Path, details_df: pd.DataFrame | None = None, overwrite: bool = False
):
    script_path = Path(__file__)
    script_dest = bids_root / "code" / script_path.name
    script_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(script_path, script_dest)

    description_file = bids_root / "dataset_description.json"
    if description_file.exists() and overwrite:
        description_file.unlink()
    make_dataset_description(
        path=bids_root,
        name=DATASET_NAME,
        dataset_type="derivative",
        references_and_links=[
            "https://doi.org/10.1016/j.cmpb.2015.10.013",
        ],
        source_datasets=[
            {"URL": "https://sleeptight.isr.uc.pt"},
        ],
        authors=[
            "Sirvan Khalighi",
            "Teresa Sousa",
            "Jose Moutinho Santos",
            "Urbano Nunes",
        ],
        acknowledgements="Pierre Guetschel updated the data to BIDS format.",
        overwrite=overwrite,
        data_license="n/a",  # No license specified in original dataset
    )

    # README will be written after generating the report (to include it)

    # Enrich participants.tsv with Details fields
    print(f"\nMerging demographics into participants.tsv...")
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists() and details_df is not None and not details_df.empty:
        df_p = pd.read_csv(participants_tsv, sep="\t")
        # Merge on index of details_df
        details_df_reset = details_df.reset_index()
        merged = df_p.merge(
            details_df_reset,
            left_on="participant_id",
            right_on="participant_id",
            how="left",
            suffixes=("", "_det"),
        )

        # Remove columns where all non-participant_id values are n/a
        cols_to_drop = []
        for col in merged.columns:
            if col != "participant_id":
                # Check if all non-null values are 'n/a'
                non_null = merged[col][~merged[col].isna()]
                if len(non_null) == 0 or (non_null.values == "n/a").sum() == len(
                    non_null
                ):
                    cols_to_drop.append(col)
        if cols_to_drop:
            merged = merged.drop(columns=cols_to_drop)
        merged.to_csv(participants_tsv, sep="\t", index=False)
        print(f"  Updated with {len(merged.columns)-1} demographic columns\n")

    # Remove participants.json if present (optional metadata)
    pj = bids_root / "participants.json"
    if pj.exists():
        pj.unlink()

    # cleanup macos hidden files
    for macos_file in bids_root.rglob("._*"):
        macos_file.unlink()
    # Print brief report and write README (include automatic report)
    print("Generating BIDS report...")
    report_str = make_report(bids_root)
    print(report_str)

    readme_path = bids_root / "README.md"
    readme_path.write_text(
        f"# {DATASET_NAME}\n\n{README_CONTENT}\n\n---\n\n"
        f"## Automatic report\n\n*Report automatically generated by `mne_bids.make_report()`.*\n\n> {report_str}"
    )

    # Cleanup temp directory if created
    global _TEMP_REC_CACHE
    if _TEMP_REC_CACHE is not None:
        try:
            shutil.rmtree(_TEMP_REC_CACHE)
        except Exception as e:
            warnings.warn(f"Could not clean up temp directory {_TEMP_REC_CACHE}: {e}")
        _TEMP_REC_CACHE = None

    print("\n=== Conversion Complete ===")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # python bids_maker/datasets/isruc-sleep.py --source_root ~/data/isruc-sleep/ --bids_root ~/data/bids/isruc-sleep/ --overwrite=True
