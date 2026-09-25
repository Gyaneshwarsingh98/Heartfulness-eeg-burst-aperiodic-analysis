# ============================================================
# 01_preprocessing.py
# Continuous EEG preprocessing for burst and aperiodic analysis
#
# Pipeline:
# 1. Load raw EGI EEG (.mff)
# 2. Remove non-EEG reference channel
# 3. Band-pass filter: 0.3–40 Hz
# 4. ICA-based ocular artifact correction
# 5. RANSAC bad-channel detection
# 6. Spherical interpolation of bad channels
# 7. Common-average reference
# 8. Save continuous preprocessed EEG (.fif)
#
# IMPORTANT:
# - The continuous recording is retained throughout preprocessing.
# - No overlapping analysis epochs are concatenated here.
# - RANSAC uses temporary 2-s epochs only for bad-channel detection.
# - The 50-Hz notch filter is intentionally omitted because the
#   signal is low-pass filtered at 40 Hz.
#
# Author: Gyaneshwar Singh
# ============================================================

import os
import numpy as np
import mne
from autoreject import Ransac


# ============================================================
# PARAMETERS
# ============================================================

LOW_CUT = 0.3
HIGH_CUT = 40.0

RANSAC_SEGMENT = 2.0

ICA_N_COMPONENTS = 0.95
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 800

ICA_CORRELATION_THRESHOLD = 0.70

REMOVE_REFERENCE_CHANNEL = "Vertex Reference"


# ============================================================
# ICA
# ============================================================

def apply_ica(raw):
    """
    Perform ICA-based ocular artifact correction.

    ICA is fitted to the continuous EEG recording. Candidate
    ocular components are identified using their maximum absolute
    correlation with predefined frontal EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.

    Returns
    -------
    raw_clean : mne.io.Raw
        ICA-corrected continuous EEG.

    ica : mne.preprocessing.ICA
        Fitted ICA object.

    suggested : ndarray
        ICA components exceeding the correlation threshold.
    """

    print("\n========================================")
    print("ICA ARTIFACT CORRECTION")
    print("========================================")

    # --------------------------------------------------------
    # Fit ICA
    # --------------------------------------------------------

    ica = mne.preprocessing.ICA(
        method="fastica",
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER
    )

    print("Fitting ICA to continuous EEG...")

    ica.fit(raw)

    print(
        f"Estimated ICA components: "
        f"{ica.n_components_}"
    )

    # --------------------------------------------------------
    # Predefined frontal channels
    # --------------------------------------------------------

    frontal_channels = [
        'E17', 'E21', 'E14', 'E22', 'E15', 'E9',
        'E18', 'E16', 'E10', 'E19', 'E11', 'E4',
        'E12', 'E5', 'E25', 'E32', 'E26', 'E23',
        'E38', 'E33', 'E34', 'E28', 'E27', 'E24',
        'E20', 'E8', 'E1', 'E121', 'E2', 'E122',
        'E3', 'E123', 'E124', 'E118', 'E116', 'E117'
    ]

    available_frontal = [
        ch for ch in frontal_channels
        if ch in raw.ch_names
    ]

    if len(available_frontal) == 0:
        raise RuntimeError(
            "No predefined frontal EEG channels were found."
        )

    picks = mne.pick_channels(
        raw.ch_names,
        include=available_frontal
    )

    # --------------------------------------------------------
    # ICA source signals
    # --------------------------------------------------------

    print("Computing ICA-component correlations...")

    ica_sources = ica.get_sources(raw).get_data()

    frontal_data = raw.get_data(
        picks=picks
    )

    # Correlation matrix:
    # ICA components × frontal EEG channels
    corr = np.corrcoef(
        ica_sources,
        frontal_data
    )

    n_ica = ica_sources.shape[0]

    correlations = corr[
        :n_ica,
        n_ica:
    ]

    max_corr = np.max(
        np.abs(correlations),
        axis=1
    )

    suggested = np.where(
        max_corr > ICA_CORRELATION_THRESHOLD
    )[0]

    # --------------------------------------------------------
    # Report candidate components
    # --------------------------------------------------------

    print("\nCandidate ocular ICA components:")

    if len(suggested) == 0:

        print(
            "No components exceeded "
            f"|r| > {ICA_CORRELATION_THRESHOLD:.2f}"
        )

    else:

        for ic in suggested:

            print(
                f"IC {ic:3d}    "
                f"max |r| = {max_corr[ic]:.3f}"
            )

    # --------------------------------------------------------
    # Visual inspection
    # --------------------------------------------------------
    #
    # The candidate components should be visually inspected
    # before being accepted as ocular artifacts.
    #
    # If your original analysis used the automatically detected
    # candidate components after visual QC, retain that procedure.
    #
    # The plots below are generated for QC.
    # --------------------------------------------------------

    if len(suggested) > 0:

        print("\nOpening ICA QC plots...")

        ica.plot_components(
            picks=suggested
        )

        ica.plot_properties(
            raw,
            picks=suggested
        )

        ica.plot_sources(
            raw
        )

    # --------------------------------------------------------
    # Component exclusion
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The candidate list is used as the exclusion list.
    #
    # If visual inspection indicates that a suggested component
    # is NOT ocular, remove its index from this list before
    # applying ICA.
    # --------------------------------------------------------

    ica.exclude = suggested.tolist()

    print(
        "\nICA components marked for removal:",
        ica.exclude
    )

    # --------------------------------------------------------
    # Apply ICA
    # --------------------------------------------------------

    raw_clean = raw.copy()

    if len(ica.exclude) > 0:

        ica.apply(
            raw_clean,
            verbose=True
        )

    print(
        f"\nICA complete."
        f"\nRemoved {len(ica.exclude)} component(s)."
    )

    return raw_clean, ica, suggested


# ============================================================
# RANSAC BAD-CHANNEL DETECTION
# ============================================================

def identify_bad_channels(raw):
    """
    Identify bad EEG channels using RANSAC.

    RANSAC is applied to temporary non-overlapping 2-second
    segments solely for channel-quality assessment.

    The temporary epochs are NOT used for subsequent feature
    extraction and are NOT concatenated.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.

    Returns
    -------
    bad_channels : list
        Channels identified as bad by RANSAC.
    """

    print("\n========================================")
    print("RANSAC BAD-CHANNEL DETECTION")
    print("========================================")

    # --------------------------------------------------------
    # Temporary fixed-length events
    # --------------------------------------------------------

    events = mne.make_fixed_length_events(
        raw,
        start=0,
        duration=RANSAC_SEGMENT
    )

    event_id = {
        "RANSAC": 1
    }

    # --------------------------------------------------------
    # Temporary epochs for RANSAC
    # --------------------------------------------------------

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=RANSAC_SEGMENT,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False
    )

    print(
        f"Temporary RANSAC epochs: "
        f"{len(epochs)}"
    )

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    ransac = Ransac(
        n_jobs=1,
        verbose=True
    )

    ransac.fit(epochs)

    bad_channels = list(
        ransac.bad_chs_
    )

    print(
        "\nBad channels identified by RANSAC:"
    )

    if len(bad_channels) == 0:
        print("None")

    else:
        for ch in bad_channels:
            print(f"  {ch}")

    # --------------------------------------------------------
    # Interpolate bad channels
    # --------------------------------------------------------

    if len(bad_channels) > 0:

        raw.info["bads"] = bad_channels

        print(
            "\nInterpolating bad channels "
            "using spherical interpolation..."
        )

        raw.interpolate_bads(
            reset_bads=True,
            mode="accurate"
        )

        print("Bad-channel interpolation complete.")

    return bad_channels


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def load_and_preprocess_eeg(file_path):
    """
    Load and preprocess one continuous EGI EEG recording.

    Processing order
    ----------------
    1. Load EGI .mff
    2. Remove Vertex Reference channel
    3. Band-pass filter 0.3–40 Hz
    4. ICA ocular artifact correction
    5. RANSAC bad-channel detection
    6. Spherical interpolation
    7. Common-average reference

    Returns
    -------
    raw : mne.io.Raw
        Final continuous preprocessed EEG.

    qc : dict
        Quality-control information.
    """

    print("\n")
    print("========================================")
    print("LOADING EEG")
    print("========================================")

    print(
        f"File: {os.path.basename(file_path)}"
    )

    # --------------------------------------------------------
    # Load EGI recording
    # --------------------------------------------------------

    raw = mne.io.read_raw_egi(
        file_path,
        preload=True,
        verbose=True
    )

    print(
        f"Sampling frequency: "
        f"{raw.info['sfreq']} Hz"
    )

    print(
        f"Number of channels before cleaning: "
        f"{len(raw.ch_names)}"
    )

    print(
        f"Recording duration: "
        f"{raw.times[-1]:.2f} s"
    )

    # --------------------------------------------------------
    # Remove non-EEG reference channel
    # --------------------------------------------------------

    if REMOVE_REFERENCE_CHANNEL in raw.ch_names:

        raw.drop_channels(
            [REMOVE_REFERENCE_CHANNEL]
        )

        print(
            f"Removed channel: "
            f"{REMOVE_REFERENCE_CHANNEL}"
        )

    # --------------------------------------------------------
    # Band-pass filtering
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # No 50-Hz notch filter is applied.
    #
    # Since the upper cutoff is 40 Hz, frequencies around
    # 50 Hz are already attenuated by the low-pass filter.
    # --------------------------------------------------------

    print(
        f"\nApplying band-pass filter: "
        f"{LOW_CUT}–{HIGH_CUT} Hz"
    )

    raw.filter(
        l_freq=LOW_CUT,
        h_freq=HIGH_CUT,
        fir_design="firwin",
        verbose=True
    )

    print("Band-pass filtering complete.")

    # --------------------------------------------------------
    # ICA
    # --------------------------------------------------------

    raw, ica, suggested_components = apply_ica(
        raw
    )

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    bad_channels = identify_bad_channels(
        raw
    )

    # --------------------------------------------------------
    # Common-average reference
    # --------------------------------------------------------
    #
    # Applied ONCE here.
    # The subsequent epoching script should NOT apply another
    # average reference.
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print("COMMON-AVERAGE REFERENCE")
    print(
        "========================================"
    )

    raw.set_eeg_reference(
        "average",
        projection=False
    )

    print(
        "Applied common-average reference."
    )

    # --------------------------------------------------------
    # Final QC
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print("FINAL PREPROCESSING QC")
    print(
        "========================================"
    )

    print(
        f"Final EEG channels: "
        f"{len(raw.ch_names)}"
    )

    print(
        f"Final recording duration: "
        f"{raw.times[-1]:.2f} s"
    )

    print(
        f"Bad channels interpolated: "
        f"{len(bad_channels)}"
    )

    print(
        f"ICA components removed: "
        f"{len(ica.exclude)}"
    )

    qc = {
        "File": os.path.basename(file_path),
        "Sampling_Frequency": raw.info["sfreq"],
        "Duration_sec": raw.times[-1],
        "Initial_Channels": len(raw.ch_names),
        "Bad_Channels": bad_channels,
        "N_Bad_Channels": len(bad_channels),
        "ICA_Components": int(ica.n_components_),
        "ICA_Removed": ica.exclude.copy(),
        "ICA_Candidates": suggested_components.copy()
    }

    return raw, qc


# ============================================================
# SAVE PREPROCESSED DATA
# ============================================================

def save_preprocessed_data(
    raw,
    qc,
    fif_dir,
    qc_dir,
    base_name
):
    """
    Save continuous preprocessed EEG and QC information.
    """

    os.makedirs(
        fif_dir,
        exist_ok=True
    )

    os.makedirs(
        qc_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save continuous FIF
    # --------------------------------------------------------

    fif_path = os.path.join(
        fif_dir,
        f"{base_name}_clean_raw.fif"
    )

    raw.save(
        fif_path,
        overwrite=True
    )

    print(
        f"\nSaved preprocessed EEG:"
        f"\n{fif_path}"
    )

    # --------------------------------------------------------
    # Save QC information
    # --------------------------------------------------------

    qc_path = os.path.join(
        qc_dir,
        f"{base_name}_QC.txt"
    )

    with open(
        qc_path,
        "w"
    ) as f:

        f.write(
            "EEG PREPROCESSING QC\n"
        )

        f.write(
            "====================\n\n"
        )

        f.write(
            f"File: {qc['File']}\n"
        )

        f.write(
            f"Sampling frequency: "
            f"{qc['Sampling_Frequency']} Hz\n"
        )

        f.write(
            f"Duration: "
            f"{qc['Duration_sec']:.3f} s\n"
        )

        f.write(
            f"Final channels: "
            f"{qc['Initial_Channels']}\n"
        )

        f.write(
            f"Number of bad channels: "
            f"{qc['N_Bad_Channels']}\n"
        )

        f.write(
            "Bad channels:\n"
        )

        for ch in qc["Bad_Channels"]:
            f.write(
                f"  {ch}\n"
            )

        f.write(
            "\nICA components estimated: "
            f"{qc['ICA_Components']}\n"
        )

        f.write(
            "ICA candidate components:\n"
        )

        f.write(
            f"  {list(qc['ICA_Candidates'])}\n"
        )

        f.write(
            "ICA components removed:\n"
        )

        f.write(
            f"  {list(qc['ICA_Removed'])}\n"
        )

    print(
        f"Saved QC information:"
        f"\n{qc_path}"
    )


# ============================================================
# PROCESS MULTIPLE FILES
# ============================================================

def process_multiple_files(
    input_dir,
    output_dir
):

    fif_dir = os.path.join(
        output_dir,
        "fif_files"
    )

    qc_dir = os.path.join(
        output_dir,
        "QC"
    )

    os.makedirs(
        fif_dir,
        exist_ok=True
    )

    os.makedirs(
        qc_dir,
        exist_ok=True
    )

    mff_files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(".mff")
        ]
    )

    print(
        f"\nFound {len(mff_files)} MFF files."
    )

    for file_name in mff_files:

        file_path = os.path.join(
            input_dir,
            file_name
        )

        base_name = os.path.splitext(
            file_name
        )[0]

        print(
            "\n\n========================================"
        )
        print(
            f"PROCESSING: {file_name}"
        )
        print(
            "========================================"
        )

        try:

            raw, qc = load_and_preprocess_eeg(
                file_path
            )

            save_preprocessed_data(
                raw=raw,
                qc=qc,
                fif_dir=fif_dir,
                qc_dir=qc_dir,
                base_name=base_name
            )

            print(
                f"\n✓ Completed: {file_name}"
            )

        except Exception as e:

            print(
                f"\n✗ ERROR processing "
                f"{file_name}:"
            )

            print(
                repr(e)
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_dir = "data/raw"
    output_dir = "data/processed"

    process_multiple_files(
        input_dir=input_dir,
        output_dir=output_dir
    )
