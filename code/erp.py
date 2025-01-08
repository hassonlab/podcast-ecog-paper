import gc
from functools import partial

import h5py
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.stats import zscore


def main(bids_root: str, band: str, out_dir: str, tmin: float, tmax: float):

    edf_path = BIDSPath(
        root=bids_root, datatype="ieeg", description=band, extension=".fif"
    )
    edf_paths = edf_path.match()

    df = pd.read_csv("../monkey/stimuli/podcast_transcript.csv")
    df.dropna(subset=["start"], inplace=True)
    df.sort_values("start", inplace=True)

    events = np.zeros((len(df), 3))
    events[:, 0] = df.start

    for edf_path in edf_paths:

        raw = mne.io.read_raw_fif(edf_path)
        sfreq = int(raw.info["sfreq"])

        raw.load_data()
        zscore_func = partial(zscore, axis=1)
        raw = raw.apply_function(zscore_func, channel_wise=False)

        sub_events = (events.copy() * sfreq).astype(int)

        epochs = mne.Epochs(
            raw,
            sub_events,
            tmin=tmin,
            tmax=tmax,
            proj=None,
            baseline=None,
            event_id=None,
            preload=True,
            event_repeated="merge",
        )
        evoked = epochs.average()

        lags = np.arange(tmin * sfreq, tmax * sfreq + 1)

        out_path = edf_path.copy()
        out_path.update(
            root=out_dir,
            datatype="erp",
            suffix="result",
            extension=".h5",
            check=False,
        )
        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="evoked", data=evoked.data)
            f.create_dataset(name="lags", data=lags)

        del raw, epochs, evoked
        gc.collect()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--bids-root", type=str, default="../monkey/derivatives/ecogprep"
    )
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("--tmin", type=float, default=-2)
    parser.add_argument("--tmax", type=float, default=2)
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    _args = parser.parse_args()
    main(**vars(_args))
