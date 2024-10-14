import gc

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from mne_bids import BIDSPath
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_features(modelname: str, layer: int = 0):

    with h5py.File(f"../monkey/stimuli/{modelname}/states.hdf5", "r") as f:
        datasets = list(f.keys())
        if len(datasets) == 1:
            key = datasets[0]
            embeddings = f[key][...]
        elif "layer" in datasets[0]:
            embeddings = f[f"layer-{layer}"][...]
        else:
            raise ValueError("Unknown states")

    df = pd.read_csv(
        f"../monkey/stimuli/{modelname}/transcript.tsv", sep="\t", index_col=0
    )

    # Reduce tokens to original words by averaging
    aligned_embeddings = []
    for _, group in df.groupby("word_idx"):
        indices = group.index.to_numpy()
        average_emb = embeddings[indices].mean(0)
        aligned_embeddings.append(average_emb)
    aligned_embeddings = np.stack(aligned_embeddings)

    df = df.groupby("word_idx").agg(dict(word="first", start="first", end="last"))

    # Remove words with no onsets
    good_mask = df["start"].notna().to_numpy()
    aligned_embeddings = aligned_embeddings[good_mask]
    df.dropna(subset=["start"], inplace=True)

    return df, aligned_embeddings


def main(
    bids_root: str,
    band: str,
    modelname: str,
    layer: int,
    n_folds: int,
    tmin: float,
    tmax: float,
    out_dir: str,
):

    if use_gpu := torch.cuda.is_available():
        print("Using GPU")
        set_backend("torch_cuda")

    edf_path = BIDSPath(
        root=bids_root, datatype="ieeg", description=band, extension=".fif"
    )
    edf_paths = edf_path.match()

    df, embeddings = get_features(modelname, layer)

    for edf_path in edf_paths:

        raw = mne.io.read_raw_fif(edf_path)
        raw = raw.pick("ecog", exclude="bads")
        sfreq = int(raw.info["sfreq"])

        # Epoch raw data
        print("Epoching data")
        events = np.zeros((len(df), 3), dtype=int)
        events[:, 0] = (df.start * sfreq).astype(int)
        epochs = mne.Epochs(
            raw,
            events,
            tmin=tmin,
            tmax=tmax,
            proj=None,
            baseline=None,
            event_id=None,
            preload=True,
            event_repeated="merge",
        )
        epochs = epochs.resample(
            sfreq=32, npad="auto", method="fft", window="hamming", n_jobs=4  # NOTE
        )

        # Prepare data for modeling
        epochs_data = epochs.get_data(copy=True)
        epochs_data = epochs_data.reshape(len(epochs), -1)
        epochs_shape = epochs._data.shape[1:]

        averaged_embeddings = embeddings[epochs.selection]
        X = averaged_embeddings.astype(np.float32)
        Y = epochs_data.astype(np.float32)

        # Prepare model
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(
                alphas=np.logspace(1, 10, 10),
                fit_intercept=True,
                cv=KFold(n_splits=5, shuffle=False),
            ),
        )

        # Train model
        scores = []
        kfold = KFold(n_folds, shuffle=False)
        print("Fitting models")
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            scaler = StandardScaler()
            Y_train = scaler.fit_transform(Y_train)
            Y_test = scaler.transform(Y_test)

            model.fit(X_train, Y_train)
            Y_preds = model.predict(X_test)
            corr = correlation_score(Y_test, Y_preds).reshape(epochs_shape)

            if use_gpu:
                corr = corr.numpy(force=True)  # for torch
                # corr = corr.asnumpy()  # for cupy
            scores.append(corr)
        scores = np.stack(scores)

        # Save results
        out_path = edf_path.copy()
        desc = modelname
        if layer is not None:
            desc = f"{modelname}-l{layer}"
        out_path.update(
            root=out_dir,
            datatype="encoding",
            description=desc.replace("-", ".").replace("_", "."),
            suffix="result",
            extension=".h5",
            check=False,
        )
        out_path.mkdir()
        lags = np.arange(tmin * sfreq, tmax * sfreq + 1, 16)
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="scores", data=scores)
            f.create_dataset(name="lags", data=lags)

        del raw, epochs
        gc.collect()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--bids-root", type=str, default="../monkey/derivatives/ecogprep"
    )
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("-m", "--modelname", type=str, default="gpt2-xl")
    parser.add_argument("-l", "--layer", type=int, default=None)
    parser.add_argument("-k", "--n-folds", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-2)
    parser.add_argument("--tmax", type=float, default=2)
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    _args = parser.parse_args()
    main(**vars(_args))
