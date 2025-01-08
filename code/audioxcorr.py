import h5py
import mne
import numpy as np
from mne_bids import BIDSPath
from scipy import signal
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags


def xcorr(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "full",
    method: str = "fft",
    norm: bool = True,
    maxlags: int = None,
):
    """General function to compute cross correlation using scipy"""

    # Center
    x = x - x.mean()
    y = y - y.mean()

    # Correlate
    corr = correlate(x, y, mode=mode, method=method)
    lags = correlation_lags(x.size, y.size, mode=mode)

    if norm:
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is not None:
        middle = (lags == 0).nonzero()[0].item()
        lags = np.arange(-maxlags, maxlags + 1)
        corr = corr[middle - maxlags : middle + maxlags + 1]

    return corr, lags


def preprocess_highqa(x, fs, to_fs, lowcut=200, highcut=5000):

    # See https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype="band")
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    assert x.ndim == 1
    # x = x[:fs * round(len(x) / fs)]  # trim to nearest second

    # Step 1. Bandpass the high quality audio
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=5)

    # Step 2. Downsample to same freq as clinical system
    # Number of new samples is N = n * (to_fs / fs)
    y = signal.resample(y, num=round(x.size / fs * to_fs))

    # Step 3. Take audio envelope
    envelope = np.abs(signal.hilbert(y - y.mean()))

    return envelope


def get_audio():
    sfreq = 512
    audio_path = "../monkey/stimuli/podcast.wav"
    highfs, highqa = wavfile.read(audio_path)
    if highqa.ndim > 1:
        highqa = highqa[:, 0]  # take first channel
    highenv = preprocess_highqa(highqa, highfs, sfreq)
    return highenv


def main(bids_root: str, band: str, out_dir: str, lag: int):

    edf_path = BIDSPath(
        root=bids_root, datatype="ieeg", description=band, extension=".fif"
    )
    edf_paths = edf_path.match()

    highenv = get_audio()

    for edf_path in edf_paths:
        out_path = edf_path.copy()
        out_path.update(
            root=out_dir,
            datatype="audioxcorr",
            suffix="result",
            extension=".h5",
            check=False,
        )

        raw = mne.io.read_raw_fif(edf_path)
        data = raw.get_data()
        maxlag = int(lag * raw.info["sfreq"])

        all_corrs = []
        for i in range(len(data)):
            corrs, lags = xcorr(data[i], highenv, maxlags=maxlag)
            all_corrs.append(corrs)
        all_corrs = np.stack(all_corrs)

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="corrs", data=all_corrs)
            f.create_dataset(name="lags", data=lags)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--bids-root", type=str, default="../monkey/derivatives/ecogprep"
    )
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    parser.add_argument("-l", "--lag", type=int, default=10)
    _args = parser.parse_args()
    main(**vars(_args))
