"""
Preprocess the Highway2 dataset into flat, standardized feature vectors for an MLP.

Each (512 freq x 243 time) dB PSD is summarized over time into per-frequency
mean / std / max / min, giving 512 * 4 = 2048 features. Features are standardized
with statistics from the training split only. Absolute power is preserved
(no per-sample normalization), since it separates the chirp distance classes.

Output: features/highway2_mlp.npz with X_train, y_train, X_val, y_val, X_test, y_test,
plus feature_mean / feature_std for normalizing new samples at inference time.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import radiomana

OUTPUT_PATH = Path("features/highway2_mlp.npz")


def extract_features(psd: torch.Tensor) -> torch.Tensor:
    """(batch, freq, time) dB PSD -> (batch, 4 * freq) time-summary features."""
    return torch.cat(
        [
            psd.mean(dim=2),
            psd.std(dim=2),
            psd.amax(dim=2),
            psd.amin(dim=2),
        ],
        dim=1,
    )


def featurize(dataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    features, labels = [], []
    for psd, label in tqdm(loader, desc=f"featurizing {name}"):
        features.append(extract_features(psd))
        labels.append(label)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def main():
    # reuse the library's stratified train/val split (no augmentation, no oversampling)
    datamodule = radiomana.HighwayDataModule()
    datamodule.setup()

    X_train, y_train = featurize(datamodule.data_train, "train")
    X_val, y_val = featurize(datamodule.data_val, "val")
    X_test, y_test = featurize(datamodule.data_test, "test")

    # standardize using training statistics only
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_val=X_val.astype(np.float32),
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print(f"saved to {OUTPUT_PATH}")
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"  {split:<5s}: X {X.shape}, class counts {np.bincount(y, minlength=9).tolist()}")


if __name__ == "__main__":
    main()
