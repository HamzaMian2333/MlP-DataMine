
"""
FIOT Highway2 - Optimized MLP Training

Creates three graphs:
1. Validation Accuracy and Macro F1
2. Per-Class Validation F1
3. Test Accuracy Across Epochs

Uses cached data for faster training.
"""

import csv
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from my_mlp import MLP
from radiomana.radiomana.datasets import HighwayDataModule


# ==========================================
# 1. CONFIGURATION
# ==========================================

BATCH_SIZE = 128
NUM_WORKERS = 0
MAX_EPOCHS = 40
LEARNING_RATE = 0.001

INPUT_SIZE = 2048
NUM_CLASSES = 9

OUTPUT_DIR = Path("mlp_graph_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "all_metrics.csv"

CLASS_NAMES = [
    "None (background a)",
    "None (background b)",
    "None (background c)",
    "None (background d)",
    "Chirp, high distance",
    "Chirp, medium distance",
    "Chirp, small distance",
    "Cigarette lighter 1",
    "Cigarette lighter 2",
]


# ==========================================
# 2. CACHE PREPROCESSED DATA
# ==========================================

def cache_dataset(dataset, name):

    print(f"\nCaching {name} dataset...", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    features = torch.empty(
        (len(dataset), INPUT_SIZE),
        dtype=torch.float32
    )

    labels = torch.empty(
        len(dataset),
        dtype=torch.long
    )

    offset = 0

    for batch_idx, (x, y) in enumerate(loader):

        # Original: (batch, 512, 243)
        x = x.float()

        # Add channel dimension
        x = x.unsqueeze(1)

        # Resize to (batch, 1, 32, 64)
        x = F.adaptive_avg_pool2d(
            x,
            (32, 64)
        )

        # Flatten to 2048 features
        x = x.flatten(start_dim=1)

        count = x.size(0)

        features[offset:offset + count] = x
        labels[offset:offset + count] = y

        offset += count

        if (batch_idx + 1) % 100 == 0:
            print(
                f"{name}: {offset}/{len(dataset)} cached",
                flush=True
            )

    print(
        f"{name} caching complete: {len(dataset)} samples",
        flush=True
    )

    return TensorDataset(features, labels)


# ==========================================
# 3. DATA MODULE
# ==========================================

class FastMLPDataModule(HighwayDataModule):

    def __init__(self):
        super().__init__(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=False
        )

        self.cache_ready = False

    def setup(self, stage=None):

        if self.cache_ready:
            return

        # Load original dataset splits
        super().setup(stage=None)

        # Preprocess once and cache
        self.data_train = cache_dataset(
            self.data_train,
            "Training"
        )

        self.data_val = cache_dataset(
            self.data_val,
            "Validation"
        )

        self.data_test = cache_dataset(
            self.data_test,
            "Testing"
        )

        self.cache_ready = True

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )


# ==========================================
# 4. MLP WITH ALL METRICS
# ==========================================

class MLPWithMetrics(MLP):

    def __init__(self):
        super().__init__(
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES
        )

        self.val_predictions = []
        self.val_targets = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=LEARNING_RATE
        )

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    def on_validation_epoch_start(self):

        self.val_predictions = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)

        loss = self.criterion(logits, y)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        self.val_predictions.append(
            predictions.detach().cpu()
        )

        self.val_targets.append(
            y.detach().cpu()
        )

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x.size(0)
        )

        return loss

    # --------------------------------------
    # TEST ACCURACY EVALUATION
    # --------------------------------------

    def calculate_test_accuracy(self):

        test_loader = (
            self.trainer.datamodule.test_dataloader()
        )

        correct = 0
        total = 0

        was_training = self.training
        self.eval()

        with torch.inference_mode():

            for x, y in test_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self(x)

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                correct += (
                    predictions == y
                ).sum().item()

                total += y.size(0)

        if was_training:
            self.train()

        return correct / total

    # --------------------------------------
    # END OF VALIDATION EPOCH
    # --------------------------------------

    def on_validation_epoch_end(self):

        # Ignore preliminary sanity checking
        if self.trainer.sanity_checking:
            return

        predictions = torch.cat(
            self.val_predictions
        ).numpy()

        targets = torch.cat(
            self.val_targets
        ).numpy()

        # Validation accuracy
        val_accuracy = accuracy_score(
            targets,
            predictions
        )

        # F1 for all nine classes
        per_class_f1 = f1_score(
            targets,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0
        )

        # Validation macro F1
        macro_f1 = float(
            np.mean(per_class_f1)
        )

        # Test accuracy at current epoch
        test_accuracy = (
            self.calculate_test_accuracy()
        )

        # Log metrics
        self.log(
            "val_acc",
            float(val_accuracy),
            prog_bar=True
        )

        self.log(
            "val_macro_f1",
            macro_f1,
            prog_bar=True
        )

        self.log(
            "epoch_test_acc",
            float(test_accuracy),
            prog_bar=False
        )

        for i, score in enumerate(per_class_f1):
            self.log(
                f"val_f1_class_{i}",
                float(score)
            )

        # Save every epoch to CSV
        with open(
            CSV_PATH,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                self.current_epoch + 1,
                float(val_accuracy),
                macro_f1,
                float(test_accuracy),
                *per_class_f1.tolist()
            ])

        print(
            f"\nEpoch {self.current_epoch + 1}"
            f" | Val Acc: {val_accuracy * 100:.2f}%"
            f" | Val Macro F1: {macro_f1 * 100:.2f}%"
            f" | Test Acc: {test_accuracy * 100:.2f}%",
            flush=True
        )


# ==========================================
# 5. GENERATE ALL THREE GRAPHS
# ==========================================

def plot_results():

    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    epochs = df["epoch"].to_numpy()

    val_accuracy = (
        df["val_acc"].to_numpy() * 100
    )

    val_macro_f1 = (
        df["val_macro_f1"].to_numpy() * 100
    )

    test_accuracy = (
        df["test_acc"].to_numpy() * 100
    )

    # Best epoch based on validation macro F1
    best_index = df["val_macro_f1"].idxmax()

    best_epoch = int(
        df.loc[best_index, "epoch"]
    )

    print(f"\nBest epoch: {best_epoch}")

    # ======================================
    # GRAPH 1: VAL ACCURACY AND MACRO F1
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Val Accuracy (%)"
    )

    plt.plot(
        epochs,
        val_macro_f1,
        marker="s",
        markersize=3,
        linestyle="--",
        linewidth=2,
        label="Val Macro F1 (%)"
    )

    plt.axvline(
        best_epoch,
        color="red",
        linestyle=":",
        label=f"Best Epoch ({best_epoch})"
    )

    plt.title(
        "Highway2 MLP: Validation Accuracy & Macro F1 across Epochs"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")

    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_accuracy_f1.png",
        dpi=300
    )

    plt.close()

    # ======================================
    # GRAPH 2: PER-CLASS VALIDATION F1
    # ======================================

    plt.figure(figsize=(12, 6))

    for i, class_name in enumerate(CLASS_NAMES):

        plt.plot(
            epochs,
            df[f"f1_class_{i}"],
            linewidth=1.8,
            label=class_name
        )

    plt.axvline(
        best_epoch,
        color="black",
        linestyle=":",
        label=f"Best Epoch ({best_epoch})"
    )

    plt.title(
        "Highway2 MLP: Per-Class Validation F1 Performance across Epochs"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Score")

    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "per_class_validation_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # GRAPH 3: TEST ACCURACY
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        epochs,
        test_accuracy,
        marker="o",
        markersize=4,
        linewidth=2,
        color="green",
        label="Test Accuracy (%)"
    )

    plt.axvline(
        best_epoch,
        color="red",
        linestyle=":",
        label=f"Best Validation Epoch ({best_epoch})"
    )

    plt.title(
        "Highway2 MLP: Test Accuracy across Epochs"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "test_accuracy_across_epochs.png",
        dpi=300
    )

    plt.close()

    # ======================================
    # FINAL RESULTS
    # ======================================

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)

    print(f"Total epochs: {len(df)}")
    print(f"Best validation epoch: {best_epoch}")

    print(
        f"Best validation accuracy: "
        f"{df['val_acc'].max() * 100:.2f}%"
    )

    print(
        f"Best validation macro F1: "
        f"{df['val_macro_f1'].max() * 100:.2f}%"
    )

    print(
        f"Test accuracy at best validation epoch: "
        f"{df.loc[best_index, 'test_acc'] * 100:.2f}%"
    )

    print(
        f"Final epoch test accuracy: "
        f"{test_accuracy[-1]:.2f}%"
    )

    print("\nAll three graphs saved successfully!")
    print("Location:", OUTPUT_DIR)

    print("=" * 50)


# ==========================================
# 6. MAIN TRAINING SCRIPT
# ==========================================

if __name__ == "__main__":

    # Reproducibility
    L.seed_everything(42)

    # Initialize metric CSV
    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "val_acc",
            "val_macro_f1",
            "test_acc",
            *[
                f"f1_class_{i}"
                for i in range(NUM_CLASSES)
            ]
        ])

    # Initialize cached dataset
    data = FastMLPDataModule()

    # Initialize MLP
    model = MLPWithMetrics()

    # Save best model using validation F1
    checkpoint = (
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=OUTPUT_DIR / "checkpoints",
            filename="best-epoch{epoch:02d}",
            monitor="val_macro_f1",
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=False
        )
    )

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    # Train the MLP
    trainer.fit(
        model,
        datamodule=data
    )

    # Final test using the best validation checkpoint
    print("\nTesting best saved model...")

    trainer.test(
        datamodule=data,
        ckpt_path="best"
    )

    # Create all three graphs
    plot_results()
