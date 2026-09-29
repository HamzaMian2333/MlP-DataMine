"""Presentation plots showing the MLP learning. Run after train_mlp.py."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from mlp import MLP
from train_mlp import CLASS_LABELS, FEATURES_PATH

OUT_DIR = Path("plots")

# reference palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SHORT_LABELS = ["BG a", "BG b", "BG c", "BG d", "Chirp far", "Chirp mid", "Chirp near", "Lighter 1", "Lighter 2"]
FAMILIES = {"Background": [0, 1, 2, 3], "Chirp jammer": [4, 5, 6], "Cigarette lighter": [7, 8]}
FAMILY_COLORS = {"Background": BLUE, "Chirp jammer": ORANGE, "Cigarette lighter": AQUA}

plt.rcParams.update(
    {
        "font.family": ["system-ui", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 13,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
    }
)


def load():
    data = np.load(FEATURES_PATH)
    split = lambda name: (torch.from_numpy(data[f"X_{name}"]), torch.from_numpy(data[f"y_{name}"]).long())
    with open("mlp_history.json") as handle:
        history = json.load(handle)
    models = {}
    for name, path in [("untrained", "mlp_init.pt"), ("trained", "mlp.pt")]:
        model = MLP(input_dim=data["X_train"].shape[1], num_classes=len(CLASS_LABELS))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        models[name] = model.eval()
    return split("train"), split("val"), split("test"), history, models


def prepend_epoch_zero(history, model, train, val):
    """Evaluate the untrained model so every curve starts at epoch 0."""
    counts = torch.bincount(train[1], minlength=len(CLASS_LABELS)).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts))
    with torch.no_grad():
        train_loss = criterion(model(train[0]), train[1]).item()
        val_logits = model(val[0])
    val_preds = val_logits.argmax(dim=1)
    history["train_loss"].insert(0, train_loss)
    history["val_loss"].insert(0, criterion(val_logits, val[1]).item())
    history["val_acc"].insert(0, (val_preds == val[1]).float().mean().item())
    history["val_f1"].insert(0, f1_score(val[1], val_preds, average="macro"))
    history["val_class_f1"].insert(
        0, f1_score(val[1], val_preds, average=None, labels=range(len(CLASS_LABELS)), zero_division=0).tolist()
    )
    return history


def end_label(ax, x, y, text, color, dy=0):
    ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", color=INK_2, fontsize=12)


def plot_learning_curves(history):
    epochs = np.arange(len(history["train_loss"]))
    best = int(np.argmax(history["val_f1"]))
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax_loss.plot(epochs, history["train_loss"], color=BLUE, label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], color=ORANGE, label="Validation loss")
    end_label(ax_loss, epochs[-1], history["train_loss"][-1], "Train", BLUE)
    end_label(ax_loss, epochs[-1], history["val_loss"][-1], "Validation", ORANGE)
    low = int(np.argmin(history["val_loss"]))
    ax_loss.annotate(
        f"Val loss lowest at epoch {low},\nthen rises: model grows overconfident",
        (low, history["val_loss"][low]),
        xytext=(low + 5, 1.55),
        color=INK_2,
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    ax_loss.set_title("Loss falls as the MLP trains")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Weighted cross-entropy")
    ax_loss.set_ylim(bottom=0)
    ax_loss.legend(loc="upper right")

    ax_metric.plot(epochs, history["val_f1"], color=BLUE, label="Macro F1")
    ax_metric.plot(epochs, history["val_acc"], color=ORANGE, label="Accuracy")
    end_label(ax_metric, epochs[-1], history["val_f1"][-1], f"F1 {history['val_f1'][-1]:.2f}", BLUE)
    end_label(ax_metric, epochs[-1], history["val_acc"][-1], f"Acc {history['val_acc'][-1]:.2f}", ORANGE)
    ax_metric.axvline(best, color=MUTED, lw=1, ls="--")
    ax_metric.annotate(
        f"Best val F1 {history['val_f1'][best]:.2f}\n(epoch {best}, kept for test)",
        (best, 0.05),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        color=INK_2,
        fontsize=11,
    )
    ax_metric.annotate(
        f"Untrained: F1 {history['val_f1'][0]:.2f}",
        (0, history["val_f1"][0]),
        xytext=(10, -4),
        textcoords="offset points",
        va="top",
        color=INK_2,
        fontsize=11,
    )
    ax_metric.set_title("Validation scores climb from near zero")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_ylim(0, 1)
    ax_metric.legend(loc="lower right", bbox_to_anchor=(1, 0.12))

    for ax in (ax_loss, ax_metric):
        ax.set_xlim(0, epochs[-1] + 7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_learning_curves.png", dpi=200)
    plt.close(fig)


def plot_family_f1(history):
    class_f1 = np.array(history["val_class_f1"])
    epochs = np.arange(len(class_f1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # nudge end labels apart where the background and lighter lines finish close together
    label_dy = {"Background": -9, "Chirp jammer": 0, "Cigarette lighter": 9}
    for family, classes in FAMILIES.items():
        f1 = class_f1[:, classes].mean(axis=1)
        ax.plot(epochs, f1, color=FAMILY_COLORS[family], label=family)
        end_label(ax, epochs[-1], f1[-1], f"{family} {f1[-1]:.2f}", FAMILY_COLORS[family], dy=label_dy[family])
    ax.set_title("Chirp jammers are learned fastest and best (validation F1 by signal family)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean per-class F1")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, epochs[-1] + 12)
    ax.legend(loc="lower right", bbox_to_anchor=(0.72, 0))
    ax.text(
        0,
        -0.26,
        "Validation set is small for rare classes (BG d: 5, lighters: 29 samples), so those lines are noisy.\n"
        "Background includes BG d, which is never predicted correctly on validation.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_family_f1_over_epochs.png", dpi=200)
    plt.close(fig)


def plot_confusion(models, test):
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, (name, model) in zip(axes, models.items()):
        with torch.no_grad():
            preds = model(test[0]).argmax(dim=1)
        cm = confusion_matrix(test[1], preds, labels=range(len(CLASS_LABELS)), normalize="true")
        f1 = f1_score(test[1], preds, average="macro")
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(cm):
            if value >= 0.005:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=10, color=SURFACE if value > 0.55 else INK_2)
        ax.set_xticks(range(len(SHORT_LABELS)), SHORT_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(SHORT_LABELS)), SHORT_LABELS)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{name.capitalize()} (test macro F1 {f1:.2f})")
        ax.grid(False)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(
        "Predictions concentrate on the diagonal after training (rows = % of each true class)",
        x=0.02,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_confusion_before_after.png", dpi=200)
    plt.close(fig)


def hidden_features(model, X):
    """Activations of the last hidden layer (input to the final Linear)."""
    with torch.no_grad():
        return model.layers[:-1](X).numpy()


def plot_embedding(models, test):
    y = test[1].numpy()
    family_of = np.empty_like(y)
    for fdx, classes in enumerate(FAMILIES.values()):
        family_of[np.isin(y, classes)] = fdx
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (name, model) in zip(axes, models.items()):
        emb = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(hidden_features(model, test[0]))
        # draw background first so the rarer jammer points sit on top
        for fdx, family in enumerate(FAMILIES):
            mask = family_of == fdx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=14 if fdx == 0 else 26,
                color=FAMILY_COLORS[family],
                alpha=0.35 if fdx == 0 else 0.9,
                edgecolors=SURFACE,
                linewidths=0.5,
                label=f"{family} ({mask.sum()})",
            )
        ax.set_title(f"{name.capitalize()} network")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines[:].set_visible(False)
    axes[1].legend(loc="upper right", bbox_to_anchor=(1, -0.02), ncol=3, markerscale=1.8, fontsize=12)
    fig.suptitle(
        "What the hidden layer sees: jammer types pull into tighter, separate clusters after training (t-SNE, test set)",
        x=0.02,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "4_hidden_layer_tsne.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    train, val, test, history, models = load()
    history = prepend_epoch_zero(history, models["untrained"], train, val)
    plot_learning_curves(history)
    plot_family_f1(history)
    plot_confusion(models, test)
    plot_embedding(models, test)
    print(f"saved plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
