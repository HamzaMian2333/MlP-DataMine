"""Train the local MLP on features produced by preprocess.py."""

import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlp import MLP

FEATURES_PATH = "features/highway2_mlp.npz"
EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
CLASS_LABELS = [
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


def evaluate(model, X, y, device, criterion=None):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        preds = logits.argmax(dim=1).cpu()
        loss = criterion(logits, y.to(device)).item() if criterion else None
    return preds.numpy(), (preds == y).float().mean().item(), f1_score(y, preds, average="macro"), loss


def main():
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(FEATURES_PATH)
    X_train, y_train = torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"]).long()
    X_val, y_val = torch.from_numpy(data["X_val"]), torch.from_numpy(data["y_val"]).long()
    X_test, y_test = torch.from_numpy(data["X_test"]), torch.from_numpy(data["y_test"]).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # inverse-frequency class weights to counter the background-class imbalance
    counts = torch.bincount(y_train, minlength=len(CLASS_LABELS)).float()
    class_weights = counts.sum() / (len(counts) * counts)

    model = MLP(input_dim=X_train.shape[1], num_classes=len(CLASS_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    torch.save(model.state_dict(), "mlp_init.pt")  # untrained weights, for before/after plots

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_class_f1": []}

    best_f1, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        val_preds, val_acc, val_f1, val_loss = evaluate(model, X_val, y_val, device, criterion)
        history["train_loss"].append(total_loss / len(X_train))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_class_f1"].append(
            f1_score(y_val, val_preds, average=None, labels=range(len(CLASS_LABELS)), zero_division=0).tolist()
        )
        print(f"epoch {epoch + 1:3d}  train_loss {total_loss / len(X_train):.4f}  val_acc {val_acc:.3f}  val_f1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # rewind to best validation F1 and test
    model.load_state_dict(best_state)
    preds, test_acc, test_f1, _ = evaluate(model, X_test, y_test, device)

    confmat = confusion_matrix(y_test, preds, labels=range(len(CLASS_LABELS)))
    print(f"\nbest val f1: {best_f1:.3f}")
    print("confusion matrix:\n", confmat)
    print("test per-class accuracy:")
    for cdx, class_label in enumerate(CLASS_LABELS):
        print(f"  class {cdx} ({class_label:<22s}): {confmat[cdx, cdx] / confmat[cdx].sum():8.3%}")
    print(f"test f1: {test_f1:.3f}")
    print(f"test acc: {test_acc:8.3%}")

    torch.save(model.state_dict(), "mlp.pt")
    print("saved best model weights to mlp.pt")

    with open("mlp_history.json", "w") as handle:
        json.dump(history, handle)
    print("saved training history to mlp_history.json")


if __name__ == "__main__":
    main()
