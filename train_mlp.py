
import torch
import torch.nn.functional as F
import lightning as L

from torch.utils.data import DataLoader, Dataset
from my_mlp import MLP
from radiomana.radiomana.datasets import HighwayDataModule


class PreprocessedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]

        # Original shape: (512, 243)
        x = x.unsqueeze(0).unsqueeze(0)

        # Convert to 32 x 64 = 2048 features
        x = F.adaptive_avg_pool2d(x, (32, 64))

        # Flatten into a vector
        x = x.reshape(2048)

        return x, y


class MLPDataModule(HighwayDataModule):

    def setup(self, stage=None):
        # Load and split the original dataset
        super().setup(stage)

        # Apply preprocessing to each split
        self.data_train = PreprocessedDataset(self.data_train)
        self.data_val = PreprocessedDataset(self.data_val)
        self.data_test = PreprocessedDataset(self.data_test)


if __name__ == "__main__":
    L.seed_everything(42)

    # Load the dataset
    data = MLPDataModule(
        batch_size=32,
        num_workers=2
    )

    # Initialize our neural network
    model = MLP(
        input_size=2048,
        num_classes=9
    )

    # Configure training
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    # Train the MLP
    trainer.fit(model, datamodule=data)

    # Evaluate the trained model
    trainer.test(model, datamodule=data)
