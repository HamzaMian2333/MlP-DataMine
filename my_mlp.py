from radiomana.radiomana.models import ModelBaseClass


import torch
import torch.nn as nn

from radiomana.radiomana.models import ModelBaseClass


class MLP(ModelBaseClass):
    """
    Multi-Layer Perceptron for classifying RF interference.
    """

    def __init__(self, input_size=2048, num_classes=9):
        # Initialize the existing base model
        super().__init__(num_classes=num_classes)

        # Define the neural network architecture
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 512),
            nn.ReLU(),

            # Second hidden layer
            nn.Linear(512, 256),
            nn.ReLU(),

            # Third hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),

            # Output layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Convert each sample into a one-dimensional vector
        x = torch.flatten(x, start_dim=1)

        # Pass input through all network layers
        return self.network(x)

    def configure_optimizers(self):
        # Adam optimizer adjusts network weights during training
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001
        )
        return optimizer


if __name__ == "__main__":
    model = MLP(input_size=2048, num_classes=9)

    # Create 32 example samples
    x = torch.randn(32, 2048)

    # Run the samples through the MLP
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    predictions = torch.argmax(output, dim=1)
    print("Predicted classes:", predictions)