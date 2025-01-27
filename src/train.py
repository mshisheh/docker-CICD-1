import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleFFNN
import torch.nn as nn
import os

epochs = int(os.getenv("EPOCHS", 10))  # Default to 10 if not set
lr = float(os.getenv("LR", 0.01))      # Default to 0.01 if not set



# Dummy data for regression (y = 2x + noise)
torch.manual_seed(0)
X = torch.arange(1000, dtype=torch.float32) / 1000  # 1000 samples, 1 feature
X = X.reshape(-1, 1)
y = 2 * X + torch.randn(1000,1) * 0.05

# Create a simple dataset and loader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss, and optimizer
model = SimpleFFNN(input_dim=1, hidden_dim=10, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
print("Training loop")
for epoch in range(epochs):  # 100 epochs
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
