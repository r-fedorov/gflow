
# Based on https://github.com/Lulu971231/code-for-Oxygen-Producing-Catalysts-from-Martian-Meteorites/tree/main


import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Data loading and preparation
# ----------------------
data = 'dynamic.xlsx'
# Read Excel data (adjust sheet names as needed)
x = pd.read_excel(data, sheet_name='metals')
y = pd.read_excel(data, sheet_name='params')

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Standardize target parameters
scaler = StandardScaler().fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Convert pandas DataFrame to numpy arrays then torch tensors (float32)
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoaders
batch_size = 256
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# Define the network (2 hidden layers of 512 neurons)
# ----------------------


class PretrainNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(PretrainNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)


input_dim = x_train.shape[1]
model = PretrainNet(input_dim=input_dim).to(device)

# ----------------------
# Loss function and optimizer
# ----------------------
criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                 verbose=True, threshold=1e-5, min_lr=1e-6)

# ----------------------
# Training loop with Early Stopping
# ----------------------
epochs = 1000
early_stop_patience = 15
best_loss = np.inf
epochs_no_improve = 0

for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    # Optionally: compute validation loss on test set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)

    # Step scheduler on training loss (you can also use validation loss)
    scheduler.step(train_loss)

    print(
        f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    # Early Stopping check based on training loss
    if train_loss + 1e-5 < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
        # Save the best model weights (optional, here we simply keep track)
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered")
        break

#load the best saved weights
model.load_state_dict(best_model_state)

# ----------------------
# Save the model and scaler
# ----------------------
save_dir = 'pre-model'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'pretrain_model.pth')
torch.save(model.state_dict(), model_path)
joblib.dump(scaler, os.path.join(save_dir, 'norm.pkl'))
print(f"Pretrained model saved to {model_path}")
