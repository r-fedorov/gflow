
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
# Load pre-trained model and scaler from pretrain phase for feature engineering
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


# Load experimental data
data = 'experimental.xlsx'
x = pd.read_excel(data, sheet_name='metals')
y = pd.read_excel(data, sheet_name='overpotential')

# Load the pre-trained model
pre_model_dir = 'pre-model'
pre_model_path = os.path.join(pre_model_dir, 'pretrain_model.pth')

input_dim = x.shape[1]
pre_model = PretrainNet(input_dim=input_dim).to(device)
pre_model.load_state_dict(torch.load(pre_model_path, map_location=device))
pre_model.eval()

# Get features from pre-trained model
x_tensor = torch.tensor(x.values, dtype=torch.float32).to(device)
with torch.no_grad():
    pre_features = pre_model(x_tensor).cpu().numpy()

# Append the three pre-trained features as new columns
x['G_OH'] = pre_features[:, 0]
x['G_O - G_OH'] = pre_features[:, 1]
x['delta_e'] = pre_features[:, 2]

# ----------------------
# Split and scale the data (target scaling)
# ----------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2)
scaler = StandardScaler().fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Convert data to torch tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoaders
batch_size = 2
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------
# Define the retraining network (3 hidden layers of 128 neurons)
# ----------------------


class RetrainNet(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(RetrainNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# The new input dimension includes original features plus the 3 additional features
retrain_input_dim = x_train.shape[1]
model = RetrainNet(input_dim=retrain_input_dim).to(device)

# ----------------------
# Loss, optimizer, and scheduler
# ----------------------
criterion = nn.L1Loss()  # MAE
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                 verbose=True, threshold=1e-5, min_lr=1e-5)

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

    # Validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)

    # Step the scheduler
    scheduler.step(train_loss)

    print(
        f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    # Early stopping based on training loss
    if train_loss + 1e-5 < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print("Early stopping triggered")
        break

# Load best model state
model.load_state_dict(best_model_state)

# ----------------------
# Save the retrained model and scaler
# ----------------------
save_dir = 'model'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'retrain_model.pth')
torch.save(model.state_dict(), model_path)
joblib.dump(scaler, os.path.join(save_dir, 'norm.pkl'))
print(f"Retrained model saved to {model_path}")
