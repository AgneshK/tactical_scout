import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import PowerTransformer, RobustScaler
import joblib
import os

# Set device to utilize the RTX 3050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TacticalAutoencoder(nn.Module):
    def __init__(self, num_features, latent_dim=8):
        super(TacticalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, num_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

if __name__ == "__main__":
    print(f"Training on: {device}")
    
    # 1. Load Data
    df = pd.read_csv('../data/players_data-2024_2025.csv')
    df = df[~df['Pos'].str.contains('GK', na=False)]
    df = df[df['Min'] > 500].reset_index(drop=True)

    # Convert cumulative stats to Per 90 (Example columns, adjust based on your CSV)
    cumulative_cols = ['Goals', 'Assists'] 
    for col in cumulative_cols:
        if col in df.columns:
            df[col] = (df[col] / df['Min']) * 90

    # Separate info and stats
    # Save metadata + key display stats so the agent can show reasoning metrics
    _meta = ['Player', 'Pos', 'Squad', 'Comp']
    _display = ['Age', 'Nation', 'MP', 'Min', 'Goals', 'Assists', 'xG', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'KP']
    player_info = df[_meta + [c for c in _display if c in df.columns]]
    tactical_stats = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Min'], errors='ignore').fillna(0)

    # 2. Preprocessing
    pt = PowerTransformer(method='yeo-johnson')
    transformed_stats = pt.fit_transform(tactical_stats)
    
    scaler = RobustScaler()
    scaled_stats = scaler.fit_transform(transformed_stats)

    # 3. Train Model
    tensor_x = torch.Tensor(scaled_stats).to(device)
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_features = tensor_x.shape[1]
    model = TacticalAutoencoder(num_features=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            stats = batch[0]
            optimizer.zero_grad()
            encoded, reconstructed = model(stats)
            loss = criterion(reconstructed, stats)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Save Artifacts for the API
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(tensor_x).cpu()

    os.makedirs('artifacts', exist_ok=True)
    torch.save(embeddings, 'artifacts/embeddings.pt')
    joblib.dump(player_info, 'artifacts/player_info.pkl')
    print("Training complete. Artifacts saved in /artifacts.")