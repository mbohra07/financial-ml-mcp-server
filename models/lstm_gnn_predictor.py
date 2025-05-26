"""
LSTM-GNN Hybrid Model for Stock Price Prediction
Combines LSTM for temporal patterns with GNN for cross-asset correlations
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    import sys
    print("Warning: PyTorch Geometric not available. LSTM-GNN will use simplified architecture.", file=sys.stderr)
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import joblib
import os
from config.settings import settings, ModelConfigs

logger = logging.getLogger(__name__)


class LSTMGNNPredictor(nn.Module):
    """Hybrid LSTM-GNN model for multi-asset price prediction"""

    def __init__(
        self,
        num_assets: int,
        lstm_input_size: int = 5,  # OHLCV
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        gnn_hidden_size: int = 64,
        gnn_num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        gnn_type: str = "GCN"
    ):
        super(LSTMGNNPredictor, self).__init__()

        self.num_assets = num_assets
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # GNN layers for cross-asset relationships
        self.gnn_layers = nn.ModuleList()
        if gnn_type == "GCN":
            self.gnn_layers.append(GCNConv(lstm_hidden_size, gnn_hidden_size))
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(GCNConv(gnn_hidden_size, gnn_hidden_size))
        elif gnn_type == "GAT":
            self.gnn_layers.append(GATConv(lstm_hidden_size, gnn_hidden_size))
            for _ in range(gnn_num_layers - 1):
                self.gnn_layers.append(GATConv(gnn_hidden_size, gnn_hidden_size))

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gnn_hidden_size, gnn_hidden_size // 2)
        self.fc2 = nn.Linear(gnn_hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass
        x: [batch_size, num_assets, sequence_length, features]
        edge_index: [2, num_edges] - graph connectivity
        edge_weight: [num_edges] - edge weights (correlations)
        """
        batch_size, num_assets, seq_len, features = x.shape

        # Reshape for LSTM: [batch_size * num_assets, seq_len, features]
        x_lstm = x.view(batch_size * num_assets, seq_len, features)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_lstm)

        # Take the last output from LSTM
        lstm_features = lstm_out[:, -1, :]  # [batch_size * num_assets, lstm_hidden_size]

        # Reshape for GNN: [batch_size * num_assets, lstm_hidden_size]
        node_features = lstm_features

        # GNN forward pass
        for i, gnn_layer in enumerate(self.gnn_layers):
            if edge_weight is not None:
                node_features = gnn_layer(node_features, edge_index, edge_weight)
            else:
                node_features = gnn_layer(node_features, edge_index)

            if i < len(self.gnn_layers) - 1:
                node_features = self.relu(node_features)
                node_features = self.dropout(node_features)

        # Final prediction layers
        output = self.relu(self.fc1(node_features))
        output = self.dropout(output)
        output = self.fc2(output)

        # Reshape back to [batch_size, num_assets, output_size]
        output = output.view(batch_size, num_assets, -1)

        return output


class LSTMGNNTrainer:
    """Trainer class for LSTM-GNN model with hyperparameter optimization"""

    def __init__(self, model_complexity: str = "auto"):
        self.model_complexity = model_complexity
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None

    def prepare_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        sequence_length: int = 60,
        correlation_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for LSTM-GNN training"""

        # Align all dataframes by date
        symbols = list(price_data.keys())
        aligned_data = {}

        # Find common date range
        common_dates = None
        for symbol, df in price_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        common_dates = sorted(list(common_dates))

        # Align data and extract features
        features_list = []
        for symbol in symbols:
            df = price_data[symbol].loc[common_dates]
            # Use OHLCV features
            features = df[['open', 'high', 'low', 'close', 'volume']].values
            features_list.append(features)

        # Stack features: [num_assets, time_steps, features]
        features_array = np.stack(features_list, axis=0)

        # Normalize features
        original_shape = features_array.shape
        features_flat = features_array.reshape(-1, features_array.shape[-1])
        features_normalized = self.scaler.fit_transform(features_flat)
        features_array = features_normalized.reshape(original_shape)

        # Create sequences
        X, y = self._create_sequences(features_array, sequence_length)

        # Create correlation-based graph
        edge_index, edge_weight = self._create_correlation_graph(
            price_data, correlation_threshold
        )

        return (
            torch.FloatTensor(X),
            torch.FloatTensor(y),
            torch.LongTensor(edge_index),
            torch.FloatTensor(edge_weight)
        )

    def _create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        num_assets, time_steps, features = data.shape

        X, y = [], []
        for i in range(sequence_length, time_steps):
            # Input sequence: [num_assets, sequence_length, features]
            X.append(data[:, i-sequence_length:i, :])
            # Target: next day's close price (index 3)
            y.append(data[:, i, 3])  # close price

        return np.array(X), np.array(y)

    def _create_correlation_graph(
        self,
        price_data: Dict[str, pd.DataFrame],
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create graph based on price correlations"""
        symbols = list(price_data.keys())
        num_assets = len(symbols)

        # Calculate correlation matrix
        returns_data = {}
        for symbol, df in price_data.items():
            returns_data[symbol] = df['close'].pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr().values

        # Create edges based on correlation threshold
        edge_index = []
        edge_weight = []

        for i in range(num_assets):
            for j in range(num_assets):
                if i != j and abs(correlation_matrix[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_weight.append(abs(correlation_matrix[i, j]))

        if not edge_index:
            # If no correlations above threshold, create self-loops
            edge_index = [[i, i] for i in range(num_assets)]
            edge_weight = [1.0] * num_assets

        return np.array(edge_index).T, np.array(edge_weight)

    def optimize_hyperparameters(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            # Suggest hyperparameters
            lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [64, 128, 256, 512])
            lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
            gnn_hidden_size = trial.suggest_categorical('gnn_hidden_size', [32, 64, 128])
            gnn_num_layers = trial.suggest_int('gnn_num_layers', 1, 4)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

            # Create model
            num_assets = X.shape[1]
            model = LSTMGNNPredictor(
                num_assets=num_assets,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                gnn_hidden_size=gnn_hidden_size,
                gnn_num_layers=gnn_num_layers,
                dropout=dropout
            ).to(self.device)

            # Train and evaluate
            val_loss = self._train_and_evaluate(
                model, X, y, edge_index, edge_weight,
                learning_rate, batch_size, epochs=50
            )

            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        return study.best_params

    def _train_and_evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        learning_rate: float,
        batch_size: int,
        epochs: int = 100
    ) -> float:
        """Train model and return validation loss"""

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)

        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X, edge_index, edge_weight)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, edge_index, edge_weight)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()

        return val_loss


# Global predictor instance
lstm_gnn_predictor = LSTMGNNTrainer()
