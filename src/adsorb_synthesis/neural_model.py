"""
Deep Neural Network for Tabular Regression with Entity Embeddings.

Architecture:
- Entity Embeddings for categorical features
- Residual MLP blocks with BatchNorm and Dropout
- Deep Ensemble for uncertainty quantification
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data with mixed features."""
    
    def __init__(
        self,
        X_numeric: np.ndarray,
        X_categorical: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        self.X_numeric = torch.tensor(X_numeric, dtype=torch.float32)
        self.X_categorical = torch.tensor(X_categorical, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
    def __len__(self):
        return len(self.X_numeric)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_numeric[idx], self.X_categorical[idx], self.y[idx]
        return self.X_numeric[idx], self.X_categorical[idx]


class ResidualBlock(nn.Module):
    """Residual MLP block with BatchNorm and Dropout."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.activation(x + self.block(x)))


class DeepTabularNet(nn.Module):
    """
    Deep Neural Network for Tabular Data.
    
    Features:
    - Entity embeddings for categorical variables
    - Stacked residual blocks
    - Skip connections
    - Output head for regression
    """
    
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: List[int],
        embedding_dims: Optional[List[int]] = None,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_numeric = n_numeric
        self.cat_cardinalities = cat_cardinalities
        
        # Embedding dimensions: rule of thumb min(50, (cardinality+1)//2)
        if embedding_dims is None:
            embedding_dims = [min(50, max(4, (card + 1) // 2)) for card in cat_cardinalities]
        self.embedding_dims = embedding_dims
        
        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim)  # +1 for unknown/padding
            for card, dim in zip(cat_cardinalities, embedding_dims)
        ])
        
        # Calculate total input dimension
        total_embed_dim = sum(embedding_dims)
        input_dim = n_numeric + total_embed_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        # Output head with uncertainty (mean and log_var)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.05)
                    
    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        # Embed categorical features
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings.append(emb_layer(x_categorical[:, i]))
        
        # Concatenate numeric and embedded features
        if len(embeddings) > 0:
            x = torch.cat([x_numeric] + embeddings, dim=1)
        else:
            x = x_numeric
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output_head(x).squeeze(-1)


class TabularPreprocessor:
    """Preprocessor for tabular data: encodes categoricals and scales numerics."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.cat_cardinalities: List[int] = []
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        
    def fit(self, X, y, categorical_cols: List[str]):
        """Fit preprocessor on training data."""
        self.categorical_cols = categorical_cols
        self.numeric_cols = [c for c in X.columns if c not in categorical_cols]
        
        # Fit label encoders for categorical features
        self.cat_cardinalities = []
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle unseen values by fitting on all unique + 'UNKNOWN'
            values = list(X[col].astype(str).unique()) + ['__UNKNOWN__']
            le.fit(values)
            self.label_encoders[col] = le
            self.cat_cardinalities.append(len(le.classes_))
        
        # Fit scaler on numeric features
        if len(self.numeric_cols) > 0:
            self.scaler.fit(X[self.numeric_cols].fillna(0))
        
        # Target normalization stats
        self.target_mean = y.mean()
        self.target_std = y.std()
        if self.target_std < 1e-8:
            self.target_std = 1.0
            
        return self
    
    def transform(self, X, y=None):
        """Transform data using fitted preprocessor."""
        # Transform numeric features
        if len(self.numeric_cols) > 0:
            X_numeric = self.scaler.transform(X[self.numeric_cols].fillna(0))
        else:
            X_numeric = np.zeros((len(X), 1))
        
        # Transform categorical features
        X_categorical = np.zeros((len(X), len(self.categorical_cols)), dtype=np.int64)
        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            # Handle unseen categories
            values = X[col].astype(str).values
            encoded = []
            for v in values:
                if v in le.classes_:
                    encoded.append(le.transform([v])[0])
                else:
                    # Map to UNKNOWN
                    encoded.append(le.transform(['__UNKNOWN__'])[0])
            X_categorical[:, i] = encoded
        
        if y is not None:
            y_normalized = (y - self.target_mean) / self.target_std
            return X_numeric, X_categorical, y_normalized.values
        
        return X_numeric, X_categorical
    
    def inverse_transform_y(self, y_pred: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to original scale."""
        return y_pred * self.target_std + self.target_mean
    
    def save(self, path: str):
        """Save preprocessor to disk."""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'cat_cardinalities': self.cat_cardinalities,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'TabularPreprocessor':
        """Load preprocessor from disk."""
        data = joblib.load(path)
        preprocessor = cls()
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.scaler = data['scaler']
        preprocessor.numeric_cols = data['numeric_cols']
        preprocessor.categorical_cols = data['categorical_cols']
        preprocessor.cat_cardinalities = data['cat_cardinalities']
        preprocessor.target_mean = data['target_mean']
        preprocessor.target_std = data['target_std']
        return preprocessor


class DeepEnsemble:
    """
    Deep Ensemble of Neural Networks for Uncertainty Quantification.
    
    Trains multiple models with different initializations and provides
    mean prediction + epistemic uncertainty (std across ensemble).
    """
    
    def __init__(
        self,
        n_models: int = 5,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 300,
        patience: int = 30,
        device: Optional[str] = None
    ):
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.models: List[DeepTabularNet] = []
        self.preprocessor: Optional[TabularPreprocessor] = None
        
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        categorical_cols: List[str] = None,
        verbose: bool = True
    ):
        """Train ensemble of models."""
        if categorical_cols is None:
            categorical_cols = []
            
        # Fit preprocessor
        self.preprocessor = TabularPreprocessor()
        self.preprocessor.fit(X_train, y_train, categorical_cols)
        
        # Transform data
        X_num_train, X_cat_train, y_train_norm = self.preprocessor.transform(X_train, y_train)
        
        if X_val is not None:
            X_num_val, X_cat_val, y_val_norm = self.preprocessor.transform(X_val, y_val)
            val_dataset = TabularDataset(X_num_val, X_cat_val, y_val_norm)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        train_dataset = TabularDataset(X_num_train, X_cat_train, y_train_norm)
        
        self.models = []
        
        for i in range(self.n_models):
            if verbose:
                print(f"  Training model {i+1}/{self.n_models}...")
            
            # Set different seed for each model
            torch.manual_seed(42 + i * 1000)
            np.random.seed(42 + i * 1000)
            
            # Create model
            model = DeepTabularNet(
                n_numeric=X_num_train.shape[1],
                cat_cardinalities=self.preprocessor.cat_cardinalities,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout
            ).to(self.device)
            
            # Optimizer with weight decay (AdamW)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2
            )
            
            # Loss function
            criterion = nn.HuberLoss(delta=1.0)
            
            # Training loop with early stopping
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                # Training
                model.train()
                train_loss = 0.0
                for X_num, X_cat, y_batch in train_loader:
                    X_num = X_num.to(self.device)
                    X_cat = X_cat.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    y_pred = model(X_num, X_cat)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                # Validation
                if X_val is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_num, X_cat, y_batch in val_loader:
                            X_num = X_num.to(self.device)
                            X_cat = X_cat.to(self.device)
                            y_batch = y_batch.to(self.device)
                            y_pred = model(X_num, X_cat)
                            val_loss += criterion(y_pred, y_batch).item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"    Early stopping at epoch {epoch+1}")
                        break
                        
                    if verbose and (epoch + 1) % 50 == 0:
                        print(f"    Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
            
            # Load best model
            if best_state is not None:
                model.load_state_dict(best_state)
            
            model.eval()
            self.models.append(model)
        
        return self
    
    def predict(self, X, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict with ensemble, optionally returning uncertainty."""
        X_num, X_cat = self.preprocessor.transform(X)
        
        X_num_t = torch.tensor(X_num, dtype=torch.float32).to(self.device)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_num_t, X_cat_t).cpu().numpy()
                # Inverse transform to original scale
                pred = self.preprocessor.inverse_transform_y(pred)
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        
        if return_std:
            std_pred = np.std(predictions, axis=0)
            return mean_pred, std_pred
        
        return mean_pred, None
    
    def save(self, output_dir: str, target_name: str):
        """Save ensemble to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        safe_target = target_name.replace('/', '_').replace(' ', '_')
        
        # Save preprocessor
        self.preprocessor.save(os.path.join(output_dir, f"preprocessor_{safe_target}.joblib"))
        
        # Save models
        model_paths = []
        for i, model in enumerate(self.models):
            path = os.path.join(output_dir, f"nn_{safe_target}_ens{i}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_numeric': model.n_numeric,
                'cat_cardinalities': model.cat_cardinalities,
                'embedding_dims': model.embedding_dims,
                'hidden_dim': self.hidden_dim,
                'n_blocks': self.n_blocks,
                'dropout': self.dropout
            }, path)
            model_paths.append(path)
        
        return model_paths
    
    @classmethod
    def load(cls, output_dir: str, target_name: str, device: Optional[str] = None) -> 'DeepEnsemble':
        """Load ensemble from disk."""
        safe_target = target_name.replace('/', '_').replace(' ', '_')
        
        ensemble = cls(device=device)
        
        # Load preprocessor
        ensemble.preprocessor = TabularPreprocessor.load(
            os.path.join(output_dir, f"preprocessor_{safe_target}.joblib")
        )
        
        # Find and load models
        i = 0
        while True:
            path = os.path.join(output_dir, f"nn_{safe_target}_ens{i}.pt")
            if not os.path.exists(path):
                break
            
            checkpoint = torch.load(path, map_location=ensemble.device)
            
            model = DeepTabularNet(
                n_numeric=checkpoint['n_numeric'],
                cat_cardinalities=checkpoint['cat_cardinalities'],
                embedding_dims=checkpoint['embedding_dims'],
                hidden_dim=checkpoint['hidden_dim'],
                n_blocks=checkpoint['n_blocks'],
                dropout=checkpoint['dropout']
            ).to(ensemble.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            ensemble.models.append(model)
            
            # Store hyperparams
            ensemble.hidden_dim = checkpoint['hidden_dim']
            ensemble.n_blocks = checkpoint['n_blocks']
            ensemble.dropout = checkpoint['dropout']
            
            i += 1
        
        ensemble.n_models = len(ensemble.models)
        return ensemble
