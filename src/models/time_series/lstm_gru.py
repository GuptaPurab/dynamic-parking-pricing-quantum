"""
LSTM and GRU models for capturing temporal demand patterns in parking data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import joblib


class ParkingLSTM:
    """
    LSTM model for parking occupancy temporal pattern recognition
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LSTM model
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 24,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(
                self.config['hidden_size'],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.config['dropout']
            )
        ])
        
        # Add additional LSTM layers
        for _ in range(self.config['num_layers'] - 2):
            model.add(layers.LSTM(
                self.config['hidden_size'],
                return_sequences=True,
                dropout=self.config['dropout']
            ))
            
        # Final LSTM layer
        if self.config['num_layers'] > 1:
            model.add(layers.LSTM(
                self.config['hidden_size'],
                return_sequences=False,
                dropout=self.config['dropout']
            ))
        
        # Output layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.config['dropout']))
        model.add(layers.Dense(1, activation='linear'))  # Regression output
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Feature data
            target: Target values
            
        Returns:
            X_sequences, y_sequences
        """
        sequence_length = self.config['sequence_length']
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(data)):
            X_sequences.append(data[i-sequence_length:i])
            y_sequences.append(target[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history and metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor='val_loss' if validation_data else 'loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_seq)
        metrics = {
            'train_mae': mean_absolute_error(y_train_seq, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
            'train_r2': r2_score(y_train_seq, train_pred),
            'final_train_loss': self.history.history['loss'][-1]
        }
        
        if validation_data:
            val_pred = self.model.predict(X_val_seq)
            metrics.update({
                'val_mae': mean_absolute_error(y_val_seq, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val_seq, val_pred)),
                'val_r2': r2_score(y_val_seq, val_pred),
                'final_val_loss': self.history.history['val_loss'][-1]
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Make predictions
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot training history"""
        if not self.history:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        if 'val_mae' in self.history.history:
            axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr_history = []
            for epoch in range(len(self.history.history['loss'])):
                lr_history.append(self.config['learning_rate'])
            axes[1, 0].plot(lr_history)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save model weights
        self.model.save_weights(f"{filepath}_weights")
        
        # Save model data
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'model_architecture': self.model.to_json()
        }
        joblib.dump(model_data, f"{filepath}_data.pkl")
    
    def load(self, filepath: str):
        """Load trained model"""
        # Load model data
        model_data = joblib.load(f"{filepath}_data.pkl")
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Reconstruct model
        model_json = model_data['model_architecture']
        self.model = keras.models.model_from_json(model_json)
        self.model.load_weights(f"{filepath}_weights")
        
        # Recompile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )


class ParkingGRU:
    """
    GRU model for parking occupancy temporal pattern recognition
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GRU model
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 24,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.history = None
        
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build GRU model architecture
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.GRU(
                self.config['hidden_size'],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.config['dropout']
            )
        ])
        
        # Add additional GRU layers
        for _ in range(self.config['num_layers'] - 2):
            model.add(layers.GRU(
                self.config['hidden_size'],
                return_sequences=True,
                dropout=self.config['dropout']
            ))
            
        # Final GRU layer
        if self.config['num_layers'] > 1:
            model.add(layers.GRU(
                self.config['hidden_size'],
                return_sequences=False,
                dropout=self.config['dropout']
            ))
        
        # Output layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.config['dropout']))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for GRU training (same as LSTM)"""
        sequence_length = self.config['sequence_length']
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(data)):
            X_sequences.append(data[i-sequence_length:i])
            y_sequences.append(target[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train GRU model (similar implementation to LSTM)"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor='val_loss' if validation_data else 'loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_seq)
        metrics = {
            'train_mae': mean_absolute_error(y_train_seq, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
            'train_r2': r2_score(y_train_seq, train_pred),
            'final_train_loss': self.history.history['loss'][-1]
        }
        
        if validation_data:
            val_pred = self.model.predict(X_val_seq)
            metrics.update({
                'val_mae': mean_absolute_error(y_val_seq, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val_seq, val_pred)),
                'val_r2': r2_score(y_val_seq, val_pred),
                'final_val_loss': self.history.history['val_loss'][-1]
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained GRU model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()


class TimeSeriesEnsemble:
    """
    Ensemble of LSTM and GRU models for robust predictions
    """
    
    def __init__(self, lstm_config: Optional[Dict] = None, gru_config: Optional[Dict] = None):
        """
        Initialize ensemble of time series models
        
        Args:
            lstm_config: LSTM configuration
            gru_config: GRU configuration
        """
        self.lstm_model = ParkingLSTM(lstm_config)
        self.gru_model = ParkingGRU(gru_config)
        self.weights = {'lstm': 0.5, 'gru': 0.5}
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train both LSTM and GRU models
        
        Returns:
            Training metrics for both models
        """
        print("Training LSTM model...")
        lstm_metrics = self.lstm_model.train(X_train, y_train, X_val, y_val)
        
        print("\nTraining GRU model...")
        gru_metrics = self.gru_model.train(X_train, y_train, X_val, y_val)
        
        # Adjust weights based on validation performance
        if X_val is not None and y_val is not None:
            lstm_val_loss = lstm_metrics.get('val_mae', lstm_metrics['train_mae'])
            gru_val_loss = gru_metrics.get('val_mae', gru_metrics['train_mae'])
            
            # Inverse weighting (better performance gets higher weight)
            total_inv_loss = (1/lstm_val_loss) + (1/gru_val_loss)
            self.weights['lstm'] = (1/lstm_val_loss) / total_inv_loss
            self.weights['gru'] = (1/gru_val_loss) / total_inv_loss
        
        self.is_trained = True
        
        return {
            'lstm_metrics': lstm_metrics,
            'gru_metrics': gru_metrics,
            'ensemble_weights': self.weights
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Weighted ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        lstm_pred = self.lstm_model.predict(X)
        gru_pred = self.gru_model.predict(X)
        
        # Weighted average
        ensemble_pred = (self.weights['lstm'] * lstm_pred + 
                        self.weights['gru'] * gru_pred)
        
        return ensemble_pred
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        return {
            'lstm': self.lstm_model.predict(X),
            'gru': self.gru_model.predict(X),
            'ensemble': self.predict(X)
        }
