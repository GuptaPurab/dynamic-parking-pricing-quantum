"""
XGBoost and LightGBM models for parking occupancy forecasting
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostForecaster:
    """
    XGBoost model for parking occupancy forecasting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize XGBoost forecaster
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names for interpretability
            
        Returns:
            Training metrics
        """
        # Initialize model
        self.model = xgb.XGBRegressor(**self.config)
        
        # Set up validation
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=50 if eval_set else None
        )
        
        # Store feature importance
        if feature_names:
            importance_scores = self.model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        # Add validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred)
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
        
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model with feature_names.")
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'XGBoost Feature Importance (Top {top_n})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = model_data['is_trained']


class LightGBMForecaster:
    """
    LightGBM model for parking occupancy forecasting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LightGBM forecaster
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names for interpretability
            
        Returns:
            Training metrics
        """
        # Initialize model
        self.model = lgb.LGBMRegressor(**self.config)
        
        # Set up validation
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50)] if eval_set else None
        )
        
        # Store feature importance
        if feature_names:
            importance_scores = self.model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        # Add validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred)
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance rankings"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model with feature_names.")
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'LightGBM Feature Importance (Top {top_n})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = model_data['is_trained']


class ForecastingEnsemble:
    """
    Ensemble of forecasting models for improved predictions
    """
    
    def __init__(self, models_config: Optional[Dict] = None):
        """
        Initialize ensemble of forecasters
        
        Args:
            models_config: Configuration for individual models
        """
        self.models_config = models_config or {}
        self.models = {}
        self.weights = None
        self.is_trained = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Model identifier
            model: Model instance
            weight: Model weight in ensemble
        """
        self.models[name] = {'model': model, 'weight': weight}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all models in ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            
        Returns:
            Training metrics for all models
        """
        all_metrics = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            print(f"Training {name}...")
            
            metrics = model.train(X_train, y_train, X_val, y_val, feature_names)
            all_metrics[name] = metrics
            
        self.is_trained = True
        return all_metrics
    
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
        
        predictions = []
        weights = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        return np.average(predictions, axis=0, weights=weights)
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary of model predictions
        """
        predictions = {}
        for name, model_info in self.models.items():
            model = model_info['model']
            predictions[name] = model.predict(X)
        
        return predictions
