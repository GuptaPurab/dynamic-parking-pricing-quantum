"""
Data preprocessing utilities for parking pricing system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional


class ParkingDataPreprocessor:
    """
    Comprehensive data preprocessing for parking pricing system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def preprocess_parking_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw parking data into ML-ready features
        
        Args:
            df: Raw parking dataset
            
        Returns:
            Processed DataFrame with engineered features
        """
        data = df.copy()
        
        # Parse datetime information
        data = self._create_datetime_features(data)
        
        # Create occupancy and demand features
        data = self._create_occupancy_features(data)
        
        # Create categorical encodings
        data = self._encode_categorical_features(data)
        
        # Create lagged features for time series
        data = self._create_lagged_features(data)
        
        # Create target variables
        data = self._create_target_variables(data)
        
        return data
    
    def _create_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        # Parse datetime if not already done
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(
                data['LastUpdatedDate'] + ' ' + data['LastUpdatedTime']
            )
        
        # Extract temporal features
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['month'] = data['datetime'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Rush hour indicators
        data['is_morning_rush'] = ((data['hour'] >= 7) & (data['hour'] <= 9)).astype(int)
        data['is_evening_rush'] = ((data['hour'] >= 17) & (data['hour'] <= 19)).astype(int)
        data['is_rush_hour'] = (data['is_morning_rush'] | data['is_evening_rush']).astype(int)
        
        # Cyclical encoding for temporal features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _create_occupancy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create occupancy and utilization features"""
        # Calculate occupancy rate
        data['occupancy_rate'] = data['Occupancy'] / data['Capacity']
        
        # Demand pressure indicators
        data['utilization_level'] = pd.cut(
            data['occupancy_rate'], 
            bins=[0, 0.3, 0.7, 0.9, 1.0], 
            labels=['low', 'medium', 'high', 'critical'],
            include_lowest=True
        )
        
        # Queue pressure
        data['queue_pressure'] = data['QueueLength'] / (data['Capacity'] * 0.1)  # Normalize by 10% of capacity
        data['high_queue'] = (data['QueueLength'] > 5).astype(int)
        
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        # Traffic condition mapping
        traffic_map = {'low': 1, 'average': 2, 'high': 3}
        vehicle_map = {'cycle': 1, 'bike': 2, 'car': 3, 'truck': 4}
        
        data['traffic_numeric'] = data['TrafficConditionNearby'].map(traffic_map)
        data['vehicle_numeric'] = data['VehicleType'].map(vehicle_map)
        
        # One-hot encoding for utilization level
        if 'utilization_level' in data.columns:
            util_dummies = pd.get_dummies(data['utilization_level'], prefix='util')
            data = pd.concat([data, util_dummies], axis=1)
        
        return data
    
    def _create_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series modeling"""
        # Sort by datetime for proper lagging
        data = data.sort_values('datetime')
        
        # Create lagged occupancy features
        for lag in [1, 3, 6, 24]:  # 1h, 3h, 6h, 24h lags
            data[f'occupancy_lag_{lag}h'] = data['occupancy_rate'].shift(lag)
            data[f'queue_lag_{lag}h'] = data['QueueLength'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            data[f'occupancy_mean_{window}h'] = data['occupancy_rate'].rolling(window).mean()
            data[f'occupancy_std_{window}h'] = data['occupancy_rate'].rolling(window).std()
        
        return data
    
    def _create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML models"""
        base_price = 10.0
        
        # Dynamic pricing formula based on demand factors
        data['optimal_price'] = base_price * (
            1.0 +
            1.5 * data['occupancy_rate'] +
            0.3 * (data['traffic_numeric'] - 1) / 2 +
            0.2 * np.clip(data['QueueLength'] / 10, 0, 1) +
            0.4 * data['IsSpecialDay'] +
            0.1 * (data['vehicle_numeric'] - 1) / 3 +
            0.2 * data['is_rush_hour']
        )
        
        # Revenue target (price * expected utilization)
        data['expected_utilization'] = np.clip(
            data['occupancy_rate'] * (1 - 0.1 * (data['optimal_price'] - base_price) / base_price),
            0, 1
        )
        data['revenue_target'] = data['optimal_price'] * data['expected_utilization']
        
        # Classification target for demand level
        data['demand_category'] = pd.cut(
            data['occupancy_rate'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['low', 'medium', 'high', 'peak'],
            include_lowest=True
        )
        
        return data
    
    def prepare_features_for_ml(self, data: pd.DataFrame, 
                               feature_groups: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for ML models
        
        Args:
            data: Processed DataFrame
            feature_groups: List of feature group names to include
            
        Returns:
            Feature matrix and feature names
        """
        if feature_groups is None:
            feature_groups = ['temporal', 'demand', 'external', 'lagged']
        
        feature_columns = []
        
        if 'temporal' in feature_groups:
            feature_columns.extend([
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ])
        
        if 'demand' in feature_groups:
            feature_columns.extend([
                'occupancy_rate', 'queue_pressure', 'high_queue',
                'traffic_numeric', 'vehicle_numeric'
            ])
        
        if 'external' in feature_groups:
            feature_columns.extend(['IsSpecialDay'])
        
        if 'lagged' in feature_groups:
            lagged_cols = [col for col in data.columns if '_lag_' in col or '_mean_' in col or '_std_' in col]
            feature_columns.extend(lagged_cols)
        
        # Filter columns that exist in data
        available_columns = [col for col in feature_columns if col in data.columns]
        
        # Handle missing values
        feature_data = data[available_columns].fillna(data[available_columns].mean())
        
        self.feature_columns = available_columns
        
        return feature_data.values, available_columns
    
    def create_sequences_for_lstm(self, data: pd.DataFrame, 
                                  sequence_length: int = 24,
                                  target_col: str = 'occupancy_rate') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU models
        
        Args:
            data: Processed DataFrame
            sequence_length: Length of input sequences
            target_col: Target column name
            
        Returns:
            X sequences and y targets
        """
        # Prepare features
        features, _ = self.prepare_features_for_ml(data)
        targets = data[target_col].values
        
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(features)):
            X_sequences.append(features[i-sequence_length:i])
            y_sequences.append(targets[i])
        
        return np.array(X_sequences), np.array(y_sequences)
