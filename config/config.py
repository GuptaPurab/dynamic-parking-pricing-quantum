"""
Configuration settings for Dynamic Parking Pricing with Traditional ML
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Model configuration
MODEL_CONFIG = {
    "forecasting_models": {
        "xgboost": {
            "enabled": True,
            "params": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        },
        "lightgbm": {
            "enabled": True,
            "params": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1
            }
        },
        "random_forest": {
            "enabled": True,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    },
    "time_series_models": {
        "lstm": {
            "enabled": True,
            "params": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 24,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            }
        },
        "gru": {
            "enabled": True,
            "params": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 24,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            }
        }
    },
    "reinforcement_learning": {
        "q_learning": {
            "enabled": True,
            "params": {
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "min_epsilon": 0.01,
                "n_actions": 10,  # Number of discrete price actions
                "state_size": 8,   # Feature vector size
                "replay_buffer_size": 10000
            }
        }
    }
}

# Reinforcement Learning Environment settings
RL_CONFIG = {
    "environment": {
        "min_price": 5.0,
        "max_price": 50.0,
        "price_step": 2.5,
        "reward_function": "revenue_optimization",  # revenue_optimization, utilization_balance
        "penalty_overpricing": -10.0,
        "penalty_underutilization": -5.0
    },
    "training": {
        "episodes": 1000,
        "max_steps_per_episode": 100,
        "update_frequency": 10,
        "target_update_frequency": 50
    }
}

# Data generation settings
DATA_CONFIG = {
    "synthetic_data": {
        "n_samples": 5000,
        "n_locations": 10,
        "time_range": {
            "start_hour": 6,
            "end_hour": 23
        },
        "base_prices": [8, 10, 12, 15, 18, 20, 22, 25, 28, 30],
        "noise_level": 0.1
    },
    "features": {
        "temporal": ["hour", "day_of_week", "is_weekend", "month", "is_rush_hour"],
        "location": ["location_id", "zone_type", "distance_to_center"],
        "demand": ["occupancy_rate", "queue_length", "historical_avg", "demand_trend"],
        "external": ["weather_score", "event_factor", "traffic_index"],
        "pricing": ["current_price", "price_elasticity", "competitor_prices"],
        "lagged_features": ["occupancy_lag_1h", "occupancy_lag_3h", "occupancy_lag_24h"]
    }
}

# Evaluation metrics
METRICS_CONFIG = {
    "regression": ["mae", "mse", "rmse", "r2", "mape"],
    "classification": ["accuracy", "precision", "recall", "f1"],
    "business": ["revenue_optimization", "utilization_rate", "customer_satisfaction"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "parking_quantum.log"
}

# Visualization settings
VIZ_CONFIG = {
    "style": "default",
    "color_palette": "husl",
    "figure_size": (12, 8),
    "dpi": 100,
    "save_figures": True,
    "figure_format": "png"
}

# API configuration (for future deployment)
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "debug": True,
    "cors_enabled": True
}

# Environment-specific overrides
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    LOGGING_CONFIG["level"] = "WARNING"
    API_CONFIG["debug"] = False
    RL_CONFIG["training"]["episodes"] = 2000

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
