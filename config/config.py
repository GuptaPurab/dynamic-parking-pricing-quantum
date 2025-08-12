"""
Configuration settings for Dynamic Parking Pricing with Quantum ML
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
    "classical_models": {
        "linear_regression": {
            "enabled": True,
            "params": {}
        },
        "random_forest": {
            "enabled": True,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        },
        "xgboost": {
            "enabled": True,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
        }
    },
    "quantum_models": {
        "quantum_svm": {
            "enabled": True,
            "feature_map": "ZZFeatureMap",
            "ansatz": "RealAmplitudes",
            "optimizer": "COBYLA"
        },
        "variational_classifier": {
            "enabled": True,
            "num_qubits": 4,
            "reps": 2,
            "optimizer": "SPSA"
        }
    }
}

# Quantum computing settings
QUANTUM_CONFIG = {
    "backend": "qasm_simulator",
    "shots": 1024,
    "max_qubits": 8,
    "noise_model": False,
    "optimization_level": 1
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
        "temporal": ["hour", "day_of_week", "is_weekend", "month"],
        "location": ["location_id", "zone_type", "distance_to_center"],
        "demand": ["occupancy_rate", "queue_length", "historical_avg"],
        "external": ["weather_score", "event_factor", "traffic_index"]
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
    QUANTUM_CONFIG["shots"] = 2048

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
