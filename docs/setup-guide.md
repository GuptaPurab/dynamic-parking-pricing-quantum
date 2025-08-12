# 🛠️ Environment Setup Guide

## Prerequisites
- **Python 3.9+** (3.9-3.11 recommended for Qiskit compatibility)
- **Git** for version control
- **8GB+ RAM** for quantum simulations
- **Windows PowerShell** or Command Prompt

## Step 1: Virtual Environment Setup

### Option A: Using venv (Recommended)
```bash
# Navigate to project directory
cd D:\learning_git\dynamic-parking-quantum

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Verify activation (should show venv path)
where python
```

### Option B: Using conda
```bash
# Create conda environment
conda create -n parking-quantum python=3.9

# Activate environment
conda activate parking-quantum
```

## Step 2: Install Dependencies

### Basic Installation
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install core packages
pip install -r requirements.txt
```

### Quantum Computing Setup
```bash
# Verify Qiskit installation
python -c "import qiskit; print(qiskit.__version__)"

# Test quantum simulator
python -c "from qiskit import Aer; print('Quantum simulator ready!')"
```

### Optional: GPU Support (for TensorFlow)
```bash
# For NVIDIA GPUs
pip install tensorflow-gpu

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Step 3: Development Tools

### Jupyter Notebook Setup
```bash
# Install Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable useful extensions
jupyter nbextension enable --py widgetsnbextension
```

### IDE Configuration (VS Code)
Install recommended extensions:
- Python
- Jupyter
- Pylance
- Quantum Development Kit

## Step 4: Verify Installation

### Create test script
```python
# test_setup.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qiskit
from sklearn.linear_model import LinearRegression

print("✅ NumPy:", np.__version__)
print("✅ Pandas:", pd.__version__)
print("✅ Matplotlib:", plt.matplotlib.__version__)
print("✅ Qiskit:", qiskit.__version__)
print("✅ Scikit-learn ready!")
print("🚀 Environment setup complete!")
```

### Run verification
```bash
python test_setup.py
```

## Step 5: Project Configuration

### Git Setup (if not already done)
```bash
git init
git add .
git commit -m "Initial project setup"
```

### Environment Variables
Create `.env` file for configuration:
```
# .env
PROJECT_NAME=dynamic-parking-quantum
QUANTUM_BACKEND=qasm_simulator
LOG_LEVEL=INFO
DATA_PATH=./data
MODELS_PATH=./models
```

## Troubleshooting

### Common Issues

**1. Qiskit Installation Failed**
```bash
# Try specific version
pip install qiskit==0.39.0 qiskit-machine-learning==0.5.0
```

**2. Memory Issues with Quantum Simulator**
```bash
# Use lighter simulator
export QISKIT_PARALLEL=FALSE
```

**3. TensorFlow GPU Not Detected**
```bash
# Install CUDA toolkit and cuDNN
# Follow TensorFlow GPU installation guide
```

**4. Pathway Installation Issues**
```bash
# Install from source if needed
pip install git+https://github.com/pathwaycom/pathway.git
```

## Next Steps

1. **✅ Complete environment setup**
2. **📚 Start with Week 1 theory (`docs/week1-theory.md`)**
3. **🧪 Run your first experiments**
4. **📝 Document any setup issues**

## Quick Start Commands

```bash
# Daily development routine
cd D:\learning_git\dynamic-parking-quantum
venv\Scripts\activate
jupyter notebook

# Or start with specific notebook
jupyter notebook notebooks/week1-experiments.ipynb
```

## Hardware Requirements

**Minimum:**
- 8GB RAM
- 4 CPU cores
- 10GB free disk space

**Recommended:**
- 16GB+ RAM
- 8+ CPU cores
- GPU with CUDA support
- SSD storage

Ready to begin your quantum machine learning journey! 🚀
