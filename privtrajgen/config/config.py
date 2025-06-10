#!/usr/bin/env python3
"""Contains all configuration parameters."""
import os
import sys
from pathlib import Path

# ########################### SEEDS ############################
# Overwritten by ValidationConfig!
RANDOM_SEED = 42

# ########################### PATHS ############################
BASE_DIR = str(Path(__file__).parent.parent.parent.resolve()) + '/'
TMP_DIR = os.path.join(BASE_DIR, 'tmp/')
LOG_DIR = os.path.join(BASE_DIR, 'logs/')
DATA_DIR = os.path.join(BASE_DIR, 'data/')
RESULT_DIR = os.path.join(BASE_DIR, 'results/')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'tensorboard/')
PARAM_PATH = os.path.join(BASE_DIR, 'parameters/')

# ########################### PrivTrace ###########################
PRIVTRACE_DIR = os.path.join(BASE_DIR, 'PrivTrace/')
PRIVTRACE_OUTPUT_DIR = os.path.join(PRIVTRACE_DIR, 'outputs/')

# ########################### TF Fixes ############################
# We mainly use PyTorch, this is mainly legacy code for old projects
env_dir = sys.prefix  # Get the current conda/venv environment directory
# Libdevice not found fix:
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={os.path.join(env_dir, 'lib')}"
# Reduce all the Keras/TensorFlow info messages (only show warning and above)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
# Link TensorRT
site_packages_path = os.path.join(env_dir, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
tensorrt_path = os.path.join(site_packages_path, 'tensorrt')
lib_path = os.path.join(env_dir, 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{site_packages_path}:{tensorrt_path}:{lib_path}"
