#!/usr/bin/env bash

echo "Setting up environment..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing might require sudo access."
    echo "Installing Python 3.12..."
    sudo apt install python3.12
fi
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Installing miniconda into ${HOME}/miniconda ..."
    echo "Downloading Conda..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniconda.sh
    echo "Installing Conda..."
    bash miniconda.sh -b -p "$HOME/miniconda"
    echo "Initialising Conda..."
    source "$HOME/miniconda/bin/activate"
    conda init
fi

# Ask the user for the environment name (Default: venv)
# shellcheck disable=SC2162
read -p "Enter the name for the virtual environment (Default: ptg): " VENV_NAME

# Set the default value if the user didn't provide one
VENV_NAME=${VENV_NAME:-ptg}

# Update Conda
conda update -n base -c defaults conda
# Create environment
conda env create --name "$VENV_NAME" -f environment.yml
# Activate environment
conda activate "$VENV_NAME"
echo "Running Unittests..."
if python3 -m unittest discover -s test; then
    echo "Setup Successful. Done."
    echo "To activate the environment, run 'conda activate $VENV_NAME'"
else
    echo "Error - Setup unsuccessful."
fi