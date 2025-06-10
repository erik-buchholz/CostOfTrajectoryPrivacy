#!/usr/bin/env bash

# Double check that the user prefers this over conda
echo -e "This script will set up a \e[1m\e[34mPython 3\e[0m venv and install the required packages."
echo -e "\e[1m\e[34mPython venv will only work if you have the correct Python version (3.12.3) installed.\e[0m"
echo -e "\e[1m\e[34mWe recommend using setup_conda.sh instead which will install the correct Python version.\e[0m"
# Ask if the user wants to continue and quite if they don't
read -p "Do you want to continue? (y/n) " -n 1 -r
# Exit if the user doesn't want to continue
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\nExiting..."
    exit 1
fi


echo "Setting up environment..."
if ! command -v python3 &> /dev/null || ! python3 -c "import venv" &> /dev/null; then
    echo "Either Python 3 is not installed or the venv module is missing. Installing them requires sudo access."
    sudo apt install python3.12 python3.12-venv
fi

# Ask the user for path to venv (Default: venv)
# shellcheck disable=SC2162
read -p "Enter the name for the virtual environment (Default: venv): " VENV_NAME

# Set the default value if the user didn't provide one
VENV_NAME=${VENV_NAME:-venv}

# Output path as italic
echo -e "Setting up the virtual environment at \e[3m${VENV_NAME}\e[0m"

python3 -m venv "$VENV_NAME"
source "$VENV_NAME"/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Running Unittests..."
if FORCE_TESTS=false RUN_SLOW_TESTS=true python3 -m unittest discover -s test; then
    echo "Setup Successful. Done."
    echo "To activate the environment, run 'source $VENV_NAME/bin/activate'"
else
    echo "Error - Setup unsuccessful."
fi
