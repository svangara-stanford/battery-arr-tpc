# This script sets up the environment for data exploration by installing necessary packages and dependencies.
python3 -m venv battery-env
source battery-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete. To deactivate the virtual environment, run 'deactivate'."