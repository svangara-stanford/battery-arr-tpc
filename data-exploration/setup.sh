# This script sets up the environment for data exploration by installing necessary packages and dependencies.
python3 -m venv battery-env
source battery-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment setup complete."
echo "Run 'source battery-env/bin/activate' to activate the virtual environment before starting Jupyter Notebook."
echo "To deactivate the virtual environment, run 'deactivate'."