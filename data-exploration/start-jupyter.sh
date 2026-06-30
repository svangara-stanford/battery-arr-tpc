# Start Jupyter Notebook for data exploration

# Check if virtual environment exists
if [ ! -d "battery-env" ]; then
    echo "Virtual environment not found. Run './setup.sh' first."
    exit 1
fi

# Activate virtual environment
source battery-env/bin/activate

echo "Starting Jupyter notebook. You can press Ctrl+C to stop Jupyter."
echo "Jupyter will open in your browser soon..."
echo ""

# Start Jupyter Lab
jupyter lab


