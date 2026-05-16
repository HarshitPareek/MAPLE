#!/bin/bash
# One-time setup script for CineGlass
set -e

echo "=== CineGlass Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "✓ Virtual environment created"
fi

source venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Link raw data
mkdir -p data/raw data/processed database
RAW_ARCHIVE="/Users/harshitpareek/Downloads/archive (1)"
for f in movies_metadata.csv keywords.csv credits.csv; do
  if [ ! -f "data/raw/$f" ] && [ -f "$RAW_ARCHIVE/$f" ]; then
    cp "$RAW_ARCHIVE/$f" "data/raw/$f"
    echo "✓ Copied $f"
  fi
done

# Run preprocessing
echo ""
echo "=== Running ML preprocessing (takes 2-5 minutes) ==="
python3 -m app.ml.preprocess

echo ""
echo "=== Setup complete! ==="
echo "Start the app with: source venv/bin/activate && python run.py"
echo "Then open: http://localhost:5000"
