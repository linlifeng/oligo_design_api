#!/bin/bash
set -e  # Exit on any error

echo "Starting deployment..."

# Navigate to project directory
cd /home/phill/dev/oligo_design_api

# Stop existing gunicorn processes
echo "Stopping existing API service..."
pkill -f "gunicorn.*run:app" || true  # Don't fail if no process exists

# If venv folder does not exist, create it
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip just in case
pip install --upgrade pip

# Pull latest code from GitHub (force pull to avoid conflicts)
echo "Pulling latest code..."
git fetch origin main
git reset --hard origin/main

# Install requirements inside venv
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the service as daemon
echo "Starting API service..."
gunicorn -w 4 -b 127.0.0.1:8000 run:app --daemon

# Wait a moment for the service to start
sleep 3

# Verify the service is running
if pgrep -f "gunicorn.*run:app" > /dev/null; then
    echo "API service started successfully"
else
    echo "Failed to start API service"
    exit 1
fi

echo "Deployment completed successfully!"

# Note: We don't deactivate since the script ends here