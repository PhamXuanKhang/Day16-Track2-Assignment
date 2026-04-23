#!/bin/bash
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting user_data setup: CPU LightGBM environment ==="

# System update + Python deps
dnf update -y
dnf install -y python3 python3-pip python3-devel gcc gcc-c++ make wget unzip

# Install Python ML packages
pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

# Create working directory
mkdir -p /home/ubuntu/ml-benchmark
chown ubuntu:ubuntu /home/ubuntu/ml-benchmark

echo "=== Environment setup complete. LightGBM is ready. ==="
echo "Next steps (run as ubuntu):"
echo "  1. Configure Kaggle credentials: ~/.kaggle/kaggle.json"
echo "  2. Download dataset: kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
echo "  3. Run benchmark: python3 ~/ml-benchmark/benchmark.py"
