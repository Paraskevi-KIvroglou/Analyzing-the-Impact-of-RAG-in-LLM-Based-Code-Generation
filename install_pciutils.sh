#!/bin/bash

# Update package list
sudo apt-get update

# Install pciutils
sudo apt-get install -y pciutils

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh