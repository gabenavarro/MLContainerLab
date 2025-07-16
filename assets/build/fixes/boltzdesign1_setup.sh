#!/bin/bash
set -e

git clone https://github.com/yehlincho/BoltzDesign1.git
cd BoltzDesign1

echo "🚀 Setting up Boltz Design Environment..."

# Check if conda is installed
if ! command -v mamba &> /dev/null; then
    echo "❌ Conda not found. Please install mamba first."
    exit 1
fi

# Install boltz
if [ -d "boltz" ]; then
    echo "📂 Installing Boltz..."
    cd boltz
    pip install -e .
    cd ..
else
    echo "❌ boltz directory not found. Please run this script from the project root."
    exit 1
fi
# Install conda dependencies; always use conda-forge, not anaconda
echo "🔧 Installing conda dependencies..."
mamba install -c conda-forge ipykernel -y

# Install Python dependencies
echo "🔧 Installing Python dependencies..."
pip install matplotlib seaborn prody tqdm PyYAML requests pypdb py3Dmol logmd==0.1.45

# Install PyRosetta
echo "⏳ Installing PyRosetta (this may take a while)..."
pip install pyrosettacolabsetup pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

# Download Boltz weights and dependencies
echo "⬇️  Downloading Boltz weights and dependencies..."
python -c "
from boltz.main import download
from pathlib import Path
cache = Path('~/.boltz')
cache.mkdir(parents=True, exist_ok=True)
download(cache)
print('✅ Boltz weights downloaded successfully!')
"

# Setup LigandMPNN if directory exists
if [ -d "LigandMPNN" ]; then
    echo "🧬 Setting up LigandMPNN..."
    cd LigandMPNN
    bash get_model_params.sh "./model_params"
    cd ..
fi

# Make DAlphaBall.gcc executable
chmod +x "boltzdesign/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# Setup Jupyter kernel for the environment
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=boltz_design --display-name="Boltz Design"

echo "🎉 Installation complete!"