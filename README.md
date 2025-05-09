# MLContainerLab

A collection of curated machine learning frameworks dockerized for local experimentation and seamless scaling to cloud compute platforms. MLContainerLab provides modular, reproducible, and optimized environments to streamline ML workflows from research to deployment.

[![GitHub license](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

## Overview

MLContainerLab is designed to solve common challenges in machine learning development:

- **Environment Consistency**: Eliminate "it works on my machine" problems with standardized environments
- **Hardware Optimization**: Pre-configured containers optimized for different hardware architectures (CPU, GPU, TPU)
- **Framework Compatibility**: Carefully curated dependency versions to avoid compatibility issues
- **Cloud Scaling**: Seamlessly transition from local development to cloud-based training
- **Reproducible Research**: Versioned environments that ensure consistent results across runs

## Repository Structure

```
MLContainerLab/
├── assets /                     # Container definitions for various ML environments
│   ├── build/                   # Base images with CUDA, Python, etc.
│   ├── config/                  # Specific ML framework configurations
│   └── images/                  # Images for examples and benchmarks
├── documentation/               # Example ML projects using the containers
│   ├── lit_datasets.ipynb       # Example notebook for downloading and preparing datasets
│   ├── vertex_ai_scaling.ipynb  # GCP Vertex AI scaling guide
│   ├── flash-attn.ipynb         # FlashAttention example - CUDA12.8 | Pytorch2.6 | Python 3.12
│   ├── mamba.ipynb              # Mamba example - CUDA12.1 | Pytorch2.6 | Python 3.10
│   └── flash-mamba.ipynb        # FlashAttention + Mamba example - CUDA12.8 | Pytorch2.6 | Python 3.12
└── scripts/                     # Utility scripts for building and running containers
    ├── build.sh                 # Build container images
    ├── run_local.sh             # Run containers locally
    └── cloud/                   # Scripts for cloud deployment
```

## Quick Start

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/username/MLContainerLab.git
   cd MLContainerLab
   ```

2. Build the container of your choice:
   ```bash
   docker build -t flash-attn-example:latest -f assets/build/Dockerfile.flashattn.cu128py26cp312 .
   ```

3. Run the container locally:
   ```bash
   docker run -dt \
       --gpus all \
       -v "$(pwd):/workspace" \
       --name flash-attention-example \
       --env NVIDIA_VISIBLE_DEVICES=all \
       flash-attn-example:latest
   ```

4. Mount running container to your IDE
   - For VSCode, use the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) to attach to the running container.
    - For PyCharm, use the [Docker plugin](https://www.jetbrains.com/help/pycharm/docker.html) to connect to the container.

### Example Notebooks

Explore our [documentation notebooks](/documentation) for comprehensive examples:

- [flash-attn.ipynb](/documentation/flash-attn.ipynb)
- [mamba.ipynb](/documentation/mamba.ipynb)
- [flash-mamba.ipynb](/documentation/flash-mamba.ipynb)

See the [datasets notebook](/documentation/lit_datasets.ipynb) for instructions on downloading and preparing these datasets with litdata.

## Cloud Deployment Examples

MLContainerLab makes it easy to scale your training to the cloud. Explore our cloud deployment guides:

### GCP Vertex AI

Our containers are optimized for seamless deployment to Google Cloud's Vertex AI platform. See the [GCP Vertex AI Scaling Guide Notebook](/documentation/vertex_ai_scaling.ipynb) for detailed instructions on:

- Building and pushing Docker images to GCP Artifact Registry
- Uploading configuration files and datasets to GCP Cloud Storage
- Configuring and submitting training jobs with optimal hardware
- Monitoring and managing your cloud training jobs

## Hardware Optimization

Our containers are specifically optimized for different hardware configurations:

- **NVIDIA Consumer GPUs** (RTX 40xx/50xx series)
- **NVIDIA Data Center GPUs** (A100, H100)
- **Multi-node** distributed setups
- **GCP Vertex AI** for scalable cloud training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project builds upon various open-source ML frameworks and tools
- Special thanks to the ML community for their invaluable resources
- Container optimizations inspired by best practices from NVIDIA, Google, and the PyTorch/TensorFlow communities