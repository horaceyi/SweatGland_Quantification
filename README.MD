# SweatGland_Quantification

Official implementation of the automated deep-learning pipeline (**DepthNet**) for high-throughput mouse sudomotor function quantification via U-Net. This repository directly supports our ***Nature Protocols* submission**.

The analytical framework orchestrates continuous algorithmic execution across major structural milestones: localized region-of-interest (ROI) tensor cropping ($288 \times 288$ pixels), background noise attenuation, neural network segmentation gating, and automated wide-format spreadsheet matrix reshaping.

---

## 💻 Environment Setup

To ensure all deep learning dependencies (PyTorch, OpenCV, etc.) are perfectly matched and to maintain absolute structural reproducibility, we provide a pre-configured `environment.yml` file for automated Conda virtual environment provisioning.

Open your terminal or Git Bash, navigate to the project directory, and execute:

```bash
# 1. Create the virtual environment from the yml file
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate depthnet_env
