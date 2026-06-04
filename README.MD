# SweatGland_Quantification

Official implementation of the automated deep-learning pipeline (**DepthNet**) for high-throughput mouse sudomotor function quantification via U-Net. This repository directly supports our ***Nature Protocols* submission**.

The analytical framework orchestrates continuous algorithmic execution across major structural milestones: localized region-of-interest (ROI) tensor cropping ($288 \times 288$ pixels), background noise attenuation, neural network segmentation gating, and automated wide-format spreadsheet matrix reshaping.

---

## 💻 Environment Setup

To ensure all deep learning dependencies (PyTorch, OpenCV, etc.) are perfectly matched and to maintain absolute structural reproducibility, we provide a pre-configured `environment.yml` file for automated Conda virtual environment provisioning.

Open your terminal or Git Bash, navigate to the project directory, and execute:

1. Create the virtual environment from the yml file:
```bash
conda env create -f environment.yml

2. Activate the newly created environment:
conda activate depthnet_env

How to Run the Pipeline
The core software code package is structured as an end-to-end executable pipeline containing three sequential production modules. Please run the scripts chronologically from your active Conda environment:

Step 1: Model Training (Optional)
If you wish to retrain the architecture on your customized datasets under our hyperparameter constraints, execute the optimization loop:
python Code/0_Train_Model.py

Step 2: Batch Inference & Spatial Gating
To execute automated batch prediction on raw localized hind paw profiles using our optimized pre-trained weights (final_model.pth):
python Code/1_Inference_Batch.py

Step 3: Post-Processing & Quantitative Extraction
To filter out non-functional spatial artifacts, execute 8-connectivity Connected Component Analysis (CCA), and export the finalized metrics:
python Code/2_PostProcess_Quantification.py
Output Destination: The finalized quantitative results will be automatically aggregated and exported into Final_Quantification_Table.csv in your root directory.

📂 Repository Structure
To verify your local repository path alignment, ensure your file tree matches the standardized organizational structure below:
SweatGland_Quantification/
├── Code/
│   ├── 0_Train_Model.py                  # Model training and optimization loop
│   ├── 1_Inference_Batch.py              # Batch tensor inference and mask generation
│   ├── 2_PostProcess_Quantification.py   # Connected Component Analysis (CCA) extraction
│   ├── Model_Architecture.py             # DepthNet neural network definitions
│   └── CustomLoss.py                     # Integrated Dice Loss function
├── Test_Data/
│   ├── image/                            # Sample raw mouse footpad images for validation
│   └── mask/                             # Standardized ROI binary masks for spatial gating
├── train_result/
│   └── weights/
│       └── final_model.pth               # Pre-trained neural network weights (52.56 MB)
├── environment.yml                       # Conda environment dependency registry
├── LICENSE                               # MIT Open-Source License
└── README.md                             # Software documentation manual
📜 License
This project is licensed under the standard MIT License - see the LICENSE file for details.
