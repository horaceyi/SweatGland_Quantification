import torch
import os
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import Model_Architecture
import matplotlib.pyplot as plt
import cv2

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = r"...\SweatGland_Quantification_v1.0\train_result\weights\final_model.pth" #Modify here

# Data Paths
base_data_path = r"...\SweatGland_Quantification_v1.0\test_data" #Modify here
image_input_path = os.path.join(base_data_path, "image")
mask_input_path = os.path.join(base_data_path, "mask")

# Output directory for visualized results and logs
output_dir = "inference_results_cleaned"
os.makedirs(output_dir, exist_ok=True)

# Calibration Parameter: Physical area per pixel (mm^2/pixel)
PIXEL_AREA_MM2 = 5.997534e-03

# 1. Model Initialization
# DepthNet(input_channels=3, output_classes=2, feature_maps=32)
model = Model_Architecture.DepthNet(3, 2, 32).to(device)
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['net'])
model.eval()

# 2. Image Preprocessing (Must match Training augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

print(f"Starting batch inference and quantification (Area in mm^2)...")

# Filter image files
img_files = [f for f in os.listdir(image_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
results_log = []

with torch.no_grad():
    for f_name in img_files:
        # A. Load Image
        img_path = os.path.join(image_input_path, f_name)
        raw_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        # B. Model Prediction (Argmax for class index)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # --- Spatial Gating (ROI Masking) ---
        # Match Image filename with corresponding ROI Mask (e.g., Img1_1.png -> Mask1_1.png)
        mask_name = f_name.replace("Img", "Mask")
        mask_path = os.path.join(mask_input_path, mask_name)

        if os.path.exists(mask_path):
            roi_mask = cv2.imread(mask_path, 0)
            _, roi_mask_bin = cv2.threshold(roi_mask, 127, 1, cv2.THRESH_BINARY)
            # Apply bitwise multiplication to isolate the specific Pad
            pred_cleaned = pred * roi_mask_bin
        else:
            print(f"Warning: ROI Mask missing for {f_name}. Processing without spatial gating.")
            pred_cleaned = pred

        # --- Post-processing: Quantification ---
        # 1. Count active sweat glands via Connected Component Analysis (CCA)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_cleaned, connectivity=8)
        gland_count = num_labels - 1  # Exclude background label

        # 2. Calculate Physical Area
        # Sum total positive pixels and multiply by calibration constant
        pixel_count = int(np.sum(pred_cleaned == 1))
        real_area_mm2 = pixel_count * PIXEL_AREA_MM2

        # C. Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Original ROI
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Source: {f_name}")
        axes[0].axis('off')

        # Plot Raw Prediction (Includes peripheral noise)
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title("Raw Model Prediction")
        axes[1].axis('off')

        # Plot Gated Prediction (Cleaned results)
        axes[2].imshow(pred_cleaned, cmap='gray')
        axes[2].set_title(f"Gated Result\nCount: {gland_count}, Area: {real_area_mm2:.4f} mm2")
        axes[2].axis('off')

        # Save visualization for audit
        plt.savefig(os.path.join(output_dir, f"res_{f_name}"))
        plt.close()

        # D. Logging
        log_entry = f"{f_name}: {gland_count} glands, {real_area_mm2:.6f} mm2"
        results_log.append(log_entry)
        print(f"Processed: {f_name} | Count: {gland_count} | Area: {real_area_mm2:.6f} mm^2")

# Save summary log to text file for post-analysis
with open(os.path.join(output_dir, "summary_cleaned.txt"), "w") as f:
    f.write("\n".join(results_log))

print(f"\nInference Complete. Data successfully logged to summary_cleaned.txt using scale factor {PIXEL_AREA_MM2}.")