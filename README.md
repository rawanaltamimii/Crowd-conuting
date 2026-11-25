<p align="center">
  <img src="assets/banner.svg" alt="Crowd Counting banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-BFE3F7?style=flat-square&logo=python&logoColor=4A5568&labelColor=FFFFFF" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FFE9C7?style=flat-square&logo=tensorflow&logoColor=4A5568&labelColor=FFFFFF" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-Vision-BDEBD6?style=flat-square&logo=opencv&logoColor=4A5568&labelColor=FFFFFF" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-Metrics-FFD6E8?style=flat-square&logo=scikitlearn&logoColor=4A5568&labelColor=FFFFFF" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Task-Regression-E4D9FF?style=flat-square&labelColor=FFFFFF" alt="Regression">
</p>

---

## 🌸 Overview

This project estimates **how many people appear in a surveillance image**. Instead of detecting each person, the task is treated as a **regression problem**: the model receives an image and outputs a single number, the crowd count.

Three deep learning approaches are built, trained, and compared on the same data split:

| # | Model | Idea |
|---|-------|------|
| 🟢 1 | **Custom CNN** | A small CNN built from scratch |
| 🌷 2 | **CSRNet-inspired** | VGG-style front-end plus dilated-convolution back-end |
| 🔵 3 | **MobileNetV2** | Transfer learning with ImageNet weights |

---

## 🌼 Dataset

| Item | Value |
|------|-------|
| Images | 2,000 frames (`.jpg`, 640×480×3) |
| Labels | `labels.csv` with `id` (image number) and `count` (people in the image) |
| Missing images | 0 |
| People per image | min **13**, max **53**, mean **31.16**, median **31**, std **6.95** |

Images are named `seq_000001.jpg`, `seq_000002.jpg`, and so on, matching the `id` column.

> 📁 The dataset is not included in this repository. Add your dataset source link here.

---

## 🧁 Pipeline

1. **Load and inspect** the labels and images (formats, shapes, sizes, missing files)
2. **Explore** the count distribution (histogram, boxplot, density levels pie chart, random samples)
3. **Preprocess**: BGR → RGB, resize to **128×128**, normalize pixels to **[0, 1]**
4. **Split** the data: **70% train (1,400) / 10% validation (200) / 20% test (400)**, `random_state=42`
5. **Train** each model with MSE loss, MAE metric, Adam (lr = 0.001), and Early Stopping (patience = 5)
6. **Evaluate** with MAE, MSE, RMSE, R², and MAPE, then compare

Brightness adjustment and horizontal flip are demonstrated in the notebook as preprocessing and augmentation examples.

---

## 🍃 Results

<p align="center">
  <img src="assets/model_comparison.png" alt="Model comparison" width="85%">
</p>

| Model | Test MSE | Test MAE | RMSE | R² | MAPE % | Approx. Performance % |
|-------|:-------:|:-------:|:----:|:--:|:------:|:---------------------:|
| 🟢 **Custom CNN** | **8.35** | **2.28** | **2.89** | **0.830** | **7.52** | **92.68** |
| 🌷 CSRNet-inspired | 67.74 | 6.79 | 8.23 | -0.381 | 26.07 | 78.18 |
| 🔵 MobileNetV2 | 11.36 | 2.69 | 3.37 | 0.768 | 8.79 | 91.37 |

*Approx. Performance % = (1 − MAE / mean test count) × 100. It is a rough indicator, not an official accuracy, since this is a regression task.*

### ✨ Key findings

- 🏆 The **Custom CNN** was the best model: on average it is off by about **2.3 people** per image.
- 🔵 **MobileNetV2** came close (MAE 2.69) while training only about 164K parameters.
- 🌷 The **CSRNet-inspired** model was unstable: its validation loss jumped between epochs and it stopped early after 8 epochs, leaving it with the weakest results.

---

## 🌿 Model Details

<details>
<summary><b>🟢 Custom CNN</b></summary>

`Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Dense(1)`

Total parameters: about 3.3M. Trained with batch size 32 for up to 30 epochs.
</details>

<details>
<summary><b>🌷 CSRNet-inspired</b></summary>

A front-end of 7 convolution layers with BatchNormalization, followed by a back-end of 4 dilated convolutions (dilation rate 2), then `GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(1)`.

Total parameters: about 3.3M. Trained with batch size 32 for up to 30 epochs.
</details>

<details>
<summary><b>🔵 MobileNetV2 (Transfer Learning)</b></summary>

MobileNetV2 (ImageNet weights, frozen) → `GlobalAveragePooling → Dense(128) → Dropout(0.3) → Dense(1)`

Inputs are prepared with `preprocess_input`. Trainable parameters: about 164K of 2.4M. Trained for 20 epochs.
</details>

---

## 🗂️ Project Structure

```
Crowd-counting/
├── assets/
│   ├── banner.svg
│   └── model_comparison.png
├── 2026deepL (1).ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install the requirements
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook "2026deepL (1).ipynb"
```

Before running, update these two paths in the notebook to point to your own data:

```python
labels_path   = "path/to/labels.csv"
images_folder = "path/to/frames"
```

---

## 🔭 Limitations and Future Work

- Frames come from video, so neighboring frames look very similar. A random split can place near-duplicate frames in both train and test, which may make the results look better than they would on truly new scenes. Splitting by time (for example, the first frames for training and the last for testing) would give a stricter test.
- Try data augmentation during training, such as horizontal flip and brightness changes.
- Try a higher resolution (for example 224×224) and fine-tune the top layers of MobileNetV2.
- Move toward density-map prediction, as in the original CSRNet, instead of predicting only a single count.

---

<p align="center">
 <b>Rawan Altamimi</b><br>
  <sub>⭐ If you like this project, give it a star!</sub>
</p>
