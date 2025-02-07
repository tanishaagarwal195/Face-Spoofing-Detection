# **Face-Spoofing-Detection Using Ensemble Learning**
Research Specialization Project

## **Overview**  
This repository contains the research work on **Face Spoofing Detection** using **ensemble learning techniques**. The study improves detection accuracy by leveraging **Local Binary Pattern (LBP) preprocessing**, deep learning models (**ResNet50, EfficientNetB0, MobileNetV2**), and **ensemble averaging** to enhance robustness against spoof attacks.  

## **Project Structure**  
```
📂 Face-Spoofing-Detection
├── 📄 SurveyPaper.pdf            # Original survey paper
├── 📄 Updated_SurveyPaper.pdf    # Updated version with latest methodology & results
├── 📂 data                       # Dataset (CASIA Face Anti-Spoofing Dataset)
│   ├── train/                    # Training images
│   ├── test/                      # Testing images
│   ├── lbp_output/                # LBP processed images
│   ├── lbp_labeled_data.csv       # LBP dataset metadata
├── 📂 models                     # Trained deep learning models
│   ├── resnet50.pth
│   ├── efficientnet_b0.pth
│   ├── mobilenet_v2.pth
├── 📂 scripts                    # Training & Evaluation Scripts
│   ├── preprocess.py             # Preprocessing: Face detection & LBP extraction
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Model evaluation script
│   ├── ensemble.py               # Ensemble learning implementation
├── 📄 README.md                   # Project documentation (this file)
```

## **Methodology**  
1. **Preprocessing**  
   - **Face Detection**: Extracts facial regions from images.  
   - **Grayscale Conversion & LBP**: Converts images and applies **Local Binary Pattern (LBP)** for texture-based feature extraction.  
   - **Image Normalization**: Resizes images to **224x224** and normalizes them for deep learning models.  

2. **Model Training**  
   - **Individual Models**:  
     - **ResNet50, EfficientNetB0, and MobileNetV2** trained on processed images.  
     - Fully connected layers modified for binary classification (real/spoof).  
   - **Loss Function**: Cross-Entropy Loss  
   - **Optimizer**: Adam (learning rate = 0.0001)  

3. **Ensemble Learning**  
   - Uses **averaging of predictions** from all three models.  
   - Reduces model-specific biases and improves overall robustness.  

4. **Evaluation**  
   - **Individual Model Accuracy**:  
     - **ResNet50** → 94.43%  
     - **EfficientNetB0** → 95.82%  
     - **MobileNetV2** → 95.82%  
   - **Ensemble Accuracy**: **98.27%**  

## **Results & Observations**  
✅ **Ensemble Learning** outperforms individual models, providing **better generalization** against attacks.  
✅ **LBP preprocessing** improves texture-based spoof detection.  
✅ **ResNet50, EfficientNetB0, and MobileNetV2** effectively classify real vs. spoofed faces.  
✅ **CASIA dataset limitations** suggest testing on **more diverse datasets** for better real-world generalization.  

## **Limitations & Future Work**  
🚀 **Challenges:**  
- **Computational Cost**: Requires GPU resources for training & real-time deployment.  
- **Dataset Generalization**: Need to validate performance on other datasets (e.g., OULU-NPU, Replay-Attack).  
- **Real-Time Processing**: Further optimization is required for **mobile & embedded systems**.  

🛠 **Future Enhancements:**  
- Experimenting with **Stacking & Boosting** techniques for better ensemble learning.  
- Expanding dataset diversity with **more attack scenarios (3D masks, deepfakes)**.  
- Exploring **lightweight deep learning models** for **on-device** inference.  

## **How to Run**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/Face-Spoofing-Detection.git
cd Face-Spoofing-Detection
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Preprocessing**  
```bash
python scripts/preprocess.py --input data/train/ --output data/lbp_output/
```

### **4️⃣ Train the Model**  
```bash
python scripts/train.py --epochs 5 --batch_size 32
```

### **5️⃣ Evaluate Performance**  
```bash
python scripts/evaluate.py --model resnet50
```

### **6️⃣ Run Ensemble Prediction**  
```bash
python scripts/ensemble.py
```

---

## **Contributors**  
👩‍💻 **Tanisha Agarwal**  
📧 **tanisha.agarwal@msds.christuniversity.in**  

---
