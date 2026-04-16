# Driver Drowsiness Detection System
### Real-Time CNN-Based Eye State Classification using Transfer Learning

**Authors:** Avani Jain, Harita Venkatesan, Jatin Rajabhoj, Jayaharsh Kosanam  
**Institution:** RV University, Bangalore, India  
**Course:** Machine Learning and Data Science (MLDS)

---

## Abstract

This project presents a real-time Driver Drowsiness Detection System using a lightweight Convolutional Neural Network (CNN) trained to classify eye states as open or closed. The system uses OpenCV for face and eye detection via Haar Cascade classifiers and a custom CNN model for binary classification. When a driver's eyes remain closed for 20 or more consecutive frames, a drowsiness alert is triggered. The system runs at 25–30 FPS on a standard laptop webcam, making it suitable for real-world deployment.

---

## Project Structure

```
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── /src
│   └── drowsiness_detector.py  ← Main detection script
├── /notebooks
│   └── model_training.ipynb    ← Training notebook (optional)
├── /data
│   └── README_data.md          ← Instructions to download dataset
├── /results
│   └── sample_output.png       ← Sample output screenshots
└── report.pdf                  ← IEEE format research paper
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the Drowsiness Detection Dataset from Kaggle:
- https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

Organize it as follows:
```
dataset/
├── open_eye/
└── closed_eye/
```
Place the `dataset/` folder in the root of the project.

### 4. Run the System
```bash
python src/drowsiness_detector.py
```

- **First run:** Trains the CNN model and saves it as `drowsiness_model.h5`
- **Subsequent runs:** Loads saved model and opens webcam directly
- **Press Q** to quit

---

## Model Details

| Property | Value |
|---|---|
| Model Type | Supervised Binary Classification |
| Architecture | Lightweight CNN (2 Conv layers) |
| Input | 24x24 grayscale eye image |
| Output | Probability (0=Closed, 1=Open) |
| Training Epochs | 10 (with early stopping) |
| Validation Accuracy | ~93% |
| Inference Speed | 25–30 FPS |

---

## How It Works

```
Webcam Feed
     ↓
Face Detection (Haar Cascade)
     ↓
Eye Region Cropped (top 55% of face)
     ↓
Resized to 24x24 Grayscale
     ↓
CNN Model → Probability (0 to 1)
     ↓
< 0.5 = Closed → increment counter
≥ 0.5 = Open   → decrement counter
     ↓
Counter ≥ 20 frames → DROWSY ALERT
```

---

## Requirements

- Python 3.11+
- Webcam
- ~500MB disk space for dataset

---

## License

This project is submitted as part of academic coursework at RV University, Bangalore.
