Here’s a draft README for your GitHub repository based on your implementation and the provided research article:

---

# E-Learning Distraction Detection via Facial and Postural Analysis

This repository implements a machine learning-based approach for detecting student distractions during e-learning lectures using human facial features and postural information. The system uses data collected via a standard webcam and employs features extracted using the [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) library and skeletal posture estimation with graph-based neural networks.

## 🚀 Overview

E-learning provides flexibility but poses challenges in monitoring students' attention and engagement. This project addresses these challenges by detecting distraction states during online lectures. The system extracts and processes facial and postural information to predict whether a student is distracted, using machine learning models such as Random Forest and XGBoost.

## 🧰 Features

- **Data Collection**: Captures video of students from the shoulders up during e-learning sessions.
- **Facial Feature Extraction**: Uses OpenFace to extract:
  - Action Units (AUs)
  - Eye and mouth areas
  - Gaze direction and angles
- **Postural Information Extraction**: Leverages GAST-Net for:
  - 3D skeletal coordinates
  - Shoulder, neck, and torso angles
- **Machine Learning Models**: Implements Random Forest, XGBoost, Decision Trees, KNN, and SVM for binary (focused/distraction) and multi-class (focused, normal, distracted) classification.
- **Performance Metrics**: Evaluates models using accuracy, precision, recall, and F1-score.

## 📁 Directory Structure

```
E-Learning-Distraction-Detection-via-Facial-and-Postural-Analysis/
├── data/                     # Input data folder (CSV files)
├── checkpoints/              # Saved model scalers
├── src/                      # Source code folder
│   ├── Preprocessing.py      # Data preprocessing and feature extraction
│   ├── Training.py           # Model training and evaluation
│   ├── Utils.py              # Utility functions
├── results/                  # Results and visualizations
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## 🛠️ Setup and Usage

### Prerequisites

- Python 3.8 or later
- Install dependencies using:

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Organize your input data in the `data/` folder.
2. Ensure CSV files contain relevant columns such as frame, timestamp, gaze angles, pose rotations, and AU values.

### Running the Code

1. **Preprocessing**: Extract features and process data using the `Preprocessing.py` script.

```bash
python src/Preprocessing.py
```

2. **Model Training and Evaluation**: Train the machine learning models using `Training.py`.

```bash
python src/Training.py
```

3. **Results**: Outputs include performance metrics and processed data saved in the `results/` folder.

## 📊 Results

- **Binary Classification**: Achieved over 90% recall with individual-specific training.
- **Multi-class Classification**: Performance drops due to imbalanced data but demonstrates the potential of integrating gaze, posture, and facial expressions.

Refer to the experiment section in the associated [research paper](https://doi.org/10.1007/s10015-022-00809-z) for more detailed results.

## 📚 Research Background

This implementation is based on the paper *"Distraction detection of lectures in e-learning using machine learning based on human facial features and postural information"* published in Artificial Life and Robotics, 2023. Key highlights include:
- Novel combination of facial AUs and skeletal data for distraction detection.
- Use of machine learning for binary and multi-class classification tasks.
- High-performance models using minimal hardware (a simple webcam).

For details, see the full [publication](https://doi.org/10.1007/s10015-022-00809-z).

## 🔮 Future Work

- Incorporate deep learning models like LSTM for time-series data.
- Expand the dataset with more participants and diverse environments.
- Enhance feature extraction to include behaviors like yawning or folded arms.

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

Let me know if you need additional customization!
