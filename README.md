# Automated Detection of Cardiopulmonary Disease from Heart and Lung Audio Using Deep Learning

A deep learning project for the automated classification of **respiratory and heart sounds** using audio signal processing, feature extraction, and hybrid **Bidirectional LSTM–Bidirectional GRU (BiLSTM–BiGRU)** neural networks.

The project consists of two classification pipelines:

* **Respiratory Sound Classification** — classifies lung sounds into six diagnostic categories.
* **Heart Sound Classification** — classifies heart sounds into three categories: Artifact, Murmur, and Normal.

The project demonstrates an end-to-end machine learning workflow covering **audio preprocessing, feature extraction, data augmentation/resampling, model training, evaluation, and prediction**.

---

## Project Overview

Cardiopulmonary diseases can produce characteristic patterns in respiratory and heart sounds. This project explores the use of deep learning to automatically identify these patterns from audio recordings.

The audio signals are processed using **Librosa**, transformed into meaningful acoustic features, and then provided to recurrent neural networks capable of learning sequential patterns.

### Main Technologies

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Respiratory Sound Classification

## Objective

The respiratory sound model classifies lung sound recordings into six categories:

1. **Bronchiectasis**
2. **Bronchiolitis**
3. **COPD**
4. **Healthy**
5. **Pneumonia**
6. **URTI**

The original respiratory dataset was processed and **Asthma** and **LRTI** recordings were excluded from the classification pipeline.

## Dataset

The project uses the **Respiratory Sound Database** containing respiratory audio recordings together with patient diagnosis information.

The notebook combines:

* Patient diagnosis information
* Demographic information
* Audio recording metadata
* Respiratory sound recordings

Audio filenames are associated with patient number, recording index, chest location, acquisition mode, and recording equipment.

> **Note:** The original dataset is not included in this repository because of dataset size and licensing/distribution considerations.

## Audio Preprocessing

The respiratory audio pipeline performs:

1. Audio loading using Librosa
2. Sampling at **22,050 Hz**
3. Envelope-based noise/silence filtering
4. Audio cleaning
5. Acoustic feature extraction

### Extracted Features

The following audio features are extracted:

* **MFCCs** — 24 coefficients
* **Chroma features** — 24 coefficients
* **Mel spectrogram features**
* **Spectral contrast**
* **Tonnetz / tonal centroid features**

These features are concatenated to form the final feature representation.

The resulting feature vector contains **189 features**, which is reshaped to:

```text
(189, 1)
```

## Handling Class Imbalance

The respiratory dataset contains an imbalanced distribution between disease classes.

To address this, the notebook uses **SMOTE (Synthetic Minority Over-sampling Technique)** to increase representation of minority classes.

The COPD class is also reduced by randomly removing half of its samples after oversampling.

The processed data is then divided into training and testing sets using an **80/20 split**.

## Respiratory Deep Learning Model

The respiratory classification model uses a hybrid recurrent architecture:

```text
Input
  ↓
Bidirectional LSTM (128 units)
  ↓
Batch Normalization
  ↓
Dropout (0.4)
  ↓
Bidirectional GRU (64 units)
  ↓
Batch Normalization
  ↓
Dropout (0.4)
  ↓
Dense (128, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (6, Softmax)
```

### Training Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Input Shape      | `(189, 1)`               |
| BiLSTM Units     | 128                      |
| BiGRU Units      | 64                       |
| Dense Layer      | 128                      |
| Output Classes   | 6                        |
| Optimizer        | Adam                     |
| Loss Function    | Categorical Crossentropy |
| Batch Size       | 32                       |
| Epochs           | 125                      |
| Validation Split | 25%                      |

The trained model is saved as:

```text
working/fcn.h5
```

## Evaluation

The respiratory model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

The confusion matrix is normalized to analyze classification performance across the six respiratory categories.

The notebook also includes an example inference pipeline that loads the trained model, extracts features from an audio recording, and produces:

* Predicted diagnosis
* Class probabilities

---

# Heart Sound Classification

## Objective

The heart sound pipeline initially works with five heart sound categories:

1. **Normal**
2. **Murmur**
3. **Extrastole**
4. **Artifact**
5. **Extrahls**

For the final classification model, these categories are grouped into three classes:

| Original Categories | Final Class |
| ------------------- | ----------- |
| Artifact            | Artifact    |
| Murmur              | Murmur      |
| Normal              | Normal      |
| Extrahls            | Normal      |
| Extrastole          | Normal      |

Therefore, the final model performs **3-class classification**:

```text
Artifact
Murmur
Normal
```

## Audio Processing

Heart sound recordings are loaded at a sampling rate of **22,050 Hz**.

The preprocessing pipeline:

* Loads audio recordings
* Limits recordings to a maximum duration of **10 seconds**
* Pads shorter recordings
* Extracts MFCC features
* Applies time-stretching augmentation

### Feature Extraction

The model uses **52 MFCC features** per audio recording.

The extracted MFCC representation is reshaped into:

```text
(52, 1)
```

### Data Augmentation

Each heart sound recording is processed in three forms:

* Original audio
* Time-stretched audio at rate **0.8**
* Time-stretched audio at rate **1.2**

This increases the number of training examples and introduces variation into the training data.

## Handling Class Imbalance

The heart sound dataset contains a significant class imbalance.

The notebook calculates class weights based on the training class distribution:

```text
Artifact → 40
Murmur   → 129
Normal   → 409
```

Class weights are supplied during model training to give greater importance to underrepresented classes.

## Heart Sound Deep Learning Model

The heart sound classifier uses another hybrid BiLSTM–BiGRU architecture:

```text
Input
  ↓
Bidirectional LSTM (256 units)
  ↓
Batch Normalization
  ↓
Bidirectional GRU (128 units)
  ↓
Batch Normalization
  ↓
Dense (64, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (32, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense (3, Softmax)
```

### Training Configuration

| Parameter       | Value                    |
| --------------- | ------------------------ |
| Input Shape     | `(52, 1)`                |
| BiLSTM Units    | 256                      |
| BiGRU Units     | 128                      |
| Dense Layers    | 64 → 32                  |
| Output Classes  | 3                        |
| Optimizer       | Adam                     |
| Learning Rate   | `0.0001`                 |
| Loss Function   | Categorical Crossentropy |
| Batch Size      | 8                        |
| Epochs          | 100                      |
| Class Weighting | Yes                      |

The notebook also uses:

* Model Checkpointing
* CSV Logging
* TensorBoard logging

## Evaluation

The heart sound model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

The confusion matrix provides class-wise information for:

```text
Artifact
Murmur
Normal
```

---

# Overall Workflow

```text
                 AUDIO DATA
                     │
          ┌──────────┴──────────┐
          │                     │
     Lung Sounds            Heart Sounds
          │                     │
          ▼                     ▼
    Audio Cleaning       Audio Preprocessing
          │                     │
          ▼                     ▼
   Feature Extraction     MFCC Extraction
          │                     │
          ▼                     ▼
       SMOTE              Data Augmentation
          │                     │
          ▼                     ▼
     Train/Test Split      Class Weighting
          │                     │
          ▼                     ▼
      BiLSTM Layer          BiLSTM Layer
          │                     │
          ▼                     ▼
       BiGRU Layer           BiGRU Layer
          │                     │
          ▼                     ▼
      Dense Layers          Dense Layers
          │                     │
          ▼                     ▼
     6-Class Output        3-Class Output
          │                     │
          └──────────┬──────────┘
                     ▼
                 Evaluation
```

---

# Model Outputs

The project generates several evaluation outputs, including:

* Training/validation performance
* Accuracy and loss curves
* Confusion matrices
* Classification reports
* Predicted class probabilities

These outputs can be stored in the repository under a dedicated `results/` directory.

---

## Project Structure

```text

├── notebooks/
│   ├── respiratory-classify-bilstm-bigru.ipynb
│   └── heartbeat-sound-bigru-blstm.ipynb
│
├── results/
│   ├── respiratory_confusion_matrix.png
│   ├── heart_confusion_matrix.png
│   ├── respiratory_training_curves.png
│   └── heart_training_curves.png
│
├── requirements.txt
├── README.md
└── .gitignore

```

> Dataset files and large generated model/checkpoint files should only be committed if their size and redistribution permissions allow it.

---

# Installation

Clone the repository and install the required Python libraries.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries

```text
numpy
pandas
matplotlib
seaborn
librosa
scikit-learn
imbalanced-learn
tensorflow
keras
scipy
tqdm
jupyter
```

---

# Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks from:

```text
notebooks/
```

Run the respiratory and heart sound notebooks sequentially.

### Important

The notebooks currently contain local Windows dataset paths such as:

```text
F:\download\resp\...
G:\2024 projects\heartbeataudio\...
```

These paths must be changed to the location of the dataset on your system before running the notebooks.

---

# Results

The notebooks contain the actual evaluation outputs, including confusion matrices and classification reports.

For reproducibility and transparency, the repository should contain the **actual model metrics generated by the notebooks** rather than manually entered or estimated values.

> Model performance can vary depending on preprocessing, dataset organization, train/test split, hardware, and training configuration.

---

# Technologies Used

| Category         | Technologies              |
| ---------------- | ------------------------- |
| Programming      | Python                    |
| Deep Learning    | TensorFlow, Keras         |
| Audio Processing | Librosa                   |
| Data Processing  | NumPy, Pandas             |
| Machine Learning | Scikit-learn              |
| Imbalanced Data  | Imbalanced-learn / SMOTE  |
| Visualization    | Matplotlib, Seaborn       |
| Development      | Jupyter Notebook, VS Code |

---

# Key Machine Learning Concepts Demonstrated

This project demonstrates practical experience with:

* Audio signal processing
* Exploratory Data Analysis
* MFCC feature extraction
* Spectrogram analysis
* Chroma features
* Mel spectrograms
* Spectral contrast
* Tonal centroid features
* Audio augmentation
* SMOTE
* Class weighting
* Label encoding
* One-hot encoding
* Train/test splitting
* Bidirectional LSTM
* Bidirectional GRU
* Batch normalization
* Dropout regularization
* Multiclass classification
* Confusion matrix analysis
* Classification reports
* Model checkpointing
* TensorBoard logging

---

# Disclaimer

This project is developed for **educational and research purposes**.

The predictions generated by the models should **not be considered a medical diagnosis** or used as a substitute for professional medical evaluation.

The project demonstrates the application of deep learning techniques to audio classification and does not represent a clinically validated diagnostic system.

---


# Author

**Ezhil Arasi**

B.Tech – Computer Science and Engineering

---
