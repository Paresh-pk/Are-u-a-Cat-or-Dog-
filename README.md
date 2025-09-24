# are-u-a-cat-or-dog

> **MobileNetV2 + Keras & MLflow** — A simple, reproducible project to train a Convolutional Neural Network (CNN) that classifies cats vs dogs and tracks experiments with MLflow.
>
> Click the link below to try the live demo.\
> [🐾 Cat vs Dog Classifier](https://cat-dog-classifier-1-uxmn.onrender.com/)\
> Please note: it may take **1–2 minutes** for the web service to wake up from sleep mode (cold start). Once it’s running, predictions will load much faster.





---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Running the project](#running-the-project)
  - [1. Local MLflow Tracking (recommended for development)](#1-local-mlflow-tracking-recommended-for-development)
  - [2. GCP-Integrated MLflow Tracking (production-ready)](#2-gcp-integrated-mlflow-tracking-production-ready)
- [Training](#training)
- [Single-image inference example](#single-image-inference-example)
- [Tips & Notes](#tips--notes)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project demonstrates a complete workflow for building, training, and tracking a Convolutional Neural Network (CNN) to classify images of cats and dogs. It uses **TensorFlow/Keras** for model building and leverages **MLflow** for comprehensive experiment tracking, from local development to a production-ready cloud setup on Google Cloud Platform (GCP).

The core of the classifier is a **MobileNetV2** model with transfer learning. The project provides two distinct workflows: one for quick local experiments and another for a scalable, cloud-based setup using GCP services.

Two runnable notebooks are provided:

- `cnn-mlflow.ipynb` — local development: trains locally and logs to `./mlruns/` (default).
- `cnn-GCP-mlflow.ipynb` — production-style setup: configures MLflow to use Cloud SQL (Postgres) and Google Cloud Storage (GCS) for artifacts.

## Features

1. 🧠 **Transfer Learning**: Utilizes `tf.keras.applications.MobileNetV2` pretrained on ImageNet for robust feature extraction.
2. 🔄 **Data Pipeline**: Implements `ImageDataGenerator` for efficient on-the-fly data augmentation and loading.
3. 📊 **MLflow Integration**: Features seamless experiment tracking with MLflow's autologging for Keras, capturing metrics, parameters, and model artifacts automatically.
4. 💻 **Local & Cloud Workflows**: Supports both a local, file-based MLflow server and a remote, production-grade server on GCP integrated with Cloud SQL and GCS.
5. 🐾 **Sample Inference**: Includes a simple script to test the final trained model on a single image.

## Repository Structure

```
are-u-a-cat-or-dog/
├── README.md
├── requirements.txt
├── cnn-mlflow.ipynb
├── cnn-GCP-mlflow.ipynb
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/
│       ├── cats/
│       └── dogs/
└── mlruns/        # created automatically by MLflow when you run
```

## Prerequisites

- Python 3.8+
- pip
- (Recommended) virtual environment
- Git

## Installation

```bash
# Clone
git clone <your-repo-url>
cd are-u-a-cat-or-dog

# Create & activate virtual environment
python -m venv venv
# Windows:
# venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Suggested ****\`\`**** entries** (example):

```
tensorflow>=2.6
mlflow
numpy
pillow
scikit-learn
matplotlib
psycopg2-binary    # only required for Postgres backend
google-cloud-storage  # only required for GCS artifacts
```

## Dataset

Place your images under the `dataset/` folder with the following layout to use `flow_from_directory`:

```
dataset/training_set/cats/*.jpg
dataset/training_set/dogs/*.jpg

dataset/test_set/cats/*.jpg
dataset/test_set/dogs/*.jpg
```

You can download the provided dataset ZIP here:

`https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=1&st=gdn6pydw&dl=0`

Unzip into `dataset/` so the folder structure above is preserved.

## Configuration

Open the notebook you plan to run and update these variables as needed:

```python
DATA_DIR = "./dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "training_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
IMAGE_SIZE = (128, 128)  # recommended; example uses 64x64 for speed
BATCH_SIZE = 32
EPOCHS = 10
```

To point to a remote MLflow server, set the tracking URI:

```python
import mlflow
mlflow.set_tracking_uri("https://your-mlflow-server-url")
```

If using GCP artifacts, set Google credentials in your environment before starting the server or running the client code:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

## Running the project

### 1. Local MLflow Tracking (recommended for development)

This mode stores MLflow data in `./mlruns/` and is the easiest way to inspect runs locally.

```bash
# Start MLflow UI (from project root)
mlflow ui --backend-store-uri ./mlruns
# Open http://127.0.0.1:5000 in your browser
```

Open `cnn-mlflow.ipynb` in Jupyter Lab / Notebook, verify dataset paths, set hyperparameters, and run cells. The notebook uses `mlflow.keras.autolog()` to capture runs automatically.

### 2. GCP-Integrated MLflow Tracking (production-ready)

This notebook demonstrates steps to run MLflow with a Postgres backend (Cloud SQL) and GCS for artifacts. High-level setup:

1. Create a PostgreSQL instance (Cloud SQL) and a database for MLflow metadata.
2. Create a GCS bucket to store artifact files.
3. Create a Compute Engine VM or Cloud Run service to host the MLflow server.
4. SSH into the compute instance and run the following (example):

```bash
sudo apt update
sudo apt install python3.11-venv -y
python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install --upgrade pip
pip install mlflow psycopg2-binary

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://DBuser:PASSWORD@DBendpoint/DBname \
  --default-artifact-root gs://GS-bucket-Name
```

Make sure the compute instance has the appropriate IAM permissions (service account) to write to the GCS bucket. Typical roles include `roles/storage.objectAdmin` for artifact writing.

On your local machine or training environment, set the tracking URI:

```python
mlflow.set_tracking_uri("https://<MLFLOW_SERVER_HOST>:5000")
```

And set `GOOGLE_APPLICATION_CREDENTIALS` if needed.

## Training

Open either `cnn-mlflow.ipynb` (local) or `cnn-GCP-mlflow.ipynb` (GCP) and run the cells. Key points:

- The project uses `MobileNetV2` as the feature extractor and initially freezes the base model (`base_model.trainable = False`).
- Increase `IMAGE_SIZE` if you want better accuracy (recommended `>= 96x96`).
- Use `mlflow.keras.autolog()` near model compile/start of training to capture parameters and metrics automatically.

Example snippet to enable autologging:

```python
import mlflow
import mlflow.keras
mlflow.keras.autolog()

with mlflow.start_run():
    model.fit(...)
```

## Single-image inference example

Below is an example Python snippet to run inference on a single image using the saved Keras model:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

img = load_img("path/to/image.jpg", target_size=(128,128))
arr = img_to_array(img) / 255.0
arr = np.expand_dims(arr, axis=0)

model = load_model("path/to/saved_model.h5")
probs = model.predict(arr)[0]
label = "dog" if probs[0] > 0.5 else "cat"
print(label, probs)
```

If you logged a model with MLflow, you can load the model from the MLflow artifact URI or using the MLflow model API.

## Tips & Notes

- **Avoid duplicate logging**: `mlflow.keras.autolog()` logs many parameters and metrics. Logging identical parameter keys manually can cause collisions. Use `mlflow.set_tag()` or unique param names for manual logs.
- **Image size**: MobileNetV2 performs better with images `>= 96x96` or `128x128` — the notebook uses `64x64` as a fast example.
- **Freezing & fine-tuning**: Start with a frozen base model. For better accuracy, unfreeze the last few layers and fine-tune using a lower learning rate.
- **Batch size & epochs**: Tune `BATCH_SIZE` and `EPOCHS` according to your GPU/CPU.
- **Reproducibility**: Set random seeds (TF, NumPy, Python `random`) if you need reproducible runs.

## Results

- Trained on \~10,000 photos (example dataset) and obtained **\~93% accuracy** (reported example result). Your mileage will vary depending on dataset size, image quality, and hyperparameters.
- All experiments and models were tracked via MLflow (local `./mlruns/` or remote server when configured).

## Troubleshooting

- **"Artifact upload failed (GCS permission)"**: Ensure the MLflow server/service account has GCS write permissions (e.g., `roles/storage.objectAdmin`) and the `GOOGLE_APPLICATION_CREDENTIALS` is configured on the server.
- \`\`\*\* missing\*\*: The `mlruns` directory is generated automatically by MLflow when you run experiments locally. Ensure the process has write permissions.
- \`\`\*\* on \*\*\*\*`flow_from_directory`\*\*: Verify your `dataset/` folder layout and the paths passed to `ImageDataGenerator.flow_from_directory()`.
- **Postgres connection issues**: Verify your connection string and firewall/authorized networks on Cloud SQL. Use a Cloud SQL Proxy or authorize the MLflow host's IP.

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request with a clear description of the change.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Paresh — feel free to open an issue or message me via the project repository for help or improvements.

---

*Thank you for using ****\`\`****. Happy training!*

