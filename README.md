# HeartGuard - Multimodal Heart Disease Detection System

A cloud machine learning project that combines three ML models to detect heart disease using ECG images and clinical data. Built with Django and designed for AWS deployment.

This is the web application repository. Trained models, notebooks, and datasets are in the separate repo:
**https://github.com/x24250511/HeartDisease_Pred_Model**

---

## What This Project Does

The system takes two types of input — ECG images and clinical patient data — and runs them through a fusion of three machine learning models to produce a diagnosis with actionable recommendations.

**Three models work together:**

1. **ECG Image CNN (ResNet18)** - Looks at an uploaded ECG image and classifies it as Normal or Abnormal. Trained on 179K+ ECG images. Test accuracy: 97%.

2. **PTB-XL XGBoost** - Uses patient metadata (age, sex, pacemaker status, diagnostic likelihoods from 21K+ PTB-XL records) to evaluate MI probability. Test accuracy: 92%.

3. **Clinical Tabular Random Forest** - Takes clinical measurements like blood pressure, cholesterol, heart rate, etc. and predicts cardiac risk level. Trained on 5,081 combined patient records. Test accuracy: 87.5%.

The three models feed into a decision fusion layer that produces one of four outcomes:
- Normal
- Abnormal – Monitoring Required
- Possible Myocardial Infarction
- High Risk Myocardial Infarction

Based on the outcome and the patient's specific values, the system generates prioritized medical recommendations.

---

## Architecture

```
User uploads ECG image and/or enters clinical data
                    |
                    v
          [ Django Web App ]
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   [ ECG CNN ]  [ PTB-XL ]  [ Tabular RF ]
   ResNet18     XGBoost      Random Forest
   Normal/      MI/          High Risk/
   Abnormal     Non-MI       Low Risk
        |           |           |
        +-----------+-----------+
                    |
                    v
        [ Decision Fusion Layer ]
                    |
                    v
    Diagnosis + Risk Score + Recommendations
```

**Three input modes are supported:**
- ECG image only → runs CNN + PTB-XL (2-model fusion)
- Clinical data only → runs Tabular RF (standalone risk assessment)
- Both ECG + clinical data → runs all 3 models (full fusion)

---

## Tech Stack

- Python 3.13
- Django 4.2
- PyTorch (ResNet18 CNN)
- XGBoost
- scikit-learn (Random Forest)
- Bootstrap 5
- PyMuPDF (PDF to image conversion)
- SQLite (dev) / PostgreSQL (production)
- AWS EC2 + S3 (deployment target)

---

## Project Structure

```
cloudmachinelearning/
├── manage.py
├── requirements.txt
├── saved_models/
│   ├── ecg_cnn_model.pth          # ResNet18 ECG classifier
│   ├── ptb_xgb_model.pkl          # PTB-XL XGBoost MI classifier
│   ├── ptb_label_encoder.pkl      # Label encoder for PTB model
│   └── tabular_model.pkl          # Random Forest risk model
├── cloudmachinelearning/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── heartguard/
    ├── models.py                  # Database models (ECGUpload, ECGResult)
    ├── views.py                   # Request handlers
    ├── services.py                # ML fusion pipeline + recommendations
    ├── forms.py                   # Upload and clinical data forms
    ├── urls.py                    # URL routing
    └── templates/heartguard/
        ├── base.html              # Base template with navbar
        ├── home.html              # Landing page
        ├── upload_create.html     # ECG upload form
        ├── clinical_create.html   # Clinical data form
        ├── upload_detail.html     # Results page with recommendations
        ├── upload_list.html       # Report history
        ├── login.html
        └── register.html
```

---

## Setup and Running Locally

```bash
# Clone
git clone https://github.com/x24250511/HeartGuard_application.git
cd HeartGuard_application/cloudmachinelearning

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision joblib scikit-learn xgboost Pillow PyMuPDF numpy

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```

Open http://127.0.0.1:8000, register an account, and start uploading ECGs or entering clinical data.

**Note:** The trained model files (.pth and .pkl) need to be in the `saved_models/` directory. If they are not included in this repo due to size, train them using the notebooks in the model repository linked above, then copy the output files to `saved_models/`.

---

## Datasets Used

All datasets and training notebooks are in: **https://github.com/x24250511/HeartDisease_Pred_Model**

| Dataset | Records | Used For |
|---------|---------|----------|
| Kaggle ECG Images | 179,000+ | CNN training (Normal vs Abnormal) |
| PTB-XL | 21,799 | XGBoost MI classification |
| UCI Heart Disease (4 hospitals) | 920 | Tabular model base |
| Fedesoriano Heart Failure Prediction | 918 | Tabular dataset expansion |
| Cardiovascular Disease (sulianova) | 70,000 (4,000 sampled) | Tabular dataset expansion |
| Statlog Heart | 270 | Tabular dataset expansion |
| **Combined tabular dataset** | **5,081** | Final RF training set |

---

## Base Paper

This project extends the work of:

> Garg, A., Sharma, B., & Khan, R. (2021). Heart disease prediction using machine learning techniques. IOP Conf. Ser.: Mater. Sci. Eng., 1022, 012046.

The original paper used KNN (86.9%) and Random Forest (82.0%) on the Cleveland dataset (303 samples). Our project extends this with multimodal fusion, expanded datasets, cloud deployment, and a recommendation engine.

---

## Team
Tejas Patil
MSC_Cloud_Computing_A – NCI,2026

---

## Disclaimer

This system is built for academic purposes only. It is not a substitute for professional medical diagnosis or advice.
