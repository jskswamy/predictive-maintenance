# Predictive Maintenance MLOps Pipeline

[![ML Pipeline](https://github.com/jskswamy/predictive-maintenance/actions/workflows/pipeline.yml/badge.svg)](https://github.com/jskswamy/predictive-maintenance/actions/workflows/pipeline.yml)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/jskswamy/predictive-maintenance)

An end-to-end ML pipeline for predicting engine maintenance needs in commercial diesel vehicle fleets.

## Overview

This project implements a **predictive maintenance system** that analyzes engine sensor data to predict whether maintenance is required before a breakdown occurs. The system achieves **99.8% recall** (catches virtually all failures) with **63% precision**.

### Key Features

- **Real-time predictions** via Streamlit web interface
- **Automated CI/CD pipeline** with GitHub Actions
- **Model versioning** on HuggingFace Model Hub
- **Dataset versioning** on HuggingFace Datasets

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Data Registration | `model_building/register.py` | Upload raw sensor data to HuggingFace |
| 2. Data Preparation | `model_building/prep.py` | Feature engineering (22 features from 6 sensors) |
| 3. Model Training | `model_building/train.py` | Train AdaBoost classifier with recall gate |
| 4. Deployment | `hosting/deploy.py` | Sync to HuggingFace Spaces |

## Resources

| Resource | URL |
|----------|-----|
| **Live Demo** | https://huggingface.co/spaces/jskswamy/predictive-maintenance |
| **Dataset** | https://huggingface.co/datasets/jskswamy/predictive-maintenance-data |
| **Model** | https://huggingface.co/jskswamy/predictive-maintenance-model |

## Local Development

```bash
# Clone and setup
git clone https://github.com/jskswamy/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token

# Run pipeline
python model_building/register.py
python model_building/prep.py
python model_building/train.py
streamlit run deployment/app.py
```

## Model Performance

| Metric | Value |
|--------|-------|
| Recall | 99.78% |
| Precision | 63.2% |
| F2 Score | 0.917 |
| ROC-AUC | 0.998 |

**Threshold:** 0.316 (optimized for maximum recall)

## Business Impact

For a 100-truck fleet: **17 breakdowns prevented** annually, **$85,000-$170,000** in cost savings.

## License

MIT License

---
*Capstone Project for PGP-AIML program*
