---
license: mit
tags:
  - predictive-maintenance
  - tabular-classification
  - sklearn
  - adaboost
pipeline_tag: tabular-classification
---

# Predictive Maintenance Model

AdaBoost classifier for predicting engine maintenance needs based on sensor readings.

## Model Description

This model predicts whether a diesel truck engine requires maintenance (1) or is operating normally (0) based on 6 sensor inputs. It uses an AdaBoost ensemble with Decision Tree base estimators, optimized for **maximum recall** to minimize missed failures.

### Architecture

- **Algorithm:** AdaBoost Classifier
- **Base Estimator:** Decision Tree (max_depth=3)
- **Ensemble Size:** 383 estimators
- **Learning Rate:** 0.261

## Performance

| Metric | Value |
|--------|-------|
| **Recall** | 99.78% |
| Precision | 63.2% |
| F2 Score | 0.917 |
| ROC-AUC | 0.70 |

**Threshold:** 0.316 (optimized for maximum recall)

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="jskswamy/predictive-maintenance-model",
    filename="best_model.joblib"
)
model = joblib.load(model_path)
prediction = model.predict_proba(features)[:, 1]
needs_maintenance = prediction > 0.316
```

## Limitations

- Trained on Class 8 diesel truck data
- Requires all 6 sensor inputs
- Static prediction (no temporal patterns)

## License

MIT License
