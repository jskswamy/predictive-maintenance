---
license: mit
task_categories:
  - tabular-classification
tags:
  - predictive-maintenance
  - iot
  - sensors
  - fleet-management
size_categories:
  - 1K<n<10K
---

# Predictive Maintenance Engine Sensor Dataset

Engine sensor readings from commercial diesel vehicles for predictive maintenance classification.

## Features

| Feature | Description | Unit |
|---------|-------------|------|
| Engine RPM | Engine revolutions per minute | RPM |
| Lub Oil Pressure | Lubrication oil pressure | bar |
| Fuel Pressure | Fuel delivery pressure | bar |
| Coolant Pressure | Cooling system pressure | bar |
| Lub Oil Temp | Lubrication oil temperature | °C |
| Coolant Temp | Engine coolant temperature | °C |
| Engine Condition | Target: 0=Normal, 1=Needs Maintenance | binary |

## Dataset Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| train | 75% | Model training |
| validation | 10% | Hyperparameter tuning |
| test | 15% | Final evaluation |

All splits are stratified by `Engine Condition` to maintain class balance.

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("jskswamy/predictive-maintenance-data")
train_df = dataset["train"].to_pandas()
```

## License

MIT License
