---
title: Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.53.0
app_file: app.py
pinned: false
license: mit
---

# Predictive Maintenance Demo

Real-time engine maintenance prediction for commercial diesel fleets.

## How to Use

1. **Adjust sensor sliders** to input current engine readings
2. **View prediction** - Green (Normal) or Red (Maintenance Required)
3. **Check probability** - Higher values indicate greater maintenance urgency

## Input Sensors

| Sensor | Normal Range | Warning Signs |
|--------|--------------|---------------|
| Engine RPM | 600-2,200 | Irregular patterns at idle |
| Lub Oil Pressure | 2-5 bar | Low pressure = bearing wear |
| Fuel Pressure | 3-10 bar | Erratic = injector issues |
| Coolant Pressure | 1-3 bar | Low = leaks, pump failure |
| Lub Oil Temp | 70-90°C | High = oil breakdown |
| Coolant Temp | 70-100°C | High = overheating risk |

---
*Capstone Project for PGP-AIML*
