# Lightweight Edge-AI Framework for Fault Diagnosis and Severity Estimation for PMSM Motor using Cross-Capacity Learning


M.Tech Embedded Systems project focused on predictive maintenance of industrial motors using fused current and vibration signals with lightweight Edge AI for STM32 deployment.

### Current Progress

- Current + vibration signal preprocessing
- Sensor fusion
- Master dataset preparation
- Lightweight 1D CNN baseline
- Initial accuracy: **91.13%**

### Dataset

- Motor ratings: 1 kW, 1.5 kW, 3 kW
- Input: 1024 × 4
- 3-phase current + vibration
- Classes: Healthy, CCSC, ITSC

### Upcoming

DS-CNN → TCN → Model Comparison → Hyperparameter Tuning → LOCO Validation → Quantization → Microcontroller Deployment
