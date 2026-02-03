# 🛴 Micromobility Demand Forecasting with LSTM/GRU

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep learning system for predicting daily demand of shared micromobility vehicles (scooters, bicycles, mopeds) using LSTM and GRU recurrent neural networks.

## 📊 Project Overview

This thesis project develops a production-ready demand forecasting system for micromobility operators. The model predicts daily trip demand 7 days ahead, enabling optimal fleet allocation and operational planning.

**Key Features:**
- 🎯 **50% improvement** over Naive Baseline
- 🧠 Comparative analysis of **LSTM vs GRU** architectures
- 📈 Handles temporal patterns: Weekly Seasonality, Trends
- 🔄 Autoregressive component for robust predictions
- 📦 Production-ready inference pipeline for real-world deployment

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
pandas, numpy, scikit-learn, matplotlib, seaborn
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/micromobility-demand-forecasting.git
cd micromobility-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p data/raw data/processed models outputs
```

### Download Dataset

1. Download the dataset from [Austin Open Data Portal](https://data.austintexas.gov/Transportation-and-Mobility/Shared-Micromobility-Vehicle-Trips/7d8e-dm7r)
2. Place `Shared_Micromobility_Vehicle_Trips_2018-2022.csv` in `data/raw/`

### Run the Notebook
```bash
jupyter notebook scooter_demand.ipynb
```

Or use the pre-trained models for inference (see [Usage](#usage)).

---

## 📁 Project Structure
```
micromobility-demand-forecasting/
├── data/
│   ├── raw/                              # Original dataset
│   │   └── Shared_Micromobility_Vehicle_Trips_2018-2022.csv
│   └── processed/                        # Cleaned & aggregated data
│       └── final_daily_micromobility_data.csv
├── models/                               # Trained model artifacts
│   ├── lstm_demand_model.pth             # LSTM weights
│   ├── gru_demand_model.pth              # GRU weights
│   ├── feature_scaler.pkl                # StandardScaler for features
│   └── input_features.json               # Feature names list
├── outputs/                              # Forecasts & examples
│   ├── company_example_data.csv
│   └── lstm_forecast_next_week.csv
├── scooter_demand.ipynb                  # Main notebook
├── requirements.txt                      # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🎯 Usage

### 1. Training from Scratch

Run all cells in `scooter_demand.ipynb` sequentially:
```python
# Sections:
# 1. Importing libraries
# 2. Loading the Dataset
# 3. Data Filtering & Cleaning
# 4. Post-Cleaning Validation
# 5. Aggregated Daily Dataset
# 6. Visualisation of Aggregated Data
# 7. Building & Training the LSTM/GRU Model
# 8. Production Inference System
```

### 2. Inference with Pre-trained Models
```python
from scooter_demand import forecast_demand

# Prepare your company's historical data (90+ days)
# Required columns: Date, TotalTrips, TotalUniqueDevices, MedianDuration, MedianDistance

forecast = forecast_demand(
    csv_path='outputs/company_example_data.csv',
    model_type='lstm',  # or 'gru'
    plot=True
)

print(forecast)
# Output:
#        Date  Predicted_Trips  DayOfWeek
# 0  2022-04-05         3,547    Tuesday
# 1  2022-04-06         3,954  Wednesday
# ...
```

---

## 🧠 Model Architecture

### LSTM/GRU Features (11)

- **Temporal**: DayOfWeek_sin/cos, Month_sin/cos, Season_sin/cos
- **Autoregressive**: TotalTrips_Lag1, TotalTrips_Lag7
- **Supply**: TotalUniqueDevices
- **Trip Characteristics**: MedianDuration, MedianDistance

**Loss Function**: Huber Loss (SmoothL1)  
**Optimizer**: AdamW    
**Scheduler**: ReduceLROnPlateau

---

## 🛠️ Technical Details

### Dependencies
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
scipy>=1.10.0
tqdm>=4.65.0
```

### Hardware Requirements

- **Minimum**: 8 GB RAM, CPU-only
- **Recommended**: 16 GB RAM
- **Training Time**: 
  - LSTM: ~0.5-1.5 min (CPU)
  - GRU: ~1.0-3.0 min (CPU)

### Reproducibility

All experiments use `seed=42` for:
- Python random
- NumPy random
- PyTorch (CPU)
- PYTHONHASHSEED

---

## 📚 Dataset

**Source**: [Austin Shared Micromobility Vehicle Trips](https://data.austintexas.gov/Transportation-and-Mobility/Shared-Micromobility-Vehicle-Trips/7d8e-dm7r)

**Coverage**:
- **Time Period**: April 2018 - April 2022 (4 years)
- **Original Size**: 15M trips
- **After Cleaning**: 13.45M trips (scooters only)

**Columns Used**:
- ID, Device ID, Vehicle Type
- Start Time, End Time
- Trip Duration, Trip Distance
- Census Tract Start/End
- Council District Start/End

**Data Cleaning**:
- Removed invalid datetime entries
- Filtered OUT_OF_BOUNDS locations
- Removed extreme outliers (duration >12h, distance >50 miles)
- Filled missing days with forward-fill (conservative approach)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author**: Fotis Diamantidis  
**Email**: diamafo@gmail.com  
**LinkedIn**: [Here](https://www.linkedin.com/in/fotis-diamantidis-24b596200/)  
**University**: Democritus Univercity of Thrace (D.U.Th.)

---
