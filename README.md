# TEONN: Neural Network for Transport Equation Approximation

![TEONN Architecture](docs/network_architecture.png) <!-- Добавьте схему архитектуры -->

Neural network for approximating 1D transport profiles from plasma physics simulations (ASTRA-based models).

## Key Features

- 🧠 Fully-connected neural network for transport profile prediction
- 🔧 Automated data preprocessing:
  - Input parameter normalization
  - Profile padding for fixed-length outputs
- 📊 Comprehensive evaluation metrics:
  - Regression: MSE, RMSE, MAE 
  - Statistical: R², Pearson R with p-value
- 💾 CSV export for predictions and analysis

## Performance Metrics

| Metric              | Value   |
|---------------------|---------|
| MSE                 | 183     |
| MAE                 | 1.2     |
| R² Score            | 0.988   |
| Pearson Coefficient | 0.999   |
| p-value             | 0.00    |

![Results](images/res.jpg)   <!-- График обучения -->

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
git clone https://github.com/Dota2Enjoyer-rab/TEONN-Neural-Network-for-approximation-of-transport-equation-.git
cd TEONN-Neural-Network-for-approximation-of-transport-equation-
pip install -r requirements.txt
```


![Example Prediction](docs/prediction_example.png) <!-- Пример предсказания -->

## Project Structure
```
TEONN/
├── docs/                   # Diagrams and visuals
├── data/                   # Training datasets
├── models/                 # Saved model weights
├── teonn/
│   ├── preprocessing.py    # Data handling
│   ├── network.py          # Core NN architecture
│   └── evaluation.py       # Metrics calculation
└── tests/                  # Unit tests
```

## References
1. ASTRA Transport Modeling Framework
2. Plasma Physics Conference 2023
3. Neural Networks for PDE Approximation (Journal of Comp. Physics)

## License
MIT © 2024 Plasma Research Group
