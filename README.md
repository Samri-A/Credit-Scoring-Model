## Credit Scoring Business Understanding

### 1. Why Basel II Pushes Us Toward Interpretable, Well‑Documented Models  
Basel II requires banks to measure credit risk accurately and to explain that measurement to regulators and auditors. A model that is easy to understand—showing how each factor (age, income, past payment behaviour, etc.) affects risk—reduces audit time, speeds up regulatory approval, and builds trust with senior management. Clear documentation also makes future maintenance easier when regulations or business priorities change.

---

### 2. Why We Build a “Proxy” Default Variable—and the Risks Involved  
Our dataset does not carry an explicit “default = Yes/No” label. To train any supervised model, we therefore create a **proxy**—for example, classifying a customer who is 90 days past due as a “default.”  
*Benefits*:  
* Allows us to train and test the model with historical data.  
* Enables early identification of at‑risk accounts before a legal default occurs.  

*Risks*:  
* If the proxy is too strict or too lenient, the model may learn patterns that do **not** reflect real default behaviour.  
* Mis‑calibration can lead to lost revenue (rejecting good borrowers) or higher losses (accepting risky borrowers), and could attract regulatory criticism for unfair lending practices.

---

### 3. Choosing Between Simple and Complex Models in a Regulated Setting  

| Aspect | Simple, Interpretable Model (e.g., Logistic Regression + WoE) | Complex, High‑Performance Model (e.g., Gradient Boosting) |
|--------|--------------------------------------------------------------|-----------------------------------------------------------|
| **Transparency** | High—each variable’s impact is clear and can be explained in plain language. | Low—interactions are non‑linear and harder to justify. |
| **Regulatory Acceptance** | Usually straightforward; documentation and validation are well‑established. | Requires extra effort (e.g., model explainability tools) and may face more questions. |
| **Predictive Power** | Adequate but may miss subtle patterns. | Often higher accuracy and better capture of non‑linear relationships. |
| **Operational Risk** | Easier to monitor, back‑test, and recalibrate. | More complex to monitor; risk of unnoticed drift or overfitting. |
| **Implementation Cost** | Lower—both in development time and compute resources. | Higher—needs tuning, stronger governance, and potentially more powerful hardware. |

**Bottom line:** In highly regulated environments, a slight drop in predictive performance is often acceptable if it means the model is transparent, auditable, and easier to defend to regulators.

---


# Credit Scoring Model

A machine learning project for credit risk assessment using transactional data. This repository contains code, data, and notebooks for building, evaluating, and serving a credit scoring model, with a focus on regulatory compliance and interpretability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [API](#api)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project aims to develop a credit scoring model using historical transaction data. The model predicts the likelihood of default, supporting risk management and lending decisions. The workflow includes data processing, exploratory analysis, model training, and API deployment.

---

## Business Understanding

- **Regulatory Focus:** Basel II compliance, emphasizing interpretable and well-documented models.
- **Proxy Default Variable:** Since the dataset lacks an explicit default label, a proxy (e.g., 90 days past due) is used for supervised learning.
- **Model Choices:** Both simple (logistic regression) and complex (gradient boosting) models are considered, balancing transparency and predictive power.

For more details, see the [Business Understanding section](README.md) in this file.

---

## Project Structure

- `data/`  
  &nbsp;&nbsp;• `raw/`  
  &nbsp;&nbsp;• `processed/`  
- `notebook/`  
  &nbsp;&nbsp;• `1.0-eda.ipynb`  
- `src/`  
  &nbsp;&nbsp;• `api/`  
  &nbsp;&nbsp;&nbsp;&nbsp;• `main.py`  
  &nbsp;&nbsp;&nbsp;&nbsp;• `pydantic_models.py`  
  &nbsp;&nbsp;• `data_processing.py`  
  &nbsp;&nbsp;• `train.py`  
- `tests/`  
- `DockerFile`  
- `docker-compose.yml`  
- `requirement.txt`  
- `.github/`  
  &nbsp;&nbsp;• `workflows/`  
  &nbsp;&nbsp;&nbsp;&nbsp;• `ci.yml`  
-


---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Samri-A/credit-scoring-model.git
    cd credit-scoring-model
    ```

2. Install dependencies:
    ```sh
    pip install -r requirement.txt
    ```

3. (Optional) Start with Docker:
    ```sh
    docker-compose up --build
    ```

---

## Usage

- **Exploratory Data Analysis:**  
  Open and run [`notebook/1.0-eda.ipynb`](notebook/1.0-eda.ipynb) for data exploration and visualization.

- **Data Processing & Model Training:**  
  Use scripts in [`src/`](src/) such as [`src/data_processing.py`](src/data_processing.py) and [`src/train.py`](src/train.py).

    ```sh
    python src/data_processing.py
    python src/train.py
    ```

- **API Deployment:**  
  The scoring API is implemented in [`src/api/main.py`](src/api/main.py).  
  To run locally:
    ```sh
    uvicorn src.api.main:app --reload
    ```

---

## Testing

Run unit tests with:

```sh
python -m unittest discover tests