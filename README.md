# LendW

## Explainable Loan Default Risk Assessment

LendW is a lightweight web-based research prototype for estimating **loan default risk** using an **XGBoost** machine learning model.

The application provides a simple interface where users can enter selected loan application characteristics. The trained model then estimates the probability of default and presents the result as a risk category.

> **Important:** LendW is a research prototype. It does not make actual lending decisions and is not intended to provide financial advice.

---

## Overview

Loan default prediction is a common machine learning problem in financial risk assessment. LendW explores how a machine learning model can be used to estimate the likelihood that a loan application may result in default.

Rather than asking users to provide a large number of variables, the prototype focuses on the features that were identified as the most influential by the trained XGBoost model.

The current prototype uses the following top features:

| Rank | Feature                     | Importance |
| ---: | --------------------------- | ---------: |
|    1 | `LoanTerm`                  |      0.065 |
|    2 | `LoanPurpose_Home`          |      0.055 |
|    3 | `NumCreditLines`            |      0.054 |
|    4 | `EmploymentType_Full-time`  |      0.052 |
|    5 | `EmploymentType_Unemployed` |      0.050 |
|    6 | `Education_High School`     |      0.048 |
|    7 | `Education_PhD`             |      0.047 |
|    8 | `MaritalStatus_Married`     |      0.047 |
|    9 | `Age`                       |      0.043 |
|   10 | `LoanPurpose_Business`      |      0.040 |

Categorical features such as `LoanPurpose_Home` and `EmploymentType_Full-time` are represented in the application through their original categorical inputs. The model pipeline performs the corresponding preprocessing.

---

## Features

### 1. Loan Application Input

The current interface asks users for:

* **Age**
* **Loan Term**
* **Loan Purpose**
* **Number of Credit Lines**
* **Employment Type**
* **Education**
* **Marital Status**

The interface intentionally focuses on the features identified as highly influential by the model.

### 2. Default Probability

After submitting an application, the model returns an estimated probability of default.

For example:

```text
Probability of Default
24.7%
```

### 3. Risk Classification

The predicted probability is converted into one of three categories:

```text
LOW RISK
MEDIUM RISK
HIGH RISK
```

The prototype currently uses the model's configured classification threshold together with the application's displayed risk ranges.

### 4. Relevant Model Features

The application displays several input characteristics related to the model's important features to make the prediction easier to understand.

These explanations are intended as feature/context indicators. They should not be interpreted as causal explanations of why a particular applicant will or will not default.

---

## Technology Stack

### Backend

* **Python**
* **Flask**
* **Pandas**
* **Joblib**
* **XGBoost**

### Frontend

* **HTML**
* **CSS**
* **JavaScript**

### Deployment

* **Docker**

---

## Application Architecture

The application follows a simple client-server architecture:

```text
User
 │
 ▼
Web Interface
(index.html)
 │
 │ POST /predict
 ▼
Flask Backend
(app.py)
 │
 ▼
Input Preparation
 │
 ▼
Preprocessing Pipeline
 │
 ▼
XGBoost Model
 │
 ▼
Default Probability
 │
 ├── Risk Classification
 ├── Relevant Features
 └── Model Information
 │
 ▼
Web Interface
```

---

## Project Structure

The project is organized approximately as follows:

```text
LendW/
│
├── app.py
├── Dockerfile
├── loan_default_model.pkl
│
└── templates/
    └── index.html
```

### `app.py`

The Flask backend is responsible for:

* Loading the trained model bundle
* Receiving prediction requests
* Validating user input
* Preparing the model input
* Running XGBoost inference
* Calculating the predicted default probability
* Generating the risk classification
* Returning the prediction as JSON

### `templates/index.html`

The frontend provides:

* Loan application input form
* Risk prediction display
* Relevant feature information
* Error handling
* Responsive layout

### `loan_default_model.pkl`

This file contains the trained model bundle, including the prediction pipeline and configured classification threshold.

---

## Running the Application with Docker

### 1. Build the Docker Image

From the project directory:

```bash
docker build --no-cache -t lendw .
```

The `--no-cache` option forces Docker to rebuild the image without using cached build layers.

### 2. Run the Container

Run:

```bash
docker run -p 5000:5000 lendw
```

The application should then be accessible at:

```text
http://127.0.0.1:5000/
```

or:

```text
http://localhost:5000/
```

### 3. Stop the Container

Find the running container:

```bash
docker ps
```

Then stop it:

```bash
docker stop <container_id>
```

---

## Running Without Docker

If Python dependencies are installed locally, the application can also be started directly:

```bash
python app.py
```

The Flask server will run on:

```text
http://127.0.0.1:5000/
```

---

## Prediction API

LendW exposes a prediction endpoint:

```text
POST /predict
```

### Request

The endpoint expects JSON containing:

```json
{
  "age": 43,
  "loan_term": 36,
  "loan_purpose": "Business",
  "num_credit_lines": 2,
  "employment_type": "Full-time",
  "education": "Bachelor's",
  "marital_status": "Married"
}
```

### Example Response

```json
{
  "probability_percent": 24.7,
  "risk_label": "MEDIUM RISK",
  "threshold_percent": 50.0,
  "assessment": "Moderate predicted risk",
  "model_name": "XGBoost"
}
```

The exact probability and model metadata depend on the trained model.

---

## Model Interpretation

The feature importance values shown in this project represent the relative importance of features within the trained XGBoost model.

For example:

```text
LoanTerm = 0.065
```

means that `LoanTerm` had a higher contribution to the model's prediction process than the other features listed below it.

Feature importance **does not mean causation**.

For example, a high importance for `LoanPurpose_Home` does not necessarily mean that choosing a home loan causes a higher or lower probability of default.

---

## Risk Categories

The prototype currently presents the model output using three categories:

| Predicted Probability | Category    |
| --------------------: | ----------- |
|                 < 20% | Low Risk    |
|           20% – < 50% | Medium Risk |
|                 ≥ 50% | High Risk   |

These categories are presentation-level classifications used by the prototype and should not be interpreted as official credit-risk standards.

---

## Limitations

LendW has several limitations:

1. **Research prototype only**
   The system is not intended for real-world lending decisions.

2. **Model-dependent predictions**
   Prediction quality depends on the dataset, preprocessing pipeline, model training process, and model version.

3. **Feature limitations**
   The current interface exposes only selected influential features rather than every feature used during model training.

4. **Default values for hidden features**
   Features that are required by the existing model pipeline but are not exposed through the current interface are populated with predefined values.

5. **Feature importance is not causality**
   Model feature importance should not be interpreted as evidence that a feature directly causes loan default.

6. **Potential dataset bias**
   A machine learning model can reproduce patterns or biases present in its training data.

7. **No real financial decision-making**
   The system does not connect to financial institutions, process real loan applications, or handle real money.

---

## Intended Use

LendW is intended for:

* Machine learning demonstrations
* Financial technology research
* Loan default prediction experiments
* Explainable AI demonstrations
* Educational projects
* Hackathons and innovation challenges

It is **not intended** for:

* Automatically approving or rejecting real loans
* Making decisions about real applicants
* Providing financial advice
* Processing real financial transactions

---

## Disclaimer

LendW is a machine learning research prototype created for educational and experimental purposes.

The predicted risk is generated by a machine learning model and should not be considered financial advice, a credit decision, or a guarantee of future loan repayment behavior.

The application should not be used to make decisions about real individuals or real financial transactions without appropriate validation, governance, regulatory compliance, fairness assessment, and human oversight.

---

## Future Improvements

Potential future improvements include:

* SHAP-based individual prediction explanations
* Additional model evaluation metrics such as PR-AUC
* Model calibration
* Fairness and bias analysis
* Feature interaction visualization
* What-if scenario simulation
* Model comparison with Logistic Regression and other classifiers
* Improved input validation
* Model versioning
* Automated model monitoring
* More comprehensive preprocessing and feature handling

---

## License

This project is intended as a research and educational prototype. Add an appropriate open-source license to this repository if the project is intended to be publicly distributed.
