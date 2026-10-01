import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_default_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Not found '{MODEL_PATH}'. "
        "Place loan_default_model.pkl in the same folder as app.py."
    )

model_bundle = joblib.load(MODEL_PATH)
PIPELINE = model_bundle["pipeline"]
THRESHOLD = model_bundle["threshold"]
MODEL_NAME = model_bundle.get("model_name", "Unknown model")

print(f"Model loaded: {MODEL_NAME} (threshold={THRESHOLD:.3f})")

DEFAULT_ROW = {
    "Age": 43,
    "Income": 82466.0,
    "LoanAmount": 15000.0,
    "CreditScore": 700,
    "MonthsEmployed": 36,
    "NumCreditLines": 2,
    "InterestRate": 13.46,
    "LoanTerm": 36,
    "DTIRatio": 0.30,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Business",
    "HasCoSigner": "Yes",
}

FEATURE_COLUMN_ORDER = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]


def risk_label(prob: float) -> str:
    if prob < 0.20:
        return "LOW RISK"
    elif prob < 0.50:
        return "MEDIUM RISK"
    return "HIGH RISK"


def build_input_row(age, loan_term, loan_purpose, num_credit_lines,
                    employment_type, education, marital_status):
    row = DEFAULT_ROW.copy()

    row["Age"] = age
    row["LoanTerm"] = loan_term
    row["LoanPurpose"] = loan_purpose
    row["NumCreditLines"] = num_credit_lines
    row["EmploymentType"] = employment_type
    row["Education"] = education
    row["MaritalStatus"] = marital_status

    return pd.DataFrame([row])[FEATURE_COLUMN_ORDER]


def main_factors(age, loan_term, loan_purpose, num_credit_lines,
                 employment_type, education, marital_status):

    factors = [
        {
            "text": f"Loan term: {loan_term} months",
            "positive": loan_term <= 36,
        },
        {
            "text": (
                "Home loan purpose"
                if loan_purpose == "Home"
                else "Business loan purpose"
                if loan_purpose == "Business"
                else f"{loan_purpose} loan purpose"
            ),
            "positive": loan_purpose in ("Home", "Business"),
        },
        {
            "text": f"{num_credit_lines} existing credit line(s)",
            "positive": num_credit_lines <= 3,
        },
        {
            "text": f"Employment type: {employment_type}",
            "positive": employment_type == "Full-time",
        },
        {
            "text": f"Education: {education}",
            "positive": education in ("High School", "PhD"),
        },
        {
            "text": f"Marital status: {marital_status}",
            "positive": marital_status == "Married",
        },
        {
            "text": f"Age: {age} years",
            "positive": 25 <= age <= 55,
        },
    ]

    return factors


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        age = int(data["age"])
        loan_term = int(data["loan_term"])
        loan_purpose = str(data["loan_purpose"])
        num_credit_lines = int(data["num_credit_lines"])
        employment_type = str(data["employment_type"])
        education = str(data["education"])
        marital_status = str(data["marital_status"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input."}), 400

    # Basic validation
    if not 18 <= age <= 100:
        return jsonify({"error": "Age must be between 18 and 100."}), 400

    if loan_term <= 0:
        return jsonify({"error": "Loan term must be greater than 0."}), 400

    if num_credit_lines < 0:
        return jsonify({"error": "Number of credit lines cannot be negative."}), 400

    allowed_purposes = {"Home", "Business", "Education", "Auto", "Other"}
    allowed_employment = {"Full-time", "Unemployed", "Part-time", "Self-employed"}
    allowed_education = {"High School", "Bachelor's", "Master's", "PhD"}
    allowed_marital = {"Married", "Single", "Divorced", "Other"}

    if loan_purpose not in allowed_purposes:
        return jsonify({"error": "Invalid loan purpose."}), 400

    if employment_type not in allowed_employment:
        return jsonify({"error": "Invalid employment type."}), 400

    if education not in allowed_education:
        return jsonify({"error": "Invalid education value."}), 400

    if marital_status not in allowed_marital:
        return jsonify({"error": "Invalid marital status."}), 400

    try:
        input_df = build_input_row(
            age=age,
            loan_term=loan_term,
            loan_purpose=loan_purpose,
            num_credit_lines=num_credit_lines,
            employment_type=employment_type,
            education=education,
            marital_status=marital_status,
        )

        probability = float(PIPELINE.predict_proba(input_df)[0, 1])
    except Exception as exc:
        return jsonify({
            "error": (
                "Model failed to process the input "
            ),
            "detail": str(exc),
        }), 500

    flagged = bool(probability >= THRESHOLD)

    response = {
        "probability_percent": round(probability * 100, 1),
        "risk_label": risk_label(probability),
        "flagged": flagged,
        "threshold_percent": round(THRESHOLD * 100, 1),
        "factors": main_factors(
            age=age,
            loan_term=loan_term,
            loan_purpose=loan_purpose,
            num_credit_lines=num_credit_lines,
            employment_type=employment_type,
            education=education,
            marital_status=marital_status,
        ),
        "assessment": (
            "Lower predicted risk" if probability < 0.20 else
            "Moderate predicted risk" if probability < 0.50 else
            "Higher predicted risk"
        ),
        "model_name": MODEL_NAME,
        "top_features": [
            "LoanTerm",
            "LoanPurpose_Home",
            "NumCreditLines",
            "EmploymentType_Full-time",
            "EmploymentType_Unemployed",
            "Education_High School",
            "Education_PhD",
            "MaritalStatus_Married",
            "Age",
            "LoanPurpose_Business",
        ],
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
