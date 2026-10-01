FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY loan_default_model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
