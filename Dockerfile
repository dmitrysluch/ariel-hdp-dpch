FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./dpch /app/dpch
# CMD ["sleep", "3600"]
CMD ["uvicorn", "dpch.server:app", "--host", "0.0.0.0", "--port", "8000"]
