FROM python:3.9-slim

RUN pip install kafka-python pandas torch

WORKDIR /app
COPY main.py .
COPY ../../five_day_return.py .
COPY ../../tsla_attention.pt .

CMD ["python", "main.py"]
