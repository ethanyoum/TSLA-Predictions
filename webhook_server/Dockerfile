FROM python:3.9-slim

# Install ping and netcat for diagnostics
RUN apt-get update && apt-get install -y iputils-ping netcat-openbsd && \
    pip install flask kafka-python

WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
