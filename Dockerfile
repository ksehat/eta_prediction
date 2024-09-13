FROM python:3.8-slim

WORKDIR /app

COPY artifact /app/artifact
COPY logs /app/logs
COPY serving /app/serving

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8080

CMD ["waitress-serve", "--host=127.0.0.1", "--port=8080", "serving.api:app"]
