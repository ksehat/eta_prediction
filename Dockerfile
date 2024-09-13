FROM python:3.8-slim

WORKDIR /app

# COPY tensorflow-2.17.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /app/
COPY artifact /app/artifact
COPY logs /app/logs
COPY serving /app/serving

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5000

# CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "5000"]
CMD ["streamlit", "run", "serving/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=5000"]
