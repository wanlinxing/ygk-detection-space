FROM python:3.11-slim

RUN pip --no-cache-dir install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip --no-cache-dir install -r requirements.txt

CMD ["python", "app.py"]