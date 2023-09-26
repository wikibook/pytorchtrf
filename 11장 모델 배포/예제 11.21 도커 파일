FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY container /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /app/app_flask.py

EXPOSE 8000

CMD ["python", "app_flask.py"]