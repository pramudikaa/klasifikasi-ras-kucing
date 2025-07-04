FROM python:3.10-slim

# Install libGL dan libglib untuk OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"]