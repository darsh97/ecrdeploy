FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]