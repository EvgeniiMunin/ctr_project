FROM python:3.9

RUN apt-get update && apt-get install nano

WORKDIR /ctr_app
COPY . /ctr_app

RUN pip install --timeout 1000 -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]