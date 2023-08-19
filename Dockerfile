FROM python:3.9

RUN apt-get update
RUN apt-get install nano

WORKDIR /ctr_app
COPY . /ctr_app

COPY models/catclf.pkl models/catclf.pkl
COPY models/ctr_transformer.pkl models/ctr_transformer.pkl

RUN pip install -r requirements.txt

ENV PATH_TO_MODEL="models/model.pkl"
ENV PATH_TO_CTR_TRANSFORMER="models/ctr_transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]