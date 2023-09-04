ctr_project
==============================

В мире онлайн-рекламы кликабельность (Click-Through Rate или CTR) является очень важной метрикой для оценки эффективности рекламы. 
В связи с этим, системы предсказания кликов имеют большое значение и широко используются для спонсорского поиска 
и ставок в режиме реального времени (real time bidding).

В данном проекте мы построим production-ready пайплайн по предсказанию кликов пользователя для мобильной Web рекламы.
За основу возьмем данные соревнования Kaggle [Avazu CTR Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction/overview/description).


## Sem1. ML Pipeline
### Описание пайплайна
Пайплайн с моделью состоит из трех основных элементов
- `make_dataset`: чтения данных
- `features/build_transformers`: обработки признаков, в которую входят
  - `DeviceCountTransformer`, `UserCountTransformer`: трансформы для расчета количества 
рекламных объявлений на пользователя или девайс
  - `CtrTransformer`: трансформы, с помощью которых кодируем категориальные переменные средним CTR
- `model_fit_predict`: обучаем классическую модель `Catboost` `а на предсказание вероятности клика для данной сессии пользователя. 

Поскольку целью данного курса являются не сами эксперименты или файнтюнинг модели, а построение пайплайна
то будем исходить из предположения, что это некоторая готовая версия модели, и нас просят катить ее в прод.
Поэтому мы сосредоточим усилия на воспроизводимости, поддерживаемости, развертке и мониторинге.


### Установка 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Обучение модели
```bash
python ctr_project/train_pipeline.py --configs configs/train_config.yaml
```

### Юнит тестирование:
```bash
pytest
```

### Результаты
Наш пайплайн будет достаточно прямолинейным и будет содержать элементы описанные ранее.
Кастомные обработки фичей обернуты в формат `sklearn transformer` для единооборазия.

![img_5.png](imgs/img_5.png)


## Sem2. Reproducibility
В этом занятии мы затроним вопрос воспроизводимости экспериментов в ML. Рассмотрим 2 инструмента воспроизводимости экспериментов над моделями:
- [DVC](https://dvc.org/) : для версионирования данных и артефактов при помощи Git синтаксиса
- [MLFlow](https://mlflow.org/): для логирования экспериментов над моделью.

Также поднимем удаленное S3 объектное хранилище в VK Cloud.

### Установка 
```bash
pip install dvc
pip install mlflow
```

### Основные команды
В процессе занятия нам потребуются следующие команды для настройка MLFLow UI и DVC и
добавления данных в удаленное хранилище.
```bash
mlflow ui

# setup DVC
dvc init
dvc add data/raw/sampled_train_5m.csv
dvc add data/raw/sampled_train_50k.csv

# create and setup remote
dvc remote add s3 s3://sem3-repro/ctr-project-train/
dvc remote modify s3 endpointurl https://hb.ru-msk.vkcs.cloud 
dvc remote modify s3 region ru-msk

# push/ pull to remote
dvc push -r s3
dvc pull -r s3

dvc repro
```

### Результаты
После прогона пайплайн у нас должна залогироваться их история в списке экспериметнов.
![img.png](imgs/img.png)

В каждом эксперименте будет записан свой список метрик, параметров модели и артефактов.
![img_1.png](imgs/img_1.png)

Мы будем иметь возможность сравнивать между собой отдельные запуски для выбора наиболее оптимального.
![img_2.png](imgs/img_2.png)

При этом в удаленном хранилище будут записаны обучающие данные, `.pkl` модели и `.json` метрики. 
Артефакты с каждого прогона сохранены под своим собственным `md5` хэшом.
![img_3.png](imgs/img_3.png)

При этом `md5` хэши для каждого эксперимента будут доступны в файле `dvc.lock`, который обновляется после каждого нового
эксперимента. Его мы тоже логируем в MLFlow, что дает нам возможность всегда иметь ссылку на состояние пайплайна
в каждом эксперименте.

```
schema: '2.0'  
stages:  
  train:  
    cmd: python train_pipeline.py --config configs/train_config.yaml  
    deps:  
    - path: configs/train_config.yaml  
      hash: md5  
      md5: 37cbcef657312c588f872fae924d1c26  
      size: 969  
    - path: data/raw/  
      hash: md5  
      md5: 75f77c6ca378b83b4d199c58e68d213f.dir  
      size: 1762301353  
      nfiles: 8  
    outs:  
    - path: models/catclf.pkl  
      hash: md5  
      md5: bef120e799a37fc607c97ec475285368  
      size: 37030  
    - path: models/metrics.json  
      hash: md5  
      md5: 7cac56ee734d1e973b3a13c392cd15e8  
      size: 167
```

![img_4.png](imgs/img_4.png)





## Организация проекта
```
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
