---
layout: post
title: "API scikit-learn"
date: 2019-08-27
tags: scikit-learn phyton sklearn api
tag-for-sollecting: scikit-learn
keywords: sklearn scikit-learn phyton машинное обучение machine learning data science API
---

API scikit-learn включает три основных интерфейса:

- estimator для построения и обучения моделей
- predictor для рассчета предсказаний
- transformer для предварительной обработки данных

### Estimators

Этот интерфейс является базовым для библиотеки. Все алгоритмы (включая обучение без учителя) scikit-learn используют этот интерфейс. Манипуляции с данными осуществляются с помощью метода fit, а все параметры задаются в виде строк.

````python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty="l1")
clf.fit(X_train, y_train)
````

Estimators используются не только для классических алгоритмов обучения (с учителем или без), но и для препроцессинга данных или извлечения важности признаков.

### Predictors

Данный интерфейс добавляет метод predict, который позволяет получить предсказания для алгоритмов обучения с учителем. Метод стандартно возвращает предсказанные метки для классификации или значения для регрессии.

````python
clf.predict(X_test)
````

### Transformers

Интерфейс создан для предварительной обработки данных, для чего используется метод transform. Метод доступен для препроцессинга, отбора и извлечения признаков и сокращения размерности.

````python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
````

Необходимо вызвать метод fit(), затем метод transform(). Естественно, можно в одну строку.

````python
scaler.fit(X_train)
X_train = StandardScaler().fit(X_train).transform(X_train)
````

## Дополнительные возможности API

### Meta-estimators

Часть алгоритмов (например, ансамбли) параметризуются с помощью более простых алгоритмов. В scikit-learn часть алгоритмов имплементирована как meta-estimators. К примеру, можно сделать обертку для логистической регрессии, заменив, таким образом, дефолтный алгоритм OneVsRest.

````python
from sklearn.multiclass import OneVsOneClassifier

ovo_lr = OneVsOneClassifier(LogisticRegression(penalty="l1"))
````

### Pipelines and feature union

API scikit-learn позволяет собирать собственный estimator из нескольких базовых. Это можно сделать с помощью Pipeline объекта или FeatureUnion. Pipeline объединяет разные оценочные функции в одну в виде последовательно выполняемой цепи функций. FeatureUnion объединяет функции таким образом, что их выходы конкатенируются. Оба объекта могут использоваться совместно.

````python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, KernelPCA
from sklearn.featute_selection import SelectKBest

union = FeatureUnion([("pca", PCA()),
                      ("kpca", KernelPCA(kernel="rbf"))])

Pipeline([("feat_union", union),
          ("feat_scl", SelectLBest(k=10)),
          ("log-reg", LogisitcRegression(penalty="l2"))
          ]).fit(X_train, y_train).predict(X_test)
````

### Model selection

Scikit-learn упрощает поиск оптимальных значений в пространстве гиперпараметров. Для этого используются две мета-функции: GridSearcCV и RandomizedSearchCV. Так же опциональными являются кросс-валидация и скор-функции, такие как AUC и среднеквадратичная ошибка. Пример поиска гиперпараметров для SVM-классификатора:

````python
from sklearn.gridsearch import GridSearchCV
from sklearn.svm import SVC
param_grid = [
    {"kernel": ["linear"], "C”": [1, 10, 100, 1000]},
    {"kernel": ["rbf"], "C": [1, 10, 100, 1000],
    "gamma": [0.001, 0.0001]}
]
clf = GridSearchCV(SVC(), param_grid, scoring="f1", cv=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
````

Больше подробностей в [документации](https://scikit-learn.org/) scikit-learn.
