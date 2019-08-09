---
layout: post
title: "Метрики оценки для отбора моделей в scikit-learn"
date: 2019-08-09
tags: scikit-learn phyton sklearn
tag-for-sollecting: scikit-learn
keywords: sklearn scikit-learn phyton машинное обучение machine learning data science 
---

В scikit-learn можно воспользоваться встроенными метриками для отбора модели. Это можно сделать с помощью аргумента scoring, который доступен в том числе и для кросс-валидации и грид-серча. Список доступен в модуле sklearn.metrics.scorer

Выглядит это примерно так:

````python
In[3]:
from sklearn.metrics.scorer import SCORERS
sorted(SCORERS.keys())
Out[3]:
['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro',
'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error',
'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro',
'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall',
'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
````

В sklearn.metrics можно получить confusion_matrix. Это позволяет построить матрицу ошибок как для бинарной, так и для мультиклассовой классификации.

Для классификации лучше всего построены:

accuracy — дефолтная метрика для классификации

f1, f1_macro, f1_micro, f1_weighted определяют f-measure (гаромническое среднее precision и recall). f1 для бинарной классификации. Macro вычисляет f-measure для каждого класса и находит их взвешенное среднее, в независимости от размера класса. Micro считает общее количество falls positive, falls negative и true positvie для каждого примера по всем классам, а затем считает f-measure. Weighted - дефолтная метрика, считается f-measure для каждого класса, находит взвешенное с учетом количества данных в каждом классе.

precision = $$\frac{TP}{TP+FP}$$

recall =  $$\frac{TP}{TP+FN}$$

f = 2*$$\frac{recall*precision}{recall+precision}$$

roc_auc — площадь под ROC-кривой

average_precision — площадь под кривой precision/recall

Для регрессий имеют значение mean_squared_error — среднеквадратическая ошибка и median_absolute_error — средняя абсолютная ошибка.

Описание доступно на [странице документации](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules). Там же можно найти подробную документацию о применении скоринговых функций для классификации и регрессии, методы применения нескольких метрик, а так-же описание make_scorer ждя создания кастомных скоринговых функций с помощью make_scorer.
