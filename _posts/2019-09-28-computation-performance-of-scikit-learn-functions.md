---
layout: post
title: "Зависимость вычислений в scikit-learn от данных и модели"
date: 2019-09-28
tags: machine-learning algorithms time-complexity scikit-learn ml-data computation-performance
tag-for-sollecting: time-complexity
keywords: машинное обучение machine learning временная сложность time complexity алгоритмы scikit-learn производительность вычислений computation performance
---

## Производительность вычислений

В scikit-learn производительность вычисления предсказаний зависит от:

- количества фичей
- распределения и разреженности данных
- временной сложности алгоритма
- извлечения фичей

### Количество фичей

Библиотека хорошо оптимизирована под небольшие дата-сеты, поэтому количество фичей начинает оказывать значительное влияние на время предсказания для данных с 250+ фичами.

![Влияние количества фичей на время предсказания](../../../assets/img/280919-01.jpg "Влияние количества фичей на время предсказания")

### Распределение и разреженность данных

Вычисления по разреженным дата-сетам (более 90% значений) можно улучшить по времени с помощью ScyPy, т.к. библиотека оптимизирована по потреблению кеша CPU. Подробнее о том, как готовить разреженные данные, можно посмотреть в [документации ScyPy](https://docs.scipy.org/doc/scipy/reference/sparse.html)

### Временная сложность алгоритмов

С ростом временной сложности падает производительность вычислений алгоритма на предсказаниях. В примерах - зависимость задержки вычисления предсказания от временной сложности линейных моделей, SVM и ансамблей.

![Зависимость задержки вычислений предсказаний от временной сложности для линейных моделей](../../../assets/img/280919-02.jpg "Зависимость от временной сложности для линейных моделей")

![Зависимость задержки вычислений предсказаний от временной сложности для SVM](../../../assets/img/280919-03.jpg "Зависимость от временной сложности для SVM")

![Зависимость задержки вычислений предсказаний от временной сложности для ансамблей](../../../assets/img/280919-04.jpg "Зависимость от временной сложности для ансамблей")

В линейных моделях используется схожая решающая функция, поэтому время предсказаний для разных линейных моделей зависит примерно одинаково от временной сложности модели. В алгоритмах с нелинейными кернелами вычислительная производительность зависит от количества векторов (чем больше, тем производительность меньше). Задержка вычисления предсказания возрастает линейно для SVM (для регрессора и классификатора) с ростом числа саппортных векторов. Для ансамблей наибольшее значение имеет количество решающих деревьев и их глубина.

Про временную сложность построения алгоритмов и подробнее про сложность расчета предсказаний можно почитать в [отдельной статье]({{site.baseurl}}{% link _posts/2019-09-08-time-complexity-of-machine-learning-algorithms.md %}).

### Извлечение фичей

На самом деле, препроцессинг данных занимает самую значительную часть времени в выдаче предсказаний. Часто очистка и трансформация данных может увеличивать задержку выдачи предсказаний в сотни раз, поэтому этот процесс необходимо постоянно внимательно и аккуратно улучшать.

## Пропускная способность предсказаний

Еще одна важная метрика производительности алгоритма - это пропускная способность, определяющая количество предсказаний за заданное время. В примере пропускная способность различных алгоритмов scikit-learn.

![Пропускная способность различных алгоритмов scikit-learn](../../../assets/img/280919-05.jpg "Пропускная способность различных алгоритмов scikit-learn")

Основной способ повышения пропускной способности предсказывающей модели - это увеличение количества экземпляров модели и распределение запросов на предсказания между экземплярами.

Больше информации и примеров про производительность вычислений scikit-learn можно прочитать в [технической документации](https://scikit-learn.org/stable/documentation.html) (смотрите [раздел 7](https://scikit-learn.org/stable/modules/computing.html) user guide).