---
layout: post
title: "Особенности препроцессинга данных в scikit-learn"
date: 2019-10-08
tags: machine-learning scikit-learn ml-data preprocessing
tag-for-sollecting: scikit-learn
keywords: машинное обучение machine learning scikit-learn preprocessing препроцессинг
---

В статье кратко раскрываются некоторые вопросы подготовки данных с помощью scikit-learn.

### Замена пропусков

Scikit-learn не поддерживает замену пропусков с разными значениями. Сначала придется последовательно перегнать все интересующие пропуски, к примеру, в *NaN*, а затем использовать инструменты препроцессинга.

[MissingIndicator](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html) позволяет сделать разметку пропусков. С помощью [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) можно выполнить замену. Поддерживаются четыре основных метода:

- mean
- most_frequent
- median
- constant (необходимо задать fill_value, чтобы не получить дефолтное значение)

Препроцессинг перегоняет данные в Numpy-формат, что означает потерю метаданных. Если это представляет проблему - препроцессинг делать средствами Pandas.

Есть еще несколько популярных методов, например, кластеризация с использованием K ближайших соседей и интерполяция. Оба метода не поставляются в scikit-learn. придется поискать реализации поверх.

### Полиномиальные признаки

Перед работой с [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) надо иметь представление о том, как будут обрабатываться пропуски, так как NaN поднимет ошибку, а 0 для полиномов всех степеней останется нулем.

Есть возможность создавать матрицу без степенных вариаций (для этого необходимо задать interaction_only=True).

Весь процесс очень затратен по оперативной памяти и на нем довольно легко столкнуться нехваткой, поэтому, если дата-сет большой, придется предварительно подумать как с ним работать.

### Категориальные признаки

Scikit-learn не поддерживает обработку категориальных признаков, только замену на численное представление. Вариантов два:

- [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) для численного представления без разделения на отдельные признаки (пропуски тоже кодируются в собственный отдельный класс)

- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) для численного представления с разделением на отдельные признаки с бинарными классами для каждого признака.

### Обработка численных признаков

Численные признаки могут можно дискретизировать с помощью scikit-learn и, таким образом, перегонять их в категориальные. Поддерживаются два основных способа:

- discretisation (или квантилизация или биннинг). Доступно в виде [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) с тремя методами: uniform (одинаковая длина бинов), quantile (одинаковое число точек в бинах), kmeans (значение определяется кластеризацией).

- binarisation с помощью [binarize](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html) — задается trashold, все что ниже или рано 0, все что выше 1.

## Препроцессинг

Собственно непосредственно сам препроцессинг описан в [разделе Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html). В scikit-learn это трансформация и нормализация данных. Делать это необходимо, так как многие алгоритмы чувствительны к выбросам, а так же распределению данных в выборке.

[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) центрирует данные, удаляет среднее значение для каждого объекта, а затем масштабирует, деля на среднее отклонение. $$x{\scriptstyle scaled} = \frac{x - u}{s}$$, где $$u$$ среднее, а $$s$$ отклонение.

[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) трансформирует признаки в выбранном диапазоне. $$x{\scriptstyle scaled} = \frac{x - \min(x)}{\max(x) - \min(x)}$$

[MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) трансформирует в диапазон $$[-1, 1]$$. Используется для центрированных вокруг нуля или разреженных данных. $$x{\scriptstyle scaled} = \frac{x}{\max(abs(x))}$$

[RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). Для данных, в которых много выбросов.

Для нормализации данных можно использовать [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html). Довольно часто это становится необходимым, когда алгоритм предсказывает, базируясь на взвешенных значениях, основанных на расстояниях между точками данных. Особенно актуально для классификации текста и кластеризации. В scikit-learn доступны три регуляризатора:

- l1
- l2
- MaxAbsScaler
