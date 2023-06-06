---
layout: post
title: "Конвейеры трансформации и кастомные трансформаторы в scikit-learn"
date: 2019-08-02
tags: scikit-learn phyton sklearn additional
tag-for-collecting: scikit-learn
keywords: sklearn scikit-learn phyton машинное обучение machine learning data science
---

В scikit-learn есть специальный класс Pipeline, с помощью которого можно создавать конструктор определяющий последовательность из шагов, трансформирующих данные в нужном порядке.

Выглядит это примерно так:

````python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

transform_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ('pd_to_np', FunctionTransformer(PdtoNp, validate=False)),
    ])
````

Вызов метода fit() запускает всю цепочку и приводит к последовательному выполнению трансформаций.

В данном случае в конвейер включены два шага трансформации данных. Первый - базовый scikit-learn-овский метод стандартизации данных ([подробнее](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)). А второй — это кастомный трансформатор.

Самый простой способ создать собственный трансформатор - это импортировать FunctionTransformer из sklearn.preprocessing. FunctionTransformer отправляет X и опционально y (массив данных и массив меток), а так же пользовательские аргументы в указанную функцию и возвращает результат. В примитивном примере я трансформирую произвольные данные в numpy массив.

````python
from sklearn.preprocessing import FunctionTransformer

def PdtoNp(some_data):
    return some_data.values
````

Подробнее можно прочитать [базе знаний scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)

Чтобы создать более сложную конструкцию, можно реализовать собственный класс, а в нем три базовых метода scikit-learn — fit() (должен возвращать self), transform() (собственно сам трансформер) и fit_transform(). Для всех трех методов обязательным аргументом является X. Если унаследовать класс от [TransformerMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html), то fit_transformer() можно не задавать, а если добавить в качестве базового класса [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) и не реализовывать *args, **kwargs в конструкторе, то будут доступны методы get_params() и set_params()

````python
from sklearn.base import BaseEstimator, TransformerMixin

class That(BaseEstimator, TransformerMixin):
    def __init__(self, this = True):
        self.this = this
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        that = self.this
        return that

transformer = That(this=False)
my_data = transformer.transform(my_data)
````

Ну и, наконец, можно собирать несколько конвейеров трансформации в один, например, так:

````python
transform_pipeline1 = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

transform_pipeline2 = Pipeline([
        ('pd_to_np', FunctionTransformer(PdtoNp, validate=False)),
    ])

full_transform_pipeline = Pipeline([
        ('transform_pipeline1', transform_pipeline1)),
        ('transform_pipeline2', transform_pipeline1)),
    ])
````

Весь этот набор инструментов scikit-learn превращает обработку данных в довольно креативный и удобный для пользователя процесс.
