---
layout: post
title: "Фишки pandas: часть 1"
date: 2021-01-19
tags: pandas python additional
tag-for-collecting: pandas
keywords: pandas, python
image: /assets/img/pandas01.jpg
---

![{{page.title}}](../../..{{page.image}})

Стартую серию заметок про различные проблемы и фишки, с которыми мне пришлось столкнуться в python библиотеке pandas. В этом "выпуске":

- метод pandas apply() не поддерживает аргумент axis. TypeError: \<lambda\>() got an unexpected keyword argument 'axis'

- ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

- как заменить все значения одного типа на какое-то одно значение в pandas

- TypeError: No matching signature found при реализации метода fillna()

- как отобразить в окне браузера все строки датафрейма в jupyter notebook

## Метод pandas apply() не поддерживает аргумент axis. TypeError: \<lambda\>() got an unexpected keyword argument 'axis'

Проблема банальна: вы применяете метод серии, а не фрейма:

```python
df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
df['col1'] = df['col1'].apply(lambda x: x * 2, axis=1)
```

```text
...
TypeError: <lambda>() got an unexpected keyword argument 'axis'
```

У версии метода для серии нет аргумента axis. Решение - или применять к фрейму или убрать аргумент axis

```python
df['col1'] = df[['col1']].apply(lambda x: x * 2, axis=1)
```

```python
df['col1'] = df['col1'].apply(lambda x: x * 2)
```

## ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

Ошибку можно получить при попытке использовать логические методы в pandas

```python
df = df[(df['col1']>0) and (df['col1']<2)]
```

```text
...
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

Проблема в том, что массивы numpy и pandas не имеют логических значений и связанно это с тем, что не всегда возможно установить когда объект True, а когда он False. В некоторых случаях мы можем считать, что массив True, только если он не нулевой длины, в некоторых мы можем ожидать, что он True, только если все объекты массива не False. Иными словами, так-как пользователи ожидают различное поведение, библиотека просто выбрасывает ValueError, чтобы предоставить пользователю самостоятельно определить какую именно логику необходимо применить при сравнении.

Как решить эту проблему зависит от контекста. Можно воспользоваться более явным сравнением, к примеру, логическими операторами &, \| и т.д.

```python
df = df[(df['col1']>0) & (df['col1']<2)]
```

## Как заменить все значения одного типа на какое-то одно значение в pandas

Задача часто появляется, когда, к примеру, нужно обнулить множество значений в таблице. Используем метод [mask](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html)

```python
df['col1'] = df['col1'].mask(df['col1'] >= 0, 666)
```

## TypeError: No matching signature found при реализации метода fillna()

Эта редкая проблема встречается при использовании метода fillna() с аргументом типа method='ffill'. Если вычисления не слишком затратные, можно реализовать решение с переводом в тип 'object':

```python
df[['this', 'that']] = df[['this', 'that']].astype('object').fillna(method="ffill").astype(np.float16)
```

Естественно, переопределите свой тип данных для выхода.

## Как отобразить в окне браузера все строки и колонки датафрейма в jupyter notebook

Иногда нужно посмотреть на результат преобразований и убедиться, что нет пропусков или получена ожидаемая структура данных. Jupyter notebook сокращает вывод. Решение можно получить через option_context()

```python
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
```

Кстати, обратите внимание как используется менеджер контекста для того, чтобы не изменять глобальные настройки отображения.
