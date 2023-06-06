---
layout: post
title: "Как преобразовать вложенные структуры JSON в массив Pandas на Python"
date: 2020-03-30
tags: ml-data phyton preprocessing machine-learning pandas json additional
tag-for-collecting: preprocessing
keywords: optimisation data preprocessing machine learning python json машинное обучение оптимизация подготовка данных pandas
---

Наиболее часто встречаемая проблема — это вложенные структуры данных, которые поставляются в JSON формате по API. Пример таких структур можно посмотреть в статье [подготовка и оптимизация данных для задач машинного обучения]({{site.baseurl}}{% link _posts/2020-03-04-data-preprocessing-and-compression-in-machine-learning.md %}). В большинстве случаев требуются дополнительные действия для того, чтобы развернуть вложения в двумерном виде. В этой статье я покажу как это сделать.

## Вложенные структуры JSON с которыми придется повозиться

Попробуем поработать с данными на примере публичного API Государственной Думы Российской Федерации. Сервис предоставляет сведения о членах обеих палат и законотворческой деятельности.

Вот пример ответа по запросу о списке депутатов и сенаторов (в данном случае :token и :app_token заменяются на ключи в реальном запросе):

```python
import requests

json_object = requests.get('http://api.duma.gov.ru/api/:token/deputies.json?app_token=:app_token')
```

Что содержалось в ответе?

```python
...
{'id': '99100491',
  'name': 'Абдулатипов Рамазан Гаджимурадович',
  'position': 'Депутат ГД',
  'isCurrent': False,
  'factions': [{'id': '72100020',
    'name': 'Депутатская группа "Российские регионы"',
    'startDate': '1996-01-16',
    'endDate': '1997-11-12'},
   {'id': '72100024',
    'name': 'Фракция Всероссийской политической партии "ЕДИНАЯ РОССИЯ"',
    'startDate': '2011-12-04',
    'endDate': '2013-01-27'}]}
...
```

В этом примере мы наблюдаем вложенный список по ключу factions, в который помещены два словаря, агрегирующих данные о вхождении депутата во фракции. Такое вложение развернуть сразу не получится. Ситуация осложняется еще и тем, что по части депутатов ключ factions отсутствует.

## Как преобразовать json в Pandas?

Мы можем разобрать json-объект с помощью [pandas.json_normalize](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html) — метода, который разбирает структурированные данные из JSON в табличный формат, а также [json.load](https://docs.python.org/3/library/json.html), который десериализует текст или байткод, содержащий json-документ в python-объекты. Применим к нашему json-объекта.

```python
import numpy as np
import pandas as pd
import json

extracted = pd.io.json.json_normalize(json_object.json())
```

Как и предполагалось, данные по factions остались неразвернутыми.

id | name | position | isCurrent | factions
--- | --- | --- | --- | ---
99100491 | Абдулатипов Рамазан Гаджимурадович | Депутат ГД | False | [{'id': '72100020', 'name': 'Депутатская группа "Российские регионы"', 'startDate': '1996-01-16', 'endDate': '1997-11-12'}, {'id': '72100024', 'name': 'Фракция Всероссийской политической партии "ЕДИНАЯ РОССИЯ"', 'startDate': '2011-12-04', 'endDate': '2013-01-27'}]

Что бы разобрать вложенный список из factions для начала избавимся от пропусков

```python
factions_with_id = json_object.json()
for i in factions_with_id:
    is_factions = i.get('factions')
    if not is_factions:
        factions_with_id.remove(i)
```

Теперь развернем factions относительно айдишников депутатов (meta='id'). Чтобы сохранить семантику в полученной таблице, будем использовать аргумент record_prefix='factions.', который добавим к вновь образуемым столбцам массива.

```python
factions_with_id = pd.io.json.json_normalize(factions_with_id,
                                             record_path='factions',
                                             meta='id',
                                             record_prefix='factions.',
                                             errors="ignore")
```

Нам удалось развернуть список словарей в полноценную таблицу:

```python
factions_with_id.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5031 entries, 0 to 5030
Data columns (total 5 columns):
factions.id           5031 non-null object
factions.name         5031 non-null object
factions.startDate    5031 non-null object
factions.endDate      5031 non-null object
id                    5031 non-null object
dtypes: object(5)
memory usage: 196.6+ KB
```

Теперь в исходном массиве удалим колонку factions и с помощью метода merge() добавим в массив развернутые данные.

```python
extracted.drop(['factions'], axis='columns', inplace=True)
summary = extracted.merge(factions_with_id, how='outer')
summary.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5703 entries, 0 to 5702
Data columns (total 8 columns):
id                    5703 non-null object
name                  5703 non-null object
position              5703 non-null object
isCurrent             5703 non-null bool
factions.id           5031 non-null object
factions.name         5031 non-null object
factions.startDate    5031 non-null object
factions.endDate      5031 non-null object
dtypes: bool(1), object(7)
memory usage: 362.0+ KB
```

Теперь все хорошо — данные извлечены и представлены в удобном, с точки зрения семантики, табличном виде:

id | name | position | isCurrent | factions.id | factions.name | factions.startDate | factions.endDate
--- | --- | --- | --- | --- | --- | --- | ---
99100491 | Абдулатипов Рамазан Гаджимурадович | Депутат ГД | False | 72100020 | Депутатская группа "Российские регионы" | 1996-01-16 | 1997-11-12
99100491 | Абдулатипов Рамазан Гаджимурадович | Депутат ГД | False | 72100024 | Фракция Всероссийской политической партии "ЕДИНАЯ РОССИЯ" | 2011-12-04 | 2013-01-27
