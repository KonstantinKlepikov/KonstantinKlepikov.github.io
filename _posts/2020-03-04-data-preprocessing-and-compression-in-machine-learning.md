---
layout: post
title: "Подготовка и оптимизация данных для задач машинного обучения"
date: 2020-03-04
tags: ml-data phyton preprocessing machine-learning pandas json additional
tag-for-collecting: preprocessing
keywords: optimisation data preprocessing machine learning машинное обучение оптимизация подготовка данных pandas python
---

Эту статью я написал, как результат митапа, на котором мой доклад был посвящен проблемам оптимизации двумерных данных.

## Проблема неоптимизированных данных

С какими задачами можно столкнуться и зачем вообще питонисту работать с данными?

- компания собрала дату, «потому что это важно»

- люди, которые проектировали «склад» данных уволились, сошли с ума или умерли

- пришло время «извлечь из этого смыслы» и «применить кейсы»

Это означает, что потребуется разобрать авгиевы конюшни данных и построить какие-то модели машинного обучения. И тут-то все как раз и вспоминают про Вас. При этом есть несколько нюансов:

- абсолютно никто не в курсе что значат все эти забавные числа и строки из базы данных

- никого особо не интересует сколько это будет стоить в расчетах (кстати, как всегда, сделать надо было вчера)

- совершенно очевидно, что на эту задачу амазоновские инстансы за 100500 миллиардов долларов Вам не выделят

Часто самым первым шагом после извлечения массива, является построение некой сырой модели данных. Тут можно применять множество подходов, таких как, к примеру, разведочный анализ. Важно, что на этом этапе приходится обращаться к большому объему информации и производить определенные манипуляции вручную, что обходится дорого по времени. **Что делать?** Попробуем разобраться в приемах оптимизации на примере данных с kaggle-соревнования [data-science-bowl-2019](https://www.kaggle.com/c/data-science-bowl-2019).

![data-science-bowl-2019](../../../assets/img/040320-01.png "data-science-bowl-2019")

Цель соревнования рассматривать не будем, нас интересует только формат данных. Особенностью этой конкретной задачи было то, что она была ограничена весьма скромными вычислительными ресурсам. А это как раз наш случай. В задаче использовался датасет, содержащий почти 12 млн строк. Для наглядности я буду использовать только 200 тыс.строк и не стану приводить в данной статье исходники или каким-то образом исследовать данные, т.к. правилами конкурса запрещено использование датасет в целях, не связанных с соревнованием. Если Вам интересно разобраться непосредственно с задачей data-science-bowl-2019, Вы всегда можете сделать это самостоятельно на [kaggle.com](https://www.kaggle.com/c/data-science-bowl-2019)

После загрузки и формирования дата-фрейма посмотрим на данные с помощью метода info() библиотеки pandas

```python
df_train.info(max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 11 entries, installation_id to world
dtypes: int64(3), object(8)
memory usage: 16.8+ MB
```

Здесь мы сразу встречаем первую ошибку, которая довольно часто в дальнейшем стоит значительных временных затрат. Дело в том, что метод info() оценивает объем данных в дата-фрейме приблизительно — нам об этом намекает скромный значек "+" после значения задействованной памяти. Особенно этот перекос становится заметен, если данные содержат объекты с вложенными структурами. Посмотрим еще раз, но теперь глубже.

```python
df_train.info(memory_usage='deep', max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 11 entries, installation_id to world
dtypes: int64(3), object(8)
memory usage: 147.3 MB
```

Почти в десять раз больше! Очевидно, что 200 тыс. строк, которые составляют менее 2% всего объема предоставленных данных, весят слишком много и их обработка потребует ресурсных затрат. Построение же модели на всех данных может оказаться и вовсе невыполнимой задачей по потребляемой памяти.

Заглянем внутрь (в качестве примера здесь и далее показаны несколько строк и заголовки колонок)

installation_id | event_id | game_session | timestamp | event_data | event_count | event_code | game_time | title | type | world
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
0001e90f | 27253bdc | 45bb1e1b6b50c07b | 2019-09-06T17:53:46.937Z | {"event_code": 2000, "event_count": 1} | 1 | 2000 | 0 | Welcome to Lost Lagoon! | Clip | NONE
0001e90f | 77261ab5 | 0848ef14a8dc6892 | 2019-09-06T17:54:56.302Z | {"version":"1.0","event_count":1,"game_time":0,"event_code":2000} | 1 | 2000 | 0 | Sandcastle Builder (Activity) | Activity | MAGMAPEAK

Часть данных являются числовыми, часть объектами. Как минимум одна серия имеет неограниченно вложенную структуру. Нам явно нужно привести данные к некоему единому виду.

## Почему нам вообще может понадобиться приводить данные к какому-то виду

- большинство алгоритмов ML требует предварительной подготовки данных, например, приведение к числовому виду

- часто данные нужно дополнительно привести к определенному виду для того чтобы срослась математика, например, нормализовать или привести к логарифмическому масштабу

- машинное обучение точно плохо понимает неразобранные составные объекты

- для большинства алгоритмов сложность по пространству/времени в основном сосредоточена на построении модели. Это хорошо для продакшена, но сильно увеличивает расходы на поиск решения

Попробуем «разобрать» json-подобный объект, находящийся в колонке event_data. Для этого воспользуемся [pandas.json_normalize](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html), который разбирает структурированные данные из JSON в табличный формат, а также [json.load](https://docs.python.org/3/library/json.html), который десериализует текст или байткод, содержащий json-документ в python-объекты. Чтобы применить эти методы ко всей серии, воспользуемся [DataFrame.apply](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html) (метод не сильно быстрый, но нам требуется выполнить эту операцию всего один раз, поэтому, в данном случае, время смело приносим в жертву).

```python
# json unpack
df_train_extracted = pd.io.json.json_normalize(
    df_train['event_data'].apply(json.loads))
```

В результате у нас получилось довольно прилично новых признаков (приведено сокращенно)

```python
df_train_extracted.info(verbose=True)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 136 columns):
event_code                  int64
event_count                 int64
version                     object
game_time                   float64
description                 object
...
gate.column                 float64
gate.side                   object
dinosaur_weight             float64
dinosaur_count              float64
chests                      object
dtypes: float64(64), int64(2), object(70)
memory usage: 207.5+ MB
```

Проблема заключается в том, что нам не удалось полностью разобрать json-подобные объекты, т.к. на самом деле они не отвечают стандарту json. Например, вот эти колонки остались неразобранными:

flower | flowers
--- | ---
0.0 | [0, 0, 0, 0, 0]
0.0 | [0, 0, 0, 0, 8]
0.0 | [8, 8, 8, 7, 8]

| shels |
| --- |
| [2, 3, 1] |
| [2, 3, 2, 1] |
| [2, 3, 2, 3, 2] |

| bottles |
| --- |
| [{'color': 'blue', 'amount': 2}, {'color': 'blue', 'amount': 1}, {'color': 'purple', 'amount': 2... |
| [{'color': 'blue', 'amount': 2}, {'color': 'blue', 'amount': 1}, {'color': 'purple', 'amount': 2... |
| [{'color': 'blue', 'amount': 2}, {'color': 'blue', 'amount': 1}, {'color': 'purple', 'amount': 2... |

| castles_placed |
| --- |
| [] |
| [{'size': 3, 'position': {'x': 567, 'y': 484, 'stage_width': 1015, 'stage_height': 762}}] |
| [{'size': 3, 'position': {'x': 567, 'y': 484, 'stage_width': 1015, 'stage_height': 762}}, {'size... |

Мы видим списки с пропусками, пустые списки, списки неравной длины, списки словарей, а так-же списки с разным числом вложенных вловарей и даже кортежи списков. Все это возникло потому, что так когда-то кому-то было удобно складировать эти данные в базу данных. Теперь это придется разобрать вручную. Данная задача выходит за рамки этой статьи. Тут я всего лишь демонстрирую то, с чем зачастую можно столкнуться при обработке данных.

**Что еще можно встретить?** Строки, пустые множества и словари. Булевые и другие значение, в том числе записанные как строки или с ошибками, например так: 'NONE', 'nul', 'none', '0' и т.п., false и true, 'False', 'True', 'Yes', 'No' и т.д. - список сильно неполный. Все это требует детального исследования и разбора.

К тому же, не все данные стоит разбирать. Иногда можно встретить готовые векторы, например координаты или списки каких-то сгруппированных значений.

В итоге, что с памятью?

```python
df_train_extracted.info(memory_usage='deep', max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 136 entries, event_code to chests
dtypes: float64(64), int64(2), object(70)
memory usage: 543.8 MB
```

Мы еще сильнее потеряли в потреблении памяти! А данные еще копать и копать...

## Битва за память

На самом деле Pandas группирует и хранит "столбцы" блоками, разбитыми по типам. Иными словами float, int и objects хранятся раздельно, причем оптимизировано, без индексов. С числами все просто — столбцы в блоке объединяются в многомерный массив NumPy. При запросе значения происходит сопоставление индекса с массивом. С объектами немного сложнее. Все это означает, что [разные объекты](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dtypes) по-разному используют память.

Поработаем с разными типами объектов отдельно.

```python
numerics_part = df_train_extracted.select_dtypes(include=['number']).copy()
objects_part = df_train_extracted.select_dtypes(include=['object']).copy()
```

**Для начала разберемся с наименьшим злом — с числами**. В Pandas используются подтипы int8, int16, int32, int64, float16, float32, float64. В нашем случае в результате сравнения при извлечении, числа оказались в наиболее затратном по памяти формате.

```python
numerics_part.info(memory_usage='deep')
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 66 columns):
event_code                  200000 non-null int64
event_count                 200000 non-null int64
game_time                   196724 non-null float64
total_duration              34617 non-null float64
duration                    67500 non-null float64
...
end_position                4 non-null float64
gate.row                    832 non-null float64
gate.column                 832 non-null float64
dinosaur_weight             1547 non-null float64
dinosaur_count              1550 non-null float64
dtypes: float64(64), int64(2)
memory usage: 100.7 MB
```

Изменим эту ситуацию и вот как: получим минимальное и максимальное значение в серии, затем сравним его с машинными лимитами для типов Numpy, после чего заменим на наименьшее.

```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for col in df.columns:
    col_type = df[col].dtypes
    if col_type in numerics:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            # последовательно сравниваем от наименьшего инта начиная с np.int8
            # наверх и переопределяем тип для серии
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            # и т.д.
        else:
            # аналогично для float
```

Мы выполнили самопальный вариант понижающего преобразования, выиграв 74.4Mb.

```python
numerics_part.info(memory_usage='deep', max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 66 entries, event_code to dinosaur_count
dtypes: float16(61), float32(3), int16(2)
memory usage: 26.3 MB
```

Но у нас все еще есть проблема:

| flower |
| --- |
| 0.0 |
| 5.0 |
| 2.0 |

Оказывается, везде в датасете использовано число с плавающей точкой, хотя на самом деле никакой потребности в этом нет - в этих данных все числа целые. Исправим эту ситуацию:

```python
numerics_part.fillna(value=-1, inplace=True)
numerics_part = numerics_part.astype(int)
```

В данном случае мы заменили все пропуски в данных на -1, что открыло для нас возможность применить преобразование типов. Прием замены пропусков на число довольно часто встречается — обычно выбирают число, которое не встречается в данных, к примеру -9999. В нашем случае в данных вообще нет отрицательных чисел, поэтому мы просто взяли наименьшее кешированное, что бы сэкономить и тут. К тому же этот трюк окажется полезен в будущем. Смотрим на результат преобразования:

```python
numerics_part.info(memory_usage='deep')
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 66 columns):
event_code                  200000 non-null int16
event_count                 200000 non-null int16
game_time                   200000 non-null int32
total_duration              200000 non-null int32
duration                    200000 non-null int32
...
end_position                200000 non-null int8
gate.row                    200000 non-null int8
gate.column                 200000 non-null int8
dinosaur_weight             200000 non-null int8
dinosaur_count              200000 non-null int8
dtypes: int16(9), int32(4), int8(53)
memory usage: 16.6 MB
```

Итого, по числам нам удалось сэкономить 83.5% по памяти! Неплохо.

**Переходим к объектам**. Тип object в Pandas хранит строковое представление. Строки хранятся фрагментировано - значение в ячейке по сути является указателем. При этом резервируется много памяти и это для нас плохо.

```python
objects_part.info(memory_usage='deep', max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 70 entries, version to chests
dtypes: object(70)
memory usage: 443.1 MB
```

Pandas предоставляет подтип category, который отображает строковые данные на индекс в int, а это то, что нам нужно, т.к. данные будут храниться не в виде указателя, а в виде словаря, в котором целочисленным значениям сопоставлены уникальным значениям данных. Перегоним наши объекты в «категории»:

```python
# subtype categoty
def object_to_category(part):
    converted_objects_part = pd.DataFrame()
    unconverted_objects_part = pd.DataFrame()
    total = len(part)
    for col in part.columns:
        try:
            unic = len(part[col].unique())
            if (unic / total) < 0.05:
                converted_objects_part.loc[:,col] = part[col].astype('category')
            else:
                converted_objects_part.loc[:,col] = part[col]
        except TypeError:
            # unhashable objects can't be categorised
            unconverted_objects_part.loc[:,col] = part[col]
    return converted_objects_part, unconverted_objects_part
```

Мы отбросим часть объектов, т.к. договорились ранее не работать с нераспакованной частью json

```python
converted_objects_part, unconverted_objects_part = object_to_category(objects_part)
converted_objects_part.info(memory_usage='deep')
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 39 columns):
version                2345 non-null category
description            69717 non-null category
identifier             68610 non-null category
...
crystal_id             2090 non-null category
location               1180 non-null category
gate.side              832 non-null category
dtypes: category(39)
memory usage: 7.9 MB
```

Обратите внимание, что мы сами управляем какие именно объекты представлять в виде категорий через отношение количества уникальных значений к общему количеству значений в серии. Теперь посмотрим объем нераспакованных данных и сложим с оптимизированными распакованными:

```python
unconverted_objects_part.info(memory_usage='deep', max_cols=0)
```

```shell
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Columns: 31 entries, castles_placed to chests
dtypes: object(31)
memory usage: 193.0 MB
```

193.0MB + 7.9MB — нам удалось сэкономить 242.2MB, больше половины. Если мы применим наш метод ко всему дата-сету, то мы увидим, что не все объекты укладываются в наш формат категорий. часть объектов имеет слишком большое число уникальных значений (хешы, айдишники и т.д.) и есть смысл оставить их в объектах.

Кстати, при определении подтипа category, пропуски в списке заменяются на дефолтное значение -1. Это очень удачно совпало с принятым нами ранее решением :)

Отдельно стоит отметить колонку timestamp. Это временная метка и в неоптимизированном виде она тоже занимает избыточное пространство.

```python
df_train['timestamp']
```

```shell
0         2019-09-06T17:53:46.937Z
1         2019-09-06T17:54:17.519Z
2         2019-09-06T17:54:56.302Z
...
199997    2019-08-02T00:06:37.107Z
199998    2019-08-02T00:06:38.480Z
199999    2019-08-02T00:06:40.684Z
Name: timestamp, Length: 200000, dtype: object
```

Временную метку оптимизируем с помощью функции pandas [pd.to_datetime](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html). Параметр format позволяет задать тип представления временной отметки. Дефолтный вот такой: “%d/%m/%Y”. Теперь метка будет выглядеть так:

```python
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_train['timestamp']
```

```shell
0        2019-09-06 17:53:46.937000+00:00
1        2019-09-06 17:54:17.519000+00:00
2        2019-09-06 17:54:56.302000+00:00
...
199997   2019-08-02 00:06:37.107000+00:00
199998   2019-08-02 00:06:38.480000+00:00
199999   2019-08-02 00:06:40.684000+00:00
Name: timestamp, Length: 200000, dtype: datetime64[ns, UTC]
```

В этом оптимизированном виде мы сэкономим дополнительные 14Mb памяти.

В завершении операций по оптимизации необходимо данные сериализовать. Сделать это можно, к примеру, с помощью стандартного модуля shelve. Чуть позже, в отдельной статье я расскажу про сложности и пути их решения, которые могут возникнуть при сериализации.

### Что еще можно сделать

- распаковать вложенности

- не все нужно распаковывать - кое-что это готовые вектора, кое-что можно пересчитать в скаляры

- избавиться от разреженности либо оптимизировать разреженные данные

Кроме того, можно не ограничиться представлением, а закодировать столбцы цифрами. Есть методы в самом Pandas. Я рекомендую использовать внешний модуль category_encoders:

- позволяет кодировать категории в цифры разными методами (их там около 20)

- можно кодировать полуструктурированные и неструктурированные данные

- можно сильно сократить выход по памяти за счет target encoding

- работает в стиле scikit-learn пайплайнов

- возвращает как Numpy массив, так и Pandas датафрейм

Как кодировать категории с помощью category_encoders читайте в [этой статье]({{site.baseurl}}{% link _posts/2019-10-19-data-preprocessing-category-encoders.md %})

Ну и, наконец, можно перегнать все в Numpy! :)
