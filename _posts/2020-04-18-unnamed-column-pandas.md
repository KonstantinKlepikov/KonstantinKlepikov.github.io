---
layout: post
title: "Столбец Unnamed в датафрейме Pandas"
date: 2020-04-18
tags: pandas preprocessing phyton
tag-for-sollecting: preprocessing
keywords: python подготовка данных pandas csv read_csv
image: /assets/img/250720banner.png
---

![{{page.title}}](../../..{{page.image}})

Довольно частая проблема, с которой может встретиться начинающий аналитик или питонист при работе с Pandas - это появление столбца Unnamed при чтении какой-то структуры в датафрейм. В этой короткой заметке пишу о том, почему это происходит и как победить проблему.

Все дело в индексе по умолчанию при чтении или записи .csv файла!

Формируя дата-фрейм из какого-либо источника данных, необходимо не забывать сохранять его без индекса, вот так:

```python
df.to_csv('c:/test.csv', index=False)
pd.read_csv('c:/test.csv')
```

```python
   a  b  c
0  1  1  1
1  2  2  2
2  3  3  3
```

Если этого не сделать, при сохранении будет добавлен «безымянный» индекс по умолчанию:

```python
df.to_csv('c:/test.csv')
pd.read_csv('c:/test.csv')
```

```python
   Unnamed: 0  a  b  c
0           0  1  1  1
1           1  2  2  2
2           2  3  3  3
```

И тогда с этим придется бороться при чтении, вот так:

```python
pd.read_csv('c:/test.csv', index_col=0)
```

```python
   a  b  c
0  1  1  1
1  2  2  2
2  3  3  3
```

Либо путем выкидывания столбца inplace:

```python
this = pd.read_csv('c:/test.csv')
this.drop(['Unnamed: 0'], axis=1, inplace=True)
this
```

```python
   a  b  c
0  1  1  1
1  2  2  2
2  3  3  3
```

А это неудобно...
