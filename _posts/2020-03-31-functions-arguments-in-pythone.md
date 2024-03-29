---
layout: post
title: "Аргументы функций в Python 3.x. Шпаргалка"
date: 2020-03-31
tags: python preprocessing additional
tag-for-collecting: preprocessing
keywords: python
---

Статью я написал потому, что постоянно забываю как обращаться к аргументам функций в разных сложных случаях. Надоело гуглить — сделал себе шпаргалку. Для упрощения ситуации речь пойдет про python 3.x.

## Основные правила

- аргументы передаются через присваивание объектов именам локальных переменных функций

- при передаче объекты никогда не копируются автоматически

- присваивание аргументов внутри функции не затрагивает вызывающий код

- при выполнении функции аргументы функции становятся ее локальными переменными в области видимости функции

- имена аргументов функции и имена переменных в области видимости функции не совмещаются

- модификация изменяемого объекта внутри функции может затронуть вызывающий код

В примере аргументы присвоен объект, на который указывает переменная из другого контекста. Внутри функции произошло присваивание локальной переменной нового объекта, что не затронуло вызывающий код:

```python
def func(a):
  print(a)
  a = 100
  print(a)

a = 10
func(a)
print(a)
```

```python
10
100
10
```

В следующем примере видно, как изменение на месте изменяемого объекта внутри функции приводит к последствиям определенного рода.

```python
def func(a):
  print(a)
  a.append('that')
  print(a)

a = ['this']
func(a)
print(a)
```

```python
['this']
['this', 'that']
['this', 'that']
```

Оператор return поддерживает возврат множественных значений, упаковывая их в коллекцию. Естественно, это можно распаковать в вызывающем коде. Например, так:

```python
def func(a, b):
 x = a + b
 y = a - b
 return a, b

result = func(5, 3)
print(result)
x, y = func(4, 2)
print(x, y)
```

```python
(5, 3)
4 2
```

## Режимы сопоставления аргументов

При объявлении функции:

- обязательные аргументы будут сопоставляться слева направо с любыми переданными позиционными аргументами при вызове функции _def func{имя)_

- стандартные значения, которые будут присвоены аргументу, если в вызове функции аргумент не передавался _def func{имя=значение)_

- _def func(*имя)_ — собирает произвольное количество оставшихся позиционных аргументов в кортеж

- _def func{**имя)_ — собирает произвольное количество оставшихся ключевых аргументов в словарь

- _def func(*остальные, имя)_ или _def func(*, имя=значение)_ — аргументы, которые должны быть переданы только по ключевому слову

При вызове функции:

- позиционные аргументы сопоставляются слева направо _func{значение)_

- ключевые сопоставляются по имени аргумента _func{имя=значение)_

- _func(*итерируемый\_объект)_ — передает все объекты в итерируемом объекте как отдельные позиционные аргументы

- _func{**словарь)_ — передает все объекты в итерируемом объекте как отдельные ключевые аргументы

При объявлении функции аргументы должны указываться в следующем порядке:

- любые позиционные аргументы _имя_

- любые стандартные аргументы _имя=значение_

- форма _\*имя_ или _*_

- аргументы, которые должны передаваться только по ключевым словам _имя_ или _имя=значение_

- форма _\*\*имя_

При вызофе функции аргументы должны указываться в следующем порядке:

- любые позиционные аргументы _значение_

- комбинация любых ключевых аргументов _имя=значение_

- _*итерируемый\_объект_

- _**словарь_

При несоблюдении указанного порядка выводится синтаксическая ошибка. При этом сопоставляться аргументы будут в следующем порядке:

1. Присваивание неключевых аргументов по позиции слева направо (порядок следования имеет значение, так как сопоставление осуществляется по позиции)

2. Присваивание ключевых аргументов по совпадающим именам (порядок следования аргументов не имеет значения, так как сопоставление происходит по имени, а не по позиции)

3. Присваивание оставшихся неключевых аргументов кортежу *имя

4. Присваивание оставшихся ключевых аргументов словарю **имя

5. Присваивание стандартных значений не присвоенным аргументам

6. Проверка, передается ли каждому аргументу только одно значение — если нет, тогда возникает ошибка

7. Присваивание именам аргументов переданных для них объектов

При этом учитывается то, какие из аргументов могут быть присвоены только по ключевым словам. Аргументы с передачей только по ключевым словам должны указываться после \*, но не могут указываться после _\*\*аргументы_. Также, конструкция \*\* не может появляться в списке аргументов сама по себе.

## Примеры

```python
def func(a, b=2, c=3):
  pass

# a обязательный, остальные нет
# предоставим значение a по позиции
func(1)

# предоставим значение a по ключу
func(a=1)

# переопределим все три значения
func(4, 5, 6)

# передадим a и переопределим c
func(1, c=6)

# такой вариант вызовет ошибку
func(1, a=2)

# это тоже неверно
func(b=2, 1)
```

В данном примере все аргументы, которым присвоено стандартное значение, необязательны при вызове функции. Объекты можно передавать в функцию как по позиции, так и по ключу.

Примеры с произвольным количеством аргументов:

```python
def func(*args):
  print(args)
```

```python
>>>func():
()
>>>func(1):
(1,)
>>>func(1, 2):
(1, 2)
```

Тоже для ключевых аргументов:

```python
def func(**kwargs):
  print(kwargs)
```

```python
>>>func():
()
>>>func(a=l, b=2)):
{'a' : 1, 'b': 2}
```

Все вместе:

```python
def func(a, *args, **kwargs):
  print(a, args, kwargs)
```

```python
>>>func():
()
>>>func(l, 2, 3, a=l, b=2):
1 (2, 3) {'a' : 1, 'b': 2}
```

Распаковка аргументов при вызове функции:

```python
def func(a, b, c):
  print(a, b, c)
```

```python
>>>args = (1, 2, 3)
>>>func(*args)
1 2 3
```

```python
>>>kwargs = {'a': 1, 'b': 2, 'c': 3}
>>>func(**kwargs)
1 2 3
```

Естественно можно сочетать позиционные, ключевые и агрегированные аргументы

```python
def func(a, b, c, d, e):
  print(a, b, c, d, e)
```

```python
>>>func(1, *(2, 3), d=4, **{'e': 5})
1 2 3 4 5
```

Наконец, мы можем указать какие из аргументов могут быть использованы только как ключевые:

```python
def func(a, *b, c):
  print(a, b, c)
```

```python
>>>func(1, 2, c=3)
1 (2,) 3
```

```python
>>>func(1, 2, 3)
TypeError: func() missing 1 required keyword-only argument: 'c'
```

Кроме того, для аргументов «только по ключу» можно задавать стандартные значения:

```python
def func(a, *b, c=3):
  print(a, b, c)
```

Аргументы «только по ключу» должны быть записаны после формы с произвольным количеством позиционных аргументов _\*аргументы_ и перед формой с произвольным количеством ключевых аргументов _\*\*аргументы_.

```python
def func(a, *b, c=3, **d):
  print(a, b, c, d)
```

```python
>>>func(1, 2, 3, c=6, x=2, y=4)
1 (2, 3) 6 {'x': 2, 'y': 4}
```

```python
def func(a, *b, **d, c=3):
SyntaxError: invalid syntax
  print(a, b, c, d)
```

Важный аспект распаковки итерируемых аргументов в индивидуальные аргументы заключается в том, что есть возможность использовать генераторы в качестве аргументов. В данном случае мы распаковали значения генераторного выражения:

```python
def func(a, b, c):
  print(a, b, c)
```

```python
>>>func(*(i for i in range(3)))
0 1 3
```

Еще один фокус — распаковка итератора ключей:

```python
>>>some_dict = dict(a=1, b=2, c=3)

# распаковка словаря
>>>func(**some_dict)
1 2 3

# распаковка итератора ключей
>>>func(*some_dict)
a b c

# или так
>>>func(*some_dict.values())
1 2 3
```

## Изменения в Python 3.8

В python 3.8 добавлена возможность указывать аргументы, которые могут быть переданы только как позиционные. Это выполняется с помощью маркера / который указывает, что все аргументы слева от него передаются исключительно как позиционные.

```python
def func(a, b, c=3, /):
  print(a, b, c)
```

```python
>>>func(1, 2, 3)
1 2 3
```

```python
>>>func(1, 2, c=3)
TypeError
```

Статья подготовлена на основе книги «Изучаем Python» Lutz, Mark
