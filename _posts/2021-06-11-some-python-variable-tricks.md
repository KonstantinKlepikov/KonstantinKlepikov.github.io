---
layout: post
title: "Несколько трюков с переменными в python"
date: 2021-06-11
tags: python additional
tag-for-collecting: python
keywords: python
image: /assets/img/110621-01.png
---

![{{page.title}}](../../..{{page.image}})

Иногда, особенно при прототипировании или сборке каких-то нестандартных шаблонов, могут возникнуть специфические задачи, такие как конвертирование строк в имена переменных, конвертирование переменных в значения ключей и т.д. В этой статье я разберу эти задачи и их решения.

## Как конвертировать переменные python в ключи словаря

Задача выглядит так: надо собрать значения ключей словаря из названий переменных, а значения словаря - из их объектов.

```python
bird = 1
fish = 'f'
dict_of_animals = {}

# some code

print(dict_of_animals)
>>>
{'bird': 1, 'fish': 'f'}
```

Проблема эта была поднята на [stack overflow](https://stackoverflow.com/questions/3972872/python-variables-as-keys-to-dict). Мне понравилось такое решение:

```python
dict_of_animals.update({k : v for k, v in locals().copy().items() if k[:2] != '__' and k != 'dict_of_animals'})
```

Дело в том, что далеко не всегда у вас есть список переменных или желание его составлять, а в данном контексте мы просто смотрим в `locals()` и забираем все, что нам нужно.

## Как конвертировать строки в названия переменных

### Вариант №1 - globals()

Функция `globals()` в python возвращает словарь текущей глобальной таблицы символов. В глобальной таблице символов хранится вся информация, относящаяся к глобальной области видимости.

```python
user_input = "apple"
globals()[user_input] = 1
print(apple)

>>> 1
```

### Вариант №2 locals() аналогично

```python
user_input = "apple"
locals()[user_input] = 1
print(apple)

>>> 1
```

### №3 exec()

```python
name = 'my_brilliant_name'
exec("%s = %d" % (name, 'Ilona'))
print(my_brilliant_name)

>>> 'Ilona'
```

## Как конвертировать ключи словаря в переменные python

Скорее всего, вам придется решить и обратную задачу - импортировать словарь в другое пространство имен и перегнать ключи словаря в названия переменных, а значения - в объекты, на которые ссылаются переменные. Проблема разбирается [здесь](https://stackoverflow.com/questions/18090672/convert-dictionary-entries-into-variables-python). Решение просто как веник:

```python
dict_of_animals = {'bird': 1, 'fish': 'f'}
locals().update(dict_of_animals)
print(bird)
1
```

Прелесть в том, что вы можете импортировать сразу несколько разных словарей содержащих одинаковые ключи и не заботиться особо о поименовании переменных - значение будет иметь только порядок апдейта.

## Как импортировать модуль, название которого определяется при импорте

Нетривиальная задача, которая может быть решена по-разному. Мы воспользуемся готовым решением - [importlib](https://docs.python.org/3/library/importlib.html). Сильно синтетический пример:

```python
import importlib

# some code

if this:
    module = importlib.import_module('module.' + this + '.that')
else:
    import module.default as module
```

Кстати, если Вас интересует скорость конкатенации строк и вообще какие бывают способы, то вот тут можно [прочитать об этом](https://stackoverflow.com/a/38362140/15966204).

## Как определить тип операционной системы в python

И напоследок, если вы гоняете все это между машинами с разнымси осями или используете зависимые от операционной системы сборки, то все очень просто. К примеру, вот так можно запускать `selenium` под windows не прописывая пути к гекодрайверу, а кинув его в удобную нам папку.

```python
import os
from selenium import webdriver

if os.name == 'nt':
    driver = webdriver.Firefox(
        executable_path=r'C:\nowhere\gekodriver\geckodriver.exe'
        )
else:
    driver = webdriver.Firefox()
```

## Предостережение

Все не так уж и прозаично. `locals()` и `globals()` не всегда приносят то, что нам нужно, а игры с `exec()` могут быть небезопасны. К тому же вы наверное заметили, что мы просто косплеим настоящий импорт - при импорте происходит практически тоже самое, только явно. Мы же добавляем переменные в контекст неявно и линтер будет страдать, а мы вместе с ним. Делать так на самом деле не надо, но иногда... Собственно, that all folks! :)
