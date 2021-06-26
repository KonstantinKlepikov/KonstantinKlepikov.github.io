---
layout: post
title: "Namedtuple для тестов в python"
date: 2021-06-25
tags: python
tag-for-sollecting: python test-drived
keywords: python, test drived development, namedtuple, разработка на оснвое тестирования
image: /assets/img/25062021-01.jpg
---

![{{page.title}}](../../..{{page.image}})

В python collection есть невероятно удобный инструмент [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) и в этой маленькой заметке речь пойдет про то, как применять его в тестировании кода.

Собственно я не тестировщик, но метод разработки на основе тестирования (test drived development) использую. Часто возникает потребность в неких тестовых объектах, например юзерах, их айтемах или других объектах, которыми надо кормить функции и классы при их разработке. В принципе для этого подходят стандартные структуры данных в python, к примеру, словари:

```python
some_data = {
    'dataset1': [1, 2, 3, 4, 5],
    'dataset2': [1, 2, 3, 4, 5],
    }
```

Все бы хорошо, да только проблемы начинаются с ростом количества случаев, которые нам необходимо протестировать, чтобы заставить код заработать. Иногда это выглядит так:

```python
some_data1 = ...
some_data2 = ...
some_data3 = ...
...
some_data666 = ...
```

Еще хуже дела начинают обстоять, если словарь содержит ссылки на рассчитываемые на лету объекты данных или другие словари/списки, в которых мы можем что-то менять. Окончательно такие наборы начинают коллапсировать, когда в какой-то момент нам надо добавить в словарь совершенно новый ключ или изменить тип содержащихся данных.

Хорошо бы как-то обобщить эту историю. [Namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) подходит!

Что эта штука делает? Именованный словарь создает сабклас от tuple, который так же поддерживает итерацию и индексацию, является неизменяемым... а еще позволяет получить значения по ключу.

```python
from collections import namedtuple

FakeUser = namedtuple('user', 
                      ['id', 'email'], 
                      defaults=['example0@example.com']
                      )
```

Синтаксис немного необычный, но его довольно просто освоить. Мы создаем класс, а так же атрибуты класса. Атрибутам класса можно присвоить дефолтные значения (обратите внимание, как список дефолтных значений заполняется с конца). Что в итоге? Создаем инстанс:

```python
fake_user1 = FakeUser()
fake_user1

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: __new__() missing 1 required positional argument: 'id'
```

Отлично, никаких больше пропущенных в спешке значений и непонятных ошибок.

```python
fake_user1 = FakeUser(0)
fake_user1

user(id=0, email='example0@example.com')
```

Мы получили готовый инстанс класса за пять секунд с отлично организованной документацией. Теперь мы можем плодить нужные нам тестовые данные на месте, не занимаясь созданием громоздких структур для тестирования. Но и это еще не все.

Мы можем создать один инстанс, а потом подменить какие-то значения, если в этом есть необходимость - стандартная библиотека предоставляет такие методы.

```python
fake_user1 = fake_user1._replace(email='omnomnom0@example.com')
fake_user1

user(id=0, email='omnomnom0@example.com')
```

Можно даже из коробки получить словарь

```python
fake_dict1 = fake_user1._asdict()
fake_dict1

OrderedDict([('id', 0), ('email', 'omnomnom0@example.com')])
```

И с легкостью получить доступ к атрибутам

```python
fake_user1.email

'omnomnom0@example.com'
```

Ну а максимум выгоды можно получить, если учесть то, что namedtuple представляет из себя простой класс. Мы можем наследоваться от него, переопределять и добавлять атрибуты и даже добавлять методы, что становится крайне полезно при расширении модели данных.

```python
class MyNewFakeUser(FakeUser):

    def remove_email(self):
        if self.email == 'omnomnom0@example.com':
            self.email = None

my_new_fake_user = MyNewFakeUser(0)
my_new_fake_user.remove_email()


Traceback (most recent call last):
  File "stdn", line 5, in remove_email
    self.email = None
AttributeError: can't set attribute
```

Упс, мы не можем так сделать - у нас вроде как неизменяемый объект. Ок, это даже к лучшему - страхует нас от поспешных действий, приводящих к ошибкам уже в самих тестах.

```python
class MyNewFakeUser(FakeUser):

    def remove_email(self):
        if self.email == 'omnomnom0@example.com':
            return 'omnomnon!'
```


Осталось только поставить все это в тесты и жизнь налаживается! :)

```python
def make_something_and_return_email(user_dict):
    ...

result = make_something_and_return_email(
    fake_user1._asdict()
    )

assert result == fake_user1.email
```

Обязательно [посмотрите документацию](https://docs.python.org/3/library/collections.html#collections.namedtuple) - у конструкции namedtuple есть еще несколько полезных свойств.
