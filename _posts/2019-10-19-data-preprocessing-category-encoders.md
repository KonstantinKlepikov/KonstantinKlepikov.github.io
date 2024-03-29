---
layout: post
title: "Подготовка данных: кодирование категориальных признаков"
date: 2019-10-28
tags: preprocessing category-encoders sklearn scikit-learn ml-data
tag-for-collecting: category-encoders
keywords: подготовка данных машинное обучение препроцессинг катигориальные признаки machine learning category encoders
---

В статье «[особенности препроцессинга данных в scikit-learn]({{site.baseurl}}{% link _posts/2019-10-08-scikit-learn-preprocessing.md %})» разбирались особенности кодирования признаков с помощью библиотеки scikit-learn. К сожалению, набор инструментов scikit-learn довольно скромный.

Часто данные содержат множественные категориальные признаки с разными представлениями категорий. В некоторых случаях категорий оказывается сравнительно много, по отношению к общему объему данных. Иногда значения в категориальных признаках имеют различные распределения, могут иметь как явный, так и неочевидный порядок. Все это не добавляет энтузиазма во время предварительной обработки.

К счастью, есть готовые решения, например библиотека **[Category Encoders](https://contrib.scikit-learn.org/categorical-encoding/) (CE)**, предоставляющая широкий набор кодировщиков категориальных признаков.

## Какие преимущества у CE

1. Наверное, самое основное — это полная совместимость с scikit-learn. Доступны методы fit, fit_transform, get_params, set_params и transform. На основе CE можно строить пайплайны в scikit-learn.

2. Поддержка numpy и pandas. Что важно — pandas dataframe можно получить и на выходе кодировщика. Иногда это весьма полезно, особенно когда нужно выполнить выборочное кодирование. Это позволяет не городить самодельный забор из кодировщиков, а использовать CE непосредственно в последовательном пайплайне scikit-learn.

3. Есть возможность задать маску для кодирования, т.е. сделать выборочное кодирование

4. Можно отбросить часть данных

5. Спроектированный кодировщик отлично портируется на рабочие данные.

## Какие задачи можно решать с помощью CE

- кодирование номинальных признаков (nominal) — признаки, порядок которых не определен (часто в табличных данных к таким признакам относятся распределения по цветам или по городам)

- кодирование упорядоченных признаков (ordinal) — признаки, порядок которых можно считать определенным (например, распределение по яркости)

В CE можно кодировать бинарные признаки, упорядоченные по алфавиту или численному возрастанию признаки, а также географические, геометрические данные, данные о времени и другие структурированные данные. Единственное ограничение — решение не предоставляет собственного способа для создания масок для выборочного кодирования.

## Что имеется в наборе CE

### Contrast Coding

Данный тип кодирование разбивает признак на уровни (в каждом только значения, относящиеся к одной категории). Затем для каждого уровня вычисляется некоторая статистика. Например, вот так это делается в [statsmodels](http://www.statsmodels.org/dev/contrasts.html). Метод подходит для кодирования номинальных и частично упорядоченных признаков.

В CE реализованы следующие кодеры:

- Backward Difference Coding — сравнивается среднее для уровня со средним предыдущего уровня

- Helmert Coding — сравнивается среднее для уровня со средним для всех последующих уровней. Больше подходит для номинальных переменных.

- Sum Coding — сравнивается среднее для уровня со средним для всех остальных уровней

- Polynomial Coding — используются линейные, квадратичные и кубические представления целевой категории. Подходит исключительно для упорядоченных признаков, интервалы между категориями которых одинаковы. Ну и довольно прожорлив по памяти.

Не реализованы:

- Deviation Coding — более общий случай суммирующего кодирования, когда сравнение идет со всеми уровнями

- Dummy Coding — сравнение со средним значением на уровне

- Simple Coding — то же самое, что и Dummy Coding, только в качестве среднего принимается среднее всех значений фиксированного уровня

- Reverse Helmet Coding

- Forward Difference Coding

Больше подробностей [смотри тут](https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/)

### Target-based Coding

Для кодирования переменных используются сведения о разметке (цели) дата-сета. Тут надо знать следующие понятия: $$y$$ общее число примеров, $$y^+$$ число примеров, отнесенных к «положительному» классу, $$n$$ число примеров на уровне, $$n^+$$ число примеров уровня, отнесенных к положительному классу, $$\alpha$$ регуляризирующий параметр, $$prior$$ среднее значение цели. В CE реализованы:

- Target Encoder. Переменная кодируется по формуле $$x^k = prior*(1 - s)$$ $$+ s*\frac{n^+}{n}$$, где $$s = \frac{1}{1 + \exp(\frac{-n - mdl}{\alpha})}$$, а $$\scriptsize mdl$$ — минимум среди всех примеров на уровне.

- James-Stein Encoder кодируется по формуле $$x^k = (1 - B) * \frac{n^+ + prior*m}{u^+ + m}$$ $$+ B*\frac{y^+}{y}$$, где $$B$$ дополнительный гиперпараметр, регулирующий переобучение

- M-estimate кодируется по формуле $$x^k = \frac{n^+}{n} + B * \frac{y^+}{y}$$, где $$m = 1... 100$$ — дополнительный гиперпараметр, регулирующий переобучение. По сути — упрощенный вариант Target Encoder. (На момент написания статьи кодировщик работает некорректно)

- Weight of Evidence (WOE) считается по формуле $$x^k = \ln(\frac{nominator}{denominator})$$, где $$nominator = \frac{n^+ + \alpha}{y^+ + 2\alpha}$$, $$denominator = \frac{n - n^+ + \alpha}{y - y^+ + 2\alpha}$$. Кодировщик часто используется для подсчета рисков и других скорингов, т.к. по сути производит упорядочивание в логарифмическом масштабе. Хорошо решает проблему стандартизации и группировки разреженных данных. Проблема — теряются данные на биннинге, не учитывается корреляция между зависимыми данными и сам кодировщик легко может переобучиться.

Альтернативой WOE является PRE (Probability Ratio Encoding). В этом случае для каждой метки вычисляется $$P(1)/P(0)$$, т.е отношение вероятностей того, что объект размечен положительно и вероятность отрицательной разметки. Этим значением заменяется значение при кодировании. $$P(0)$$ естественно ни при каких условиях не принимается равным 0. В CE метод не реализован.

- Leave One Out (LOO) считается среднее цели для примера выбранной категории, для случая, когда пример удален из дата-сета.

- Catboost Encoder улучшенный LOO ([документация](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html))

У всех target-based кодировщиков имеет место проблема риска переобучения, так как используются данные о разметке дата-сета. Два варианта решения — дополнительная регуляризация и двойная кросс-валидация. Кроме того LOO и Catboost Encoder работают плохо, если реальные данные имеют другую размерность, что приводит к сдвигу по отноешнию к данным, на которых производилось обучение.

### Остальные кодировщики

В CE реализовано несколько базовых кодировщиков:

- One Hot — аналог OnHotEncoder в scikit-learn или get_dummies в pandas

- Ordinal только для упорядоченных признаков (обратите внимание при импорте, что класс в CE называется также, как OrdinalEncoder в scikit-learn)

- Binary конвертит категории признака в бинарный код. Категории конвертятся в номера по порядку, затем номера кодируются на двоичной базе, а затем единицы и нули из полученых кодированных значений выносятся в новые признаки. По сути, $$n$$ категорий переводятся в $$log_{2}n$$ бинаризированных признаков. Это весьма значительно сокращает количество признаков на выходе по сравнению с One Hot и полезно, когда число кодируемых категорий значительно.

- Base N комбинирует One Hot и Binary

Кроме того, реализован Hashing, позволяющий хешировать категории. Это аналог FeatureHasher (последний больше подходит для работы с текстом). Также походит для кодирования большого числа категорий, так как сокращает размерность на выходе по сравнению с One Hot.

### Чего нет в CE

Не реализован Label Encoding. Можно воспользоваться factorize из pandas или LabelEncoder из sklearn или Ordinal, что то же самое. Не реализован Frequincy Encoding, когда категории кодируются в числовом виде в зависимости от частоты для признака в диапазоне [0, 1]. Кроме того, нет методов для кодирования признаков, которые явно упорядочены циклически, например, дат и времени. Ну и, естественно, нет методов для конструирования признаков, таких как разделение выражения на части и т.д. Все это придется написать самостоятельно или искать более специфическое решение.

## Как использовать кодирование категориальных признаков

Есть множество разных рекомендаций. Все их можно свести к следующему:

1. Определить, относится ли вообще признак к категориальному и посчитать количество категорий.

2. Необходимо понять природу категориального признака, отнести его к номинальному или упорядоченному типу, установить циклические зависимости для категорий и другие особенности распределения категорий

3. Необходимо знать, к какому распределению данных лучше приспособлена обучаемая модель. Это позволит сразу выбрать наиболее оптимальный метод кодирования

4. Методы, использующие гиперпараметры, желательно сразу включить в цикл поиска оптимального решения.

5. Для частично упорядоченных признаков понять, можно ли соотнести категории с численными значениями, расположенными на эквивалентных интервалах. Если да, то стоит подумать о применении Ordinal Encoding. если нет, то можно попробовать один из методов контрастного кодирования. Если же количество категорий велико и есть проблемы с вычислительной сложностью по памяти, можно попробовать один из методов кодирования, сокращающий выходную размерность, например, хеширование. При этом надо помнить, что это может привести к потере данных.

6. Для неупорядоченных признаков можно использовать One Hot, если число категорий невелико. Большое количество категорий  можно также бинаризировать или хешировать, если потеря части данных приемлема. Если нет, придется использовать один из методов target encoding. Необходимо помнить о переобучении и построить, при необходимости, двойную кросс-валидацию.

7. На практике, на мой взгляд, можно миксовать различные методы, попутно расширяя пространство признаков. Например, можно сконструировать новые признаки, применить относительно простые кодировщики к части исходных, а на выходе закодировать весь пул по цели. Пространство для экспериментов ограничено, как всегда, только по вычислительной сложности.

Дополнительные статьи по этой тематике: [один](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02), [два](https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
