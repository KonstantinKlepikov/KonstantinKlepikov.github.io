---
layout: post
title: "Краткий справочник по терминам статистики, которые пригодятся для data science. Часть 1"
date: 2019-12-15
tags: statistics data-science machine-learning
tag-for-collecting: statistics
keywords: Статистика data science машинное обучение machine-learning
---

В данной статье кратко пройду по терминологии из практической статистики и ее применеии в data science. Термины приводятся в упрощенном виде, исключительно для понимания общих принципов. Источник для данной компиляции — книга [Practical Statistics for Data Scientists](https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/), Andrew Bruce, Peter Bruce

## Основные оценки

**Среднее (mean)** — сумма значений, разделенная на их количество. $$\bar{x} = \frac{\underset{i}{\overset{n}\sum}\ x{\tiny i}}{\quad n \quad}$$

**Среднее усеченное (trimmed mean)** — среднее, считаемое после отбрасывания определенного количества значений с обоих концов последовательности. $$\bar{x} = \frac{\underset{i=p+1}{\overset{n-p}\sum}\ x{\tiny i}}{\quad n - 2p \quad}$$, где $$p$$ — количество отбрасываемых значений. Устраняет влияние предельных значений.

**Среднее взвешенное (weighted mean)** — среднее произведений всех значений на их веса, деленное на сумму всех весов. $$\bar{x}{\tiny w} = \frac{\underset{i=p+1}{\overset{n-p}\sum}\ w{\tiny i} x{\tiny i}}{\underset{i}{\overset{n}\sum}\ w{\tiny i}}$$. Используется для усреднения данных, имеющих внутреннюю неоднородность.

**Медиана (median)** — значение, расположенное посередине сортированного списка данных. Если число данных четное, за медиану принимается среднее арифметическое двух значений, которые делят сортированные данные на две половины. Взвешенная медиана (weighted median) — данные сортируются, затем вычисляется сумма весов таким образом, чтобы для левой и правой частей данных сумма весов была одинаковой.

**Выброс (outlier)** — значение данных, сильно удаленное от основного кластера значений. Оценка, на которую не влияют выбросы, называется робастной (robust). Медиана - робастная оценка.

На практике медиана будет встречаться в ML в масштабных задачах реже неробастных оценок. Обычно данных достаточно для построения несмещенных моделей и влияние выбросов не критично. «Усечение» — прием, который частенько можно встретить и используется он для повышения обобщающей способности модели.

## Вариабельность

**Отклонение (deviation)** — разница между наблюдаемым значением и оценкой центрального положения (например, средним).

**Среднее абсолютное отклонение (mean absolute deviation)** — среднее взятых по модулю (абсолютных) значений отклонений от центрального положения. $$mad = \frac{\underset{i=1}{\overset{n}\sum}\ \vert x{\tiny i} - \bar{x}\vert}{n}$$. Еще известно как «манхэттенская норма» или $$l1$$. Является робастной оценкой.

**Дисперсия (variance)** — среднее квадратических отклонений. $$var = \frac{\underset{i=1}{\overset{n}\sum}\ (x{\tiny i} - \bar{x})^2}{n}$$ Дисперсия чувствительна к выбросам.

**Стандартное отклонение (standard deviation)**  — квадратный корень из дисперсии или «евклидова норма» или $$l2$$ норма. Так же, как и дисперсия, чувствительно к выбросам.

Для вычисления несмещенной оценки отклонений используется деление на $$n-1$$ вместо $$n$$. На практике в ML не сильно важно — смещенная оценка или не смещенная, так как данных обычно очень много.

**Медианное абсолютное отклонение от медианы (median absolute deviation from median)** — робастная оценка вариабельности, медиана абсолютных значений отклонений от медианы.

### Порядковые статистики

**Размах (range)** — разница между самым большим и самым маленьким значением в наборе данных.

**Процентиль (percentile)** — значение, при котором $$p$$ процентов значений принимает данное значение либо меньше, а $$(100-p)$$ процентов — данное значение или больше.

**Межквартальный размах (IQR)** — разница между 25-м и 75-м процентилем.

Расчет процентилей — затратная задача, так как требует сортировки всех данных. Для больших наборов используется приблизительный процентиль (например, алгоритм Zhang-Wang-2007).

**Мода (mode)** — наиболее часто встречающееся значение в наборе данных.

**Математическое ожидание (expected value)** — среднее (взвешенное по вероятностям возможных значений) значение случайной величины. Очень упрощая, принимается мат.ожидание так: каждый исход события умножается на вероятность его наступления, эти значения суммируются. Более подробно во всех смыслах икрмин неплохо [описан в википедии](https://en.wikipedia.org/wiki/Expected_value).

### Корреляция и ковариация

Корелляция переменных имеет большое значение в построении моделей ML. Переменная x считается коррелирующей с переменной y, если изменение первой можно соотнести с изменениями второй. При оценке корреляции определяют направленность — переменные положительно кореллируют, если большим значениям x соответствуют большие значения y, а отрицательно, если наоборот.

**Коэффициент корреляции (correlation coefficient)** — метрический показатель, измеряющий степень связи переменных (в диапазоне от -1 до +1). При этом смещение в отрицательную область говорит об отрицательной корреляции, в положительную о положительной, а близость к 0 об отсутствии. Часто говорят, что наблюдается слабая или сильная корреляция. $$r = \frac{\underset{i=1}{\overset{n}\sum}\ (x{\tiny i} - \bar{x})(y{\tiny i} - \bar{y})}{(n-1) s{\tiny x} s{\tiny y}}$$. В данном случае мы имеем дело с коэфициентом корреляции Пирсона, где вверху учитывается сумма отклонений от среднего для двух переменных, а внизу произведение их стандартных отклонений.

**Ковариация** позволяет определить наличие связи между переменными, но не показывает насколько сильно они связаны. В формуле коэффициента корреляции Пирсона ковариация: $$\frac{\underset{i=1}{\overset{n}\sum}\ (x{\tiny i} - \bar{x})(y{\tiny i} - \bar{y})}{(n-1)} = cov$$

В предварительном анализе данных для построения моделей можно встретить **корреляционную матрицу** — двумерный массив, в котором строки и столбцы представлены переменными, а значение ячеек коэфициентами корреляции. Такая матрица позволяет установить нелинейную связь переменных, когда коэфциент корреляции становится бесполезным.
