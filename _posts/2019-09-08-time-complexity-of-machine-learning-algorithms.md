---
layout: post
title: "Временная сложность алгоритмов машинного обучения"
date: 2019-09-08
tags: machine-learning algorithms time-complexity scikit-learn
tag-for-sollecting: machine-learning
keywords: машинное обучение machine learning временная сложность time complexity алгоритмы scikit-learn
---

**Временная сложность** $$T(n)$$ определяет количество операций, которые необходимо выполнить алгоритму для обработки входных данных объемом n.

Показатель сложности усредняется, но на практике необходимо исходить из худшего случая, при котором для обработки входных данных требуется максимальное количество операций. Для этого используется так называемая нотация **«O» большое**. Грубо говоря, $$O(n)$$ выражает доминантный член функции стоимости алгоритма в худшем случае.

Пример из [статьи в википедии](https://en.wikipedia.org/wiki/Time_complexity), демонстрирующий рост числа операций с ростом объема входных данных для алгоритмов разной временной сложности.

![time complexity](../../../assets/img/080919-1.jpg)

Далее в этой статье я попробую собрать данные о временной сложности различных реализаций ML/DL алгоритмов. В статье $$n$$ — количество признаков, $$m$$ — количество образцов.

## Алгоритмы из scikit-learn

**[Nearest Neighbors Algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)**. Временная сложность зависит от имплементации.

- brute force (наиболее нативная имплементация, дистанция считается между всеми точками дата-сета). Сложность при построении $$O(n*m^{2})$$. Сложность при запросе $$O(n*m)$$. Время расчета не зависит от структуры данных и количества «ближайших соседей».

- K-D tree. Сложность при построении $$O(n*m\log(m))$$. Сложность при запросе для небольшого числа фич $$n < 20$$ — $$O(n\log(m))$$. Для большего числа измерений, сложность возрастает до $$O(n*m)$$. Время расчета сильно зависит от структуры данных и растет с увеличением количества «ближайших соседей».

- Ball Tree. Сложность при построении $$O(n*m\log(m))$$. Сложность при запросе — $$O(n\log(m))$$. Время расчета сильно зависит от структуры данных и растет с увеличением количества «ближайших соседей».

Больше подробностей смотри в [документации](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms)

**[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**. Временная сложность $$O(m*n^{2,4})$$ — $$O(m*n^{3})$$ в зависимости от реализации.

**[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)**. Временная сложность $$O(m*n)$$.

**[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)**. Временная сложность $$O(m^{2}*n)$$ — $$O(m^{3}*n)$$. Алгоритм реализован на библиотеке [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

**[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)**. Временная сложность $$O(m*n)$$. Алгоритм реализован на базе библиотеки liblinear, который реализует [Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) (подробнее [тут](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf))

**[DecisionTree](https://scikit-learn.org/stable/modules/tree.html)**. Нахождение оптимального дерева является [NP-полной задачей](https://en.wikipedia.org/wiki/NP-completeness) и ее временная сложность $$O(\exp(m))$$.

Временная сложность построения сбалансированного бинарного дерева для наивной имплементации $$O(n*m\log(m))$$, сложность расчета по построенной модели — $$O(\log(m))$$. На практике сбалансированного дерева не получается, поэтому сложность построения дерева возростает до $$O(n*m^{2}\log(m))$$. В scikit-learn используется препроцессинг, что позволяет уменьшить итоговую сложность для всего дерева до $$O(n*m\log(m))$$. [Подробнее](https://scikit-learn.org/stable/modules/tree.html#complexity).

[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Обход дерева имеет временную сложность $$O(\frac{\log(m)}{\log(2)})$$, ну или $$O(\log{\scriptscriptstyle 2}(m))$$. Если осуществляется сравнение по всем признакам, то $$O(n*m\log(m))$$, поэтому важно определять количество сравниваемых признаков.

**[RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#parameters)**. $$O(M*m\log(m))$$, где $$M$$ — число решающих деревьев. Сложность можно уменьшить с помощью параметров.

**[Multi-layer Perceptron](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network)**. $$O(m*n*h^{k}*o*i)$$, где $$h$$ — количество нейронов, $$k$$ — число скрытых слоев, $$o$$ — количество выходов и $$i$$ — число итераций.

Статья будет дополняться...
