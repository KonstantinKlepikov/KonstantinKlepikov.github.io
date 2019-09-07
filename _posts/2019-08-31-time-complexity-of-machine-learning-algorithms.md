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

Далее в этой статье я попробую собрать данные о временной сложности различных реализаций ML/DL алгоритмов.

**[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)** из пакета scikit-learn. Временная сложность $$O(n^{2,4})$$ — $$O(n^{3})$$ в зависимости от реализации (здесь и далее $$n$$ — количество признаков). Или $$O(m)$$, где $$m$$ — количество образцов.

**[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)** из пакета scikit-learn. Временная сложность $$O(m*n)$$. Алгоритм реализован на базе библиотеки liblinear, который реализует [Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) (подробнее [тут](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf))

**[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)** из пакета scikit-learn. Временная сложность $$O(m*n)$$.

**[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)** из пакета scikit-learn. Временная сложность $$O(m^{2}*n)$$ — $$O(m^{3}*n)$$. Алгоритм реализован на библиотеке [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

**[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)** из пакета scikit-learn. Нахождение оптимального дерева является [NP-полной задачей](https://en.wikipedia.org/wiki/NP-completeness) и ее временная сложность $$O(\exp(m))$$. Обход дерева имеет временную сложность $$O(\frac{\log(m)}{\log(2)})$$, ну или $$O(\log{\scriptscriptstyle 2}(m))$$. Если осуществляется сравнение по всем признакам, то $$O(n*m\log(m))$$, поэтому важно определять количество сравниваемых признаков.

Статья будет дополняться...
