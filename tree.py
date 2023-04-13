from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from data_gen import *


def ejercicio_1():
    test_case = generar_espirales(10000)
    plot(test_case)
    graficar_espirales_entrenados(150, test_case)
    graficar_espirales_entrenados(600, test_case)
    graficar_espirales_entrenados(3000, test_case)


def graficar_espirales_entrenados(n, test_case):
    case = generar_espirales(n)
    clf = entrenar(case)
    clasificar(test_case, clf)


def entrenar(case):
    X_train, y_train = case.iloc[:, :-1], case.iloc[:, -1:]
    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_impurity_decrease=0.005,
        random_state=0,
        min_samples_leaf=5,
    )
    clf.fit(X_train, y_train)
    return clf


def clasificar(test_case, clf):
    X_test = test_case.iloc[:, :-1]
    clasificacion = clf.predict(X_test)
    test_case["Class"] = clasificacion
    plot(test_case)
