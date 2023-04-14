from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_gen import *
import graphviz


def espirales_entrenados(n, test_case):
    case = generar_espirales(n)
    clf = entrenar(case)
    case_clasificado = clasificar(test_case, clf)
    return case_clasificado


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
    results = X_test
    results["Class"] = clasificacion
    return results


def ejercicio_1():
    test_case = generar_espirales(10000)
    plot(test_case)
    plot(espirales_entrenados(150, test_case))
    plot(espirales_entrenados(600, test_case))
    plot(espirales_entrenados(3000, test_case))


def puntos_entrenados(values,test_case):
    clf = entrenar(values)
    return clasificar(test_case, clf)


def ejercicio_2():
    sizes = [125, 250, 500, 1000, 2000, 4000]
    training_set = []
    accuracy_results = []

    c = 0.78
    d = 2
    centros_a = centros_eja(d)
    centros_b = centros_ejb(d)

    test_case_a = generar_valores(centros_a, c * sqrt(d), d, 1000)
    test_case_b = generar_valores(centros_b, c, d, 1000)

    for i in range(len(sizes)):
        training_set.append([])
        accuracy_results.append([])

        for j in range(20):
            n = sizes[i]
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            values_clasificados = puntos_entrenados(values,test_case_a)
            training_set[i].append(values)
            class_clasificados = values_clasificados.iloc[:, -1:]
            class_test = test_case_a.iloc[:, -1:]
            
            results_training = class_clasificados.transpose().values.tolist()[0]
            results_actually = class_test.transpose().values.tolist()[0]

            accuracy = accuracy_score(results_training,results_actually)
            print(1-accuracy)

