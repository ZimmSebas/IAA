from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_gen import *
from matplotlib import pyplot as mpl


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


def clasifico_y_errores(values, test_case):
    clf = entrenar(values)

    values_clasificados_test = clasificar(test_case, clf)
    values_clasificados_training = clasificar(values, clf)

    class_clasificados_test = values_clasificados_test.iloc[:, -1:]
    class_test = test_case.iloc[:, -1:]

    class_clasificados_training = values_clasificados_training.iloc[:, -1:]
    class_values = values.iloc[:, -1:]

    results_training_test = class_clasificados_test.transpose().values.tolist()[0]
    results_real_test = class_test.transpose().values.tolist()[0]

    results_training_values = class_clasificados_training.transpose().values.tolist()[0]
    results_real_values = class_values.transpose().values.tolist()[0]

    test_accuracy = accuracy_score(results_training_test, results_real_test)
    values_accuracy = accuracy_score(results_training_values, results_real_values)

    return (1 - test_accuracy, 1 - values_accuracy, clf.tree_.node_count)


def plot_error_lines(results, labels, sizes):
    colors = ["red", "red", "blue", "blue"]
    line = ["-", "-", "-", "-"]
    markers = ["o", "v", "o", "v"]

    for i in range(len(results)):
        mpl.plot(
            sizes,
            results[i],
            color=colors[i],
            label=labels[i],
            linestyle=line[i],
            marker=markers[i],
        )

    mpl.xlabel("Sizes")
    mpl.ylabel("Error")
    mpl.legend()

    mpl.show()


def plot_tree_sizes(nodes, labels, sizes):
    colors = ["red", "blue"]
    line = ["-", "-"]
    markers = ["o", "o"]

    for i in range(len(nodes)):
        mpl.plot(
            sizes,
            nodes[i],
            color=colors[i],
            label=labels[i],
            linestyle=line[i],
            marker=markers[i],
        )

    mpl.xlabel("Sizes")
    mpl.ylabel("Size of Tree")
    mpl.legend()

    mpl.show()


def ejercicio_2():
    sizes = [125, 250, 500, 1000, 2000, 4000]
    accuracy_results_parallel_on_test = []
    accuracy_results_parallel_on_training = []
    accuracy_results_diagonal_on_test = []
    accuracy_results_diagonal_on_training = []
    accuracy_results = []
    node_sizes_parallel = []
    node_sizes_diagonal = []
    node_sizes = []
    labels = []

    c = 0.78
    d = 2
    centros_a = centros_eja(d)
    centros_b = centros_ejb(d)

    test_error_total = 0.0
    values_error_total = 0.0

    test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)
    test_case_b = generar_valores(centros_b, c, d, 10000)

    for i in range(len(sizes)):
        node_totals = 0

        for j in range(20):
            n = sizes[i]
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_a)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals_parallel = node_totals / 20

        accuracy_results_parallel_on_test.append(test_error_total)
        accuracy_results_parallel_on_training.append(values_error_total)
        node_sizes_parallel.append(node_totals_parallel)

    labels.append("Diagonal_on_test")
    labels.append("Diagonal_on_training")

    accuracy_results.append(accuracy_results_parallel_on_test)
    accuracy_results.append(accuracy_results_parallel_on_training)

    for i in range(len(sizes)):
        node_totals = 0

        for j in range(20):
            n = sizes[i]
            values = generar_valores(centros_b, c, d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_b)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals = node_totals / 20

        accuracy_results_diagonal_on_test.append(test_error_total)
        accuracy_results_diagonal_on_training.append(values_error_total)
        node_sizes_diagonal.append(node_totals)

    accuracy_results.append(accuracy_results_diagonal_on_test)
    labels.append("Parallel_on_test")
    accuracy_results.append(accuracy_results_diagonal_on_training)
    labels.append("Parallel_on_training")
    node_sizes.append(node_sizes_parallel)
    node_sizes.append(node_sizes_diagonal)

    plot_error_lines(accuracy_results, labels, sizes)
    plot_tree_sizes(node_sizes, labels, sizes)


def ejercicio_3():
    c_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    accuracy_results_parallel_on_test = []
    accuracy_results_parallel_on_training = []
    accuracy_results_diagonal_on_test = []
    accuracy_results_diagonal_on_training = []
    accuracy_results = []
    node_sizes_parallel = []
    node_sizes_diagonal = []
    node_sizes = []
    labels = []

    n = 250
    d = 5
    centros_a = centros_eja(d)
    centros_b = centros_ejb(d)

    test_error_total = 0.0
    values_error_total = 0.0

    for c in c_values:
        node_totals = 0

        test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)
        test_case_b = generar_valores(centros_b, c, d, 10000)

        for j in range(20):
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_a)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals_parallel = node_totals / 20

        accuracy_results_parallel_on_test.append(test_error_total)
        accuracy_results_parallel_on_training.append(values_error_total)
        node_sizes_parallel.append(node_totals_parallel)

    labels.append("Diagonal_on_test")
    labels.append("Diagonal_on_training")

    accuracy_results.append(accuracy_results_parallel_on_test)
    accuracy_results.append(accuracy_results_parallel_on_training)

    for c in c_values:
        node_totals = 0

        test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)
        test_case_b = generar_valores(centros_b, c, d, 10000)

        for j in range(20):
            values = generar_valores(centros_b, c, d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_b)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals = node_totals / 20

        accuracy_results_diagonal_on_test.append(test_error_total)
        accuracy_results_diagonal_on_training.append(values_error_total)
        node_sizes_diagonal.append(node_totals)

    accuracy_results.append(accuracy_results_diagonal_on_test)
    labels.append("Parallel_on_test")
    accuracy_results.append(accuracy_results_diagonal_on_training)
    labels.append("Parallel_on_training")
    node_sizes.append(node_sizes_parallel)
    node_sizes.append(node_sizes_diagonal)

    plot_error_lines(accuracy_results, labels, c_values)
    plot_tree_sizes(node_sizes, labels, c_values)


def clasificador_dist_centro(centros, test_case):
    clasificacion = []

    for v in test_case:  # Re-trasponer.
        dist_centro_0 = dist(v, centros[0])
        dist_centro_1 = dist(v, centros[1])
        if dist_centro_0 >= dist_centro_1:
            clasificacion.append(0)
        else:
            clasificacion.append(1)

    results_real_values = test_case.iloc[:, -1:].transpose().values.tolist()[0]

    print(results_real_values)
    print(clasificacion)
    test_accuracy = accuracy_score(clasificacion, results_real_values)

    return test_accuracy


def ejercicio_3_1():
    c_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    accuracy_results_parallel_on_test = []
    accuracy_results_parallel_on_training = []
    accuracy_results_diagonal_on_test = []
    accuracy_results_diagonal_on_training = []
    accuracy_results = []
    labels = []

    n = 250
    d = 5
    centros_a = centros_eja(d)
    centros_b = centros_ejb(d)

    test_error_total = 0.0
    values_error_total = 0.0

    for c in c_values:
        test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)

        for j in range(20):
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            (test_error) = clasificador_dist_centro(centros_a, test_case_a)
            test_error_total += test_error

        test_error_total = test_error_total / 20

        accuracy_results_parallel_on_test.append(test_error_total)

    labels.append("Ideal_on_test")

    accuracy_results.append(accuracy_results_parallel_on_test)
    accuracy_results.append(accuracy_results_parallel_on_training)

    for c in c_values:
        test_case_b = generar_valores(centros_b, c, d, 10000)

        for j in range(20):
            values = generar_valores(centros_b, c, d, n)
            (test_error, values_error) = clasificador_dist_centro(values, test_case_b)
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals = node_totals / 20

        accuracy_results_diagonal_on_test.append(test_error_total)
        accuracy_results_diagonal_on_training.append(values_error_total)

    accuracy_results.append(accuracy_results_diagonal_on_test)
    labels.append("Parallel_on_test")
    accuracy_results.append(accuracy_results_diagonal_on_training)
    labels.append("Parallel_on_training")

    plot_error_lines(accuracy_results, labels, c_values)


def ejercicio_4():
    d_values = [2, 4, 8, 16, 32]
    accuracy_results_parallel_on_test = []
    accuracy_results_parallel_on_training = []
    accuracy_results_diagonal_on_test = []
    accuracy_results_diagonal_on_training = []
    accuracy_results = []
    node_sizes_parallel = []
    node_sizes_diagonal = []
    node_sizes = []
    labels = []

    n = 250
    c = 5

    test_error_total = 0.0
    values_error_total = 0.0
    node_totals = 0.0

    for d in d_values:
        node_totals = 0

        centros_a = centros_eja(d)
        test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)

        for j in range(20):
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_a)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals_parallel = node_totals / 20

        accuracy_results_parallel_on_test.append(test_error_total)
        accuracy_results_parallel_on_training.append(values_error_total)
        node_sizes_parallel.append(node_totals_parallel)

    labels.append("Diagonal_on_test")
    labels.append("Diagonal_on_training")

    accuracy_results.append(accuracy_results_parallel_on_test)
    accuracy_results.append(accuracy_results_parallel_on_training)

    for d in d_values:
        centros_b = centros_ejb(d)
        test_case_b = generar_valores(centros_b, c, d, 10000)

        for j in range(20):
            values = generar_valores(centros_b, c, d, n)
            (test_error, values_error, nodes) = clasifico_y_errores(values, test_case_b)
            node_totals += nodes
            test_error_total += test_error
            values_error_total += values_error

        test_error_total = test_error_total / 20
        values_error_total = values_error_total / 20
        node_totals = node_totals / 20

        accuracy_results_diagonal_on_test.append(test_error_total)
        accuracy_results_diagonal_on_training.append(values_error_total)
        node_sizes_diagonal.append(node_totals)

    accuracy_results.append(accuracy_results_diagonal_on_test)
    labels.append("Parallel_on_test")
    accuracy_results.append(accuracy_results_diagonal_on_training)
    labels.append("Parallel_on_training")
    node_sizes.append(node_sizes_parallel)
    node_sizes.append(node_sizes_diagonal)

    plot_error_lines(accuracy_results, labels, d_values)
    plot_tree_sizes(node_sizes, labels, d_values)
