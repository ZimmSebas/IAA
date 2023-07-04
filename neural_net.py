from sklearn.metrics import mean_squared_error, zero_one_loss
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from csv import reader
from sys import maxsize


def crear_red(eta, alfa, epocas_por_entrenamiento, N2, typ="clas", gamma=0.0):
    if typ == "clas":
        red = MLPClassifier(
            hidden_layer_sizes=(N2,),
            activation="logistic",
            solver="sgd",
            alpha=0.0,
            batch_size=1,
            learning_rate="constant",
            learning_rate_init=eta,
            momentum=alfa,
            nesterovs_momentum=False,
            tol=0.0,
            warm_start=True,
            max_iter=epocas_por_entrenamiento,
        )
    else:
        red = MLPRegressor(
            hidden_layer_sizes=(N2,),
            activation="logistic",
            solver="sgd",
            alpha=gamma,
            batch_size=1,
            learning_rate="constant",
            learning_rate_init=eta,
            momentum=alfa,
            nesterovs_momentum=False,
            tol=0.0,
            warm_start=True,
            max_iter=epocas_por_entrenamiento,
        )
    return red


def entrenar_red(
    red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test, mse=False
):
    best_error = 1

    all_error_train = []
    all_error_val = []
    all_error_test = []

    for i in range(evaluaciones):
        red.fit(X_train, np.ravel(y_train))

        results_train = red.predict(X_train)
        results_val = red.predict(X_val)
        results_test = red.predict(X_test)

        if mse:
            error_train = mean_squared_error(results_train, y_train)
            error_val = mean_squared_error(results_val, y_val)
            error_test = mean_squared_error(results_test, y_test)
        else:
            error_train = zero_one_loss(results_train, y_train)
            error_val = zero_one_loss(results_val, y_val)
            error_test = zero_one_loss(results_test, y_test)

        all_error_train.append(error_train)
        all_error_val.append(error_val)
        all_error_test.append(error_test)

        if best_error > error_val:
            best_red = deepcopy(red)

    return best_red, all_error_train, all_error_val, all_error_test


def ejercicio_1():
    capas_intermedias = [2, 10, 20, 40]
    eta = 0.1  # learning rate
    alfa = 0.9  # momemtum
    epocas_por_entrenamiento = 20
    evaluaciones = 1000

    test = generar_espirales(2000)
    case = generar_espirales(600)

    X_raw, y_raw = case.iloc[:, :-1], case.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    # Usé 42 como seed porque lo recomendaba la documentación oficial :)
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    for N2 in capas_intermedias:
        red = crear_red(eta, alfa, epocas_por_entrenamiento, N2)
        best_red, all_error_train, all_error_val, all_error_test = entrenar_red(
            red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test
        )
        results_with_best_red = best_red.predict(X_test)
        df_results = test.copy(deep=True)
        df_results["Class"] = results_with_best_red
        plot(df_results)


def ejercicio_2():
    momemtums = [0, 0.5, 0.9]
    learning_rates = [0.1, 0.01, 0.001]
    evaluaciones = 300
    N2 = 6
    epocas_por_entrenamiento = 50

    errors_train = []
    errors_val = []
    errors_test = []

    with open("TP_2/dos_elipses.data") as csvfile:
        lines = reader(csvfile)
        data = pd.DataFrame(lines)

    with open("TP_2/dos_elipses.test") as csvfile:
        lines = reader(csvfile)
        test = pd.DataFrame(lines)

    X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    errors_table = []

    for eta in learning_rates:
        for alfa in momemtums:
            mean_train_error = []
            mean_val_error = []
            mean_test_error = []

            best_mean_test_error = maxsize

            for i in range(10):
                red = crear_red(eta, alfa, epocas_por_entrenamiento, N2)

                best_red, error_train, error_val, error_test = entrenar_red(
                    red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test
                )

                errors_train.append(error_train)
                errors_val.append(error_val)
                errors_test.append(error_test)

            np_error_train = np.asarray(errors_train)
            np_error_val = np.asarray(errors_val)
            np_error_test = np.asarray(errors_test)

            mean_train_error = np_error_train.mean(axis=0)
            mean_val_error = np_error_val.mean(axis=0)
            mean_test_error = np_error_test.mean(axis=0)

            min_val_error = np.min(mean_val_error)
            pos_min_val_error = np.where(mean_val_error == min_val_error)[0][0]
            min_test_error = mean_test_error[pos_min_val_error]

            if min_test_error < best_mean_test_error:
                best_mean_test_error = min_test_error
                best_eta = eta
                best_alfa = alfa
                best_train_error = mean_train_error.copy()
                best_val_error = mean_val_error.copy()
                best_test_error = mean_test_error.copy()

            errors_table.append([eta, alfa, min_test_error])

    print(
        "El mejor caso fue eta: {0}, alfa: {1} with error de test promedio: {2}".format(
            best_eta, best_alfa, best_mean_test_error
        )
    )

    errores = []
    for i in range(evaluaciones):
        errores.append(
            [best_train_error[i], i * epocas_por_entrenamiento, "Error train"]
        )
        errores.append(
            [best_val_error[i], i * epocas_por_entrenamiento, "Error validación"]
        )
        errores.append([best_test_error[i], i * epocas_por_entrenamiento, "Error test"])

    df_errors = pd.DataFrame(errores, columns=["Error", "Épocas", "Clase"])
    df_table = pd.DataFrame(errors_table, columns=["eta", "alfa", "Media error test"])
    df_errors.to_csv("TP_2/errors_training_ej_2.csv", index=False)
    df_table.to_csv("TP_2/table_ej_2.csv", index=False)


def ejercicio_2_print():
    df_errors_training = pd.read_csv("TP_2/errors_training_ej_2.csv")
    plot_errors(df_errors_training, title="Eta = 0.001 - Alfa = 0.9")


def ejercicio_3():
    alfa = 0.9  # momemtum
    eta = 0.01  # learning_rate
    evaluaciones = 400
    N2 = 30
    epocas_por_entrenamiento = 50

    ratios = [0.95, 0.75, 0.5]

    columns = list(range(5)) + ["Class"]

    data = pd.read_csv(
        "TP_2/ikeda.data",
        names=columns,
        header=None,
        skipinitialspace=True,
        delim_whitespace=True,
    )
    test = pd.read_csv(
        "TP_2/ikeda.test",
        names=columns,
        header=None,
        skipinitialspace=True,
        delim_whitespace=True,
    )

    X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    for ratio in ratios:
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, y_raw, test_size=ratio, random_state=42
        )
        red = crear_red(eta, alfa, epocas_por_entrenamiento, N2, typ="regr")

        best_red, error_train, error_val, error_test = entrenar_red(
            red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test, mse=True
        )

        errores = []

        for i in range(evaluaciones):
            errores.append(
                [error_train[i], i * epocas_por_entrenamiento, "Error train"]
            )
            errores.append(
                [error_val[i], i * epocas_por_entrenamiento, "Error validación"]
            )
            errores.append([error_test[i], i * epocas_por_entrenamiento, "Error test"])

        df_errors = pd.DataFrame(errores, columns=["Error", "Épocas", "Clase"])
        df_errors.to_csv("TP_2/errors_ej_3_" + str(ratio) + ".csv", index=False)

    ejercicio_3_print()


def ejercicio_3_print():
    df_errors_training = pd.read_csv("TP_2/errors_ej_3_0.95.csv")
    plot_errors(df_errors_training, title="Errors with 0.95")
    df_errors_training = pd.read_csv("TP_2/errors_ej_3_0.75.csv")
    plot_errors(df_errors_training, title="Errors with 0.75")
    df_errors_training = pd.read_csv("TP_2/errors_ej_3_0.5.csv")
    plot_errors(df_errors_training, title="Errors with 0.5")


def entrenar_red_con_gamma(
    red, evaluaciones, X_train, y_train, X_test, y_test, mse=True
):
    best_error = 1

    all_error_train = []
    all_error_test = []
    all_wsum = []

    for i in range(evaluaciones):
        red.fit(X_train, np.ravel(y_train))

        results_train = red.predict(X_train)
        results_test = red.predict(X_test)

        if mse:
            error_train = mean_squared_error(results_train, y_train)
            error_test = mean_squared_error(results_test, y_test)
        else:
            error_train = zero_one_loss(results_train, y_train)
            error_test = zero_one_loss(results_test, y_test)

        wsum = sum(map(lambda weight: np.sum(np.power(weight, 2)), red.coefs_))

        all_error_train.append(error_train)
        all_error_test.append(error_test)
        all_wsum.append(wsum)

        if best_error > error_test:
            best_red = deepcopy(red)

    return best_red, all_error_test, all_error_train, all_wsum


def ejercicio_4():
    alfa = 0.3  # momemtum
    eta = 0.05  # learning_rate
    evaluaciones = 4000
    N2 = 6
    epocas_por_entrenamiento = 20

    gammas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    columns = list(range(5)) + ["Class"]

    data = pd.read_csv(
        "TP_2/ssp.data", names=columns, sep=",", header=None, skipinitialspace=True,
    )
    test = pd.read_csv(
        "TP_2/ssp.test", names=columns, sep=",", header=None, skipinitialspace=True,
    )

    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    for gamma in gammas:
        red = crear_red(
            eta, alfa, epocas_por_entrenamiento, N2, typ="regr", gamma=gamma
        )

        best_red, error_train, error_test, wsum = entrenar_red_con_gamma(
            red, evaluaciones, X_train, y_train, X_test, y_test
        )

        best_red.fit(X_train, np.ravel(y_train))

        # best_red.coefs_ son los pesos, creo

        errors = []
        weights = []

        for i in range(evaluaciones):
            errors.append([error_train[i], i * epocas_por_entrenamiento, "Error train"])
            errors.append([error_test[i], i * epocas_por_entrenamiento, "Error test"])
            weights.append([wsum[i], i * epocas_por_entrenamiento])

        df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
        df_errors.to_csv("TP_2/errors_ej_4_" + str(gamma) + ".csv", index=False)
        df_weights = pd.DataFrame(weights, columns=["Weight", "Épocas"])
        df_weights.to_csv("TP_2/weights_ej_4_" + str(gamma) + ".csv", index=False)

        # 1e05 es mejor.


def ejercicio_4_print():
    df_errors_training = pd.read_csv("TP_2/errors_ej_4_1e-05.csv")
    plot_errors(df_errors_training, title="Errors with 1e-05")
    df_errors_training = pd.read_csv("TP_2/errors_ej_4_0.1.csv")
    plot_errors(df_errors_training, title="Errors with 0.1")
    df_weights_training = pd.read_csv("TP_2/weights_ej_4_1e-05.csv")
    plot_weights(df_weights_training, title="Weights with 1e-05")
    df_weights_training = pd.read_csv("TP_2/weights_ej_4_0.1.csv")
    plot_weights(df_weights_training, title="Weights with 0.1")


def ejercicio_5():
    alfa = 0.9  # momemtum
    eta = 0.1  # learning_rate
    evaluaciones = 400
    N2 = 6
    epocas_por_entrenamiento = 20
    gamma = 10 ** -5
    c = 0.78
    n = 250

    d_values = [2, 4, 8, 16, 32]

    errors = []

    for d in d_values:
        centros_a = centros_eja(d)

        test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)
        X_test, y_test = test_case_a.iloc[:, :-1], test_case_a.iloc[:, -1:]

        test_error_para = 0.0
        values_error_para = 0.0
        test_error_diag = 0.0
        values_error_diag = 0.0

        for j in range(20):
            values = generar_valores(centros_a, c * sqrt(d), d, n)
            red = crear_red(eta, alfa, epocas_por_entrenamiento, N2, gamma=gamma)
            X_train, y_train = values.iloc[:, :-1], values.iloc[:, -1:]

            best_red, t_errors, v_errors, wsums = entrenar_red_con_gamma(
                red, evaluaciones, X_train, y_train, X_test, y_test, mse=False
            )

            results_train = best_red.predict(X_train)
            results_test = best_red.predict(X_test)

            test_error_para += zero_one_loss(results_test, y_test)
            values_error_para += zero_one_loss(results_train, y_train)

        test_error_para = test_error_para / 20
        values_error_para = values_error_para / 20

        centros_b = centros_ejb(d)
        test_case_b = generar_valores(centros_b, c, d, 10000)
        X_test, y_test = test_case_b.iloc[:, :-1], test_case_b.iloc[:, -1:]

        for j in range(20):
            values = generar_valores(centros_b, c, d, n)
            red = crear_red(eta, alfa, epocas_por_entrenamiento, N2, gamma=gamma)
            X_train, y_train = values.iloc[:, :-1], values.iloc[:, -1:]

            best_red, test_error, values_error, wsums = entrenar_red_con_gamma(
                red, evaluaciones, X_train, y_train, X_test, y_test, mse=False
            )

            results_train = best_red.predict(X_train)
            results_test = best_red.predict(X_test)

            test_error_diag += zero_one_loss(results_test, y_test)
            values_error_diag += zero_one_loss(results_train, y_train)

        test_error_diag = test_error_diag / 20
        values_error_diag = values_error_diag / 20

        errors.append([test_error_para, d, "Test_Parallel_NN"])
        errors.append([values_error_para, d, "Val_Parallel_NN"])
        errors.append([test_error_diag, d, "Test_Diagonal_NN"])
        errors.append([values_error_diag, d, "Val_Diagonal_NN"])

    df_errors = pd.DataFrame(errors, columns=["Error", "D", "Type"])
    df_errors.to_csv("TP_2/errors_ej_5.csv", index=False)


def plot_error_lines_with_dimensions(error_dataframe):
    colors = ["red", "red", "blue", "blue", "green", "green", "orange", "orange"]
    line = [":", "-", ":", "-", ":", "-", ":", "-"]

    types = list(pd.unique(error_dataframe["Type"]))

    print(error_dataframe)

    for i in range(len(types)):
        df = error_dataframe[error_dataframe["Type"] == types[i]]
        mpl.plot(
            df["D"], df["Error"], color=colors[i], label=types[i], linestyle=line[i],
        )

    mpl.xlabel("Sizes")
    mpl.ylabel("Error")
    mpl.legend()

    mpl.show()


def ejercicio_5_print():
    df_errors_tree = pd.read_csv("TP_1/errors_ej_4.csv")
    df_errors_nn = pd.read_csv("TP_2/errors_ej_5.csv")
    df_errors = pd.concat([df_errors_tree, df_errors_nn])
    print(df_errors)
    plot_error_lines_with_dimensions(df_errors)


def ejercicio_6_1():
    alfa = 0.9  # momemtum
    eta = 0.1  # learning_rate
    evaluaciones = 3000
    N2 = 6
    epocas_por_entrenamiento = 200

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_data, X_test, y_data, y_test = train_test_split(
        X, y, random_state=42, test_size=1 / 3
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, random_state=42, test_size=0.2
    )

    red = crear_red(eta, alfa, epocas_por_entrenamiento, N2)

    best_red, error_train, error_val, error_test = entrenar_red(
        red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test
    )

    errors = []

    for i in range(evaluaciones):
        errors.append([error_train[i], i * epocas_por_entrenamiento, "Train error"])
        errors.append([error_val[i], i * epocas_por_entrenamiento, " Validation error"])
        errors.append([error_test[i], i * epocas_por_entrenamiento, "Test error"])

    df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
    df_errors.to_csv("TP_2/errors_ej_6_1.csv", index=False)
    plot_errors(df_errors, title="Errores en Iris Multiclase")


def ejercicio_6_2():
    alfa = 0.9  # momemtum
    eta = 0.1  # learning_rate
    evaluaciones = 3000
    N2 = 6
    epocas_por_entrenamiento = 50

    data = pd.read_csv("TP_2/faces.data", header=None)
    test = pd.read_csv("TP_2/faces.test", header=None)

    # print(data.max())
    # print(data.max().max())

    maximum = max(data.max().max(), test.max().max())

    X_data, y_data = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    X_data = X_data / maximum
    X_test = X_test / maximum

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, random_state=42, test_size=0.2
    )

    red = crear_red(eta, alfa, epocas_por_entrenamiento, N2)

    best_red, error_train, error_val, error_test = entrenar_red(
        red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test
    )

    errors = []

    for i in range(evaluaciones):
        errors.append([error_train[i], i * epocas_por_entrenamiento, "Train error"])
        errors.append([error_val[i], i * epocas_por_entrenamiento, " Validation error"])
        errors.append([error_test[i], i * epocas_por_entrenamiento, "Test error"])

    df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
    print(df_errors)
    df_errors.to_csv("TP_2/errors_ej_6_2.csv", index=False)
    plot_errors(df_errors, title="Errores en Faces")
