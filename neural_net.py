from sklearn.metrics import accuracy_score, mean_squared_error
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from csv import reader
from statistics import mean
from sys import maxsize


def crear_red(eta, alfa, epocas_por_entrenamiento, N2):
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
            error_train = 1.0 - accuracy_score(results_train, y_train)
            error_val = 1.0 - accuracy_score(results_val, y_val)
            error_test = 1.0 - accuracy_score(results_test, y_test)

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
            print(np_error_test)
            print(mean_test_error)
            print(pos_min_val_error)
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
    print(best_train_error)
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

    print(df_table)
    print(df_errors)

def ejercicio_2_print():
    df_errors_training = pd.read_csv("TP_2/errors_training_ej_2.csv")
    plot_errors(df_errors_training,title = "Eta = 0.001 - Alfa = 0.9")