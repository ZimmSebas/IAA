from sklearn.metrics import accuracy_score
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


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


def entrenar_red(red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test):
    best_error = 1

    all_error_train = []
    all_error_val = []
    all_error_test = []

    for i in range(evaluaciones):
        red.fit(X_train, np.ravel(y_train))

        results_train = red.predict(X_train)
        results_val = red.predict(X_val)
        results_test = red.predict(X_test)

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
    eta = 0.1
    alfa = 0.9
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
        df_results["Clase"] = results_with_best_red
        plot(test)


def ejercicio_2():
    momemtums = [0, 0.5, 0.9]
    learning_rates = [0.1, 0.01, 0.001]
    entrenamientos = 10
