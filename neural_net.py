from sklearn.metrics import accuracy_score
from data_gen import *
from matplotlib import pyplot as mpl
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

""" def entrenar(case, eta, alfa, epocas_por_entrenamiento, N2):
    X_train, y_train = case.iloc[:, :-1], case.iloc[:, -1:]
    clasif = MLPClassifier(
        hidden_layer_sizes = (N2,),
        activation = 'logistic',
        solver = 'sgd', 
        alpha = 0.0, 
        batch_size = 1, 
        learning_rate = 'constant', 
        learning_rate_init = eta,
        momentum = alfa,
        nesterovs_momentum = False,
        tol = 0.0,
        warm_start = True,
        max_iter = epocas_por_entrenamiento
    )
    entrenar_red(clasif, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test)
    return clasif


def clasificar(red, clasif):


def entrenar_red(red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test):
     #mi código
     #red.fit(X_train, y_train)
     #más código
     return best_red, error_train, error_val, error_test

def espirales_entrenados(n, test_case, eta, alfa, epocas_por_entrenamiento):
    case = generar_espirales(n)
    clf = entrenar(case, eta, alfa, epocas_por_entrenamiento)
    case_clasificado = clasificar(test_case, clf)
    return case_clasificado """


def ejercicio_1():
    capas_intermedias = [2, 10, 20, 40]
    eta = 0.1
    alfa = 0.9
    epocas_por_entrenamiento = 20

    test = generar_espirales(2000)
    case = generar_espirales(600)
    X_raw, y_raw = case.iloc[:, :-1], case.iloc[:, -1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    # Usé 42 como seed porque lo recomendaba la documentación oficial :)

    print(X_train)
    print(y_train)
    print(X_val)
    print(y_val)

    for N2 in capas_intermedias:
        a = 2
    return
