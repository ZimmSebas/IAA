from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

# defino parámetros de la red
epocas_por_entrenamiento = 25  # numero de epocas que entrena cada vez
eta = 0.01  # learning rate
alfa = 0.9  # momentum
N2 = 60  # neuronas en la capa oculta

# defino MLP para regresión
regr = MLPRegressor(
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
# defino MLP para clasificación
clasif = MLPClassifier(
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
print(regr)


def entrenar_red(red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test):
    # mi código
    # red.fit(X_train, y_train)
    # más código
    return best_red, error_train, error_val, error_test


regr, e_train, e_val, e_test = entrenar_red(
    regr, epocas, X_train, y_train, X_val, y_val, X_test, y_test
)
import matplotlib.pyplot as plt

plt.plot(range(epocas), e_train, label="train", linestyle=":")
plt.plot(range(epocas), e_val, label="validacion", linestyle="-.")
plt.plot(range(epocas), e_test, label="test", linestyle="-")
plt.legend()
plt.show()
