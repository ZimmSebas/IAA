import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl


def centros_ej1(d):
    c_1 = []
    c_2 = []

    for i in range(d):
        c_1.append(1)
        c_2.append(-1)

    return (c_1, c_2)


def centros_ej2(d):
    c_1 = []
    c_2 = []

    for i in range(d):
        if not (i):
            c_1.append(1)
            c_2.append(-1)
        else:
            c_1.append(0)
            c_2.append(0)

    return (c_1, c_2)


def generar_valores(centros, c, d, n):
    (c_1, c_2) = centros
    std_dev = c**2
    dev_mat = np.diag([std_dev] * d)
    values_centro_1 = (
        np.random.default_rng()
        .multivariate_normal(mean=c_1, cov=dev_mat, size=n // 2)
        .tolist()
    )
    values_centro_2 = (
        np.random.default_rng()
        .multivariate_normal(mean=c_2, cov=dev_mat, size=n // 2)
        .tolist()
    )
    values = (values_centro_1, values_centro_2)
    dataframe = pd.DataFrame(values)
    return dataframe


def plot(df):
    print(df[0])
    mpl.scatter(df[0], "r+", label="Clase 0")
    mpl.scatter(df[1], "bo", label="Clase 1")
    mpl.show()


def pretest():
    d = 2
    n = 8
    c = 0.75
    centros = centros_ej1(d)
    dataf = generar_valores(centros, c, d, n)
    plot(dataf)


def test_ej_a():
    d = 2
    n = 200
    c = 0.75
    centros = centros_ej1(d)
    (valores_clase_1, valores_clase_2) = generar_valores(centros, c, d, n)
    plot(valores_clase_1, valores_clase_2)


def test_ej_b():
    d = 2
    n = 200
    centros = centros_ej2(d)
    (valores_clase_1, valores_clase_2) = generar_valores(centros, c, d, n)
    plot(valores_clase_1, valores_clase_2)
