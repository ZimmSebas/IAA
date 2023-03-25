import pandas as pd
import numpy as np
from matplotlib import pyplot as mpl


def centros_eja(d):
    c_1 = []
    c_2 = []

    for i in range(d):
        c_1.append(1)
        c_2.append(-1)

    return (c_1, c_2)


def centros_ejb(d):
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

    values_centro_1 = [v + [0] for v in values_centro_1]
    values_centro_2 = [v + [1] for v in values_centro_2]

    print(values_centro_1)

    dataframe = pd.DataFrame(
        values_centro_1 + values_centro_2, columns=list(range(d)) + ["Class"]
    )

    return dataframe


def plot(df):
    color_decide = np.where(df["Class"] == 1, "DarkBlue", "DarkGreen")

    axis = df.plot.scatter(
        0,
        1,
        c=color_decide,
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")

    axis.grid(which="both", color="grey", linewidth=1, linestyle="-")
    mpl.show()


def test_ej_1a():
    d = 2
    n = 200
    c = 0.75
    centros = centros_eja(d)
    dataframe = generar_valores(centros, c, d, n)
    plot(dataframe)


def test_ej_1b():
    d = 2
    n = 200
    c = 0.75
    centros = centros_ejb(d)
    dataframe = generar_valores(centros, c, d, n)
    plot(dataframe)


def test_ej_2a():
    d = 4
    n = 5000
    c = 2
    centros = centros_eja(d)
    dataframe = generar_valores(centros, c, d, n)
    print(dataframe.groupby(["Class"]).mean())
    print(dataframe.groupby(["Class"]).std())


def test_ej_2b():
    d = 4
    n = 5000
    c = 2
    centros = centros_ejb(d)
    dataframe = generar_valores(centros, c, d, n)
    print(dataframe.groupby(["Class"]).mean())
    print(dataframe.groupby(["Class"]).std())
