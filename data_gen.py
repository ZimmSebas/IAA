import pandas as pd
import numpy as np
from math import dist, sqrt
from cmath import polar
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

    dataframe = pd.DataFrame(
        values_centro_1 + values_centro_2, columns=list(range(d)) + ["Class"]
    )

    return dataframe


def plot(df, title=None):
    color_decide = np.where(df["Class"] == 1, "DarkBlue", "DarkGreen")

    axis = df.plot.scatter(
        0,
        1,
        c=color_decide,
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")

    if title != None:
        axis.set_title(title)

    axis.grid(which="both", color="grey", linewidth=1, linestyle="-")
    mpl.show()


def plot_error_lines_with_dimensions(error_dataframe):
    colors = [
        "red",
        "red",
        "blue",
        "blue",
        "green",
        "green",
        "orange",
        "orange",
        "purple",
        "purple",
        "indigo",
        "indigo",
    ]
    line = [":", "-", ":", "-", ":", "-", ":", "-", ":", "-", ":", "-"]

    types = list(pd.unique(error_dataframe["Type"]))

    for i in range(len(types)):
        df = error_dataframe[error_dataframe["Type"] == types[i]]
        mpl.plot(
            df["D"],
            df["Error"],
            color=colors[i],
            label=types[i],
            linestyle=line[i],
        )

    mpl.xlabel("Sizes")
    mpl.ylabel("Error")
    mpl.legend()

    mpl.show()


def plot_error_lines(results, labels, sizes):
    colors = ["red", "red", "blue", "blue", "green", "green"]
    line = ["-", "-", "-", "-", "-", "-"]
    markers = ["o", "v", "o", "v", "o", "v"]

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


def plot_errors(df_errors, title):
    clases = pd.unique(df_errors["Clase"])

    colors = ["red", "blue", "green"]
    linestyles = [":", "-.", "-"]

    mpl.title(title)
    for i in range(len(clases)):
        df_clase = df_errors[df_errors["Clase"] == clases[i]]
        mpl.plot(
            df_clase["Épocas"],
            df_clase["Error"],
            color=colors[i],
            label=clases[i],
            linestyle=linestyles[i],
        )

    mpl.xlabel("Épocas")
    mpl.ylabel("Errors")
    mpl.legend()

    mpl.show()

def plot_error_bins(df_errors, title):
    clases = pd.unique(df_errors["Clase"])

    colors = ["red", "blue", "green"]
    linestyles = [":", "-.", "-"]

    mpl.title(title)
    for i in range(len(clases)):
        df_clase = df_errors[df_errors["Clase"] == clases[i]]
        mpl.plot(
            df_clase["Bins"],
            df_clase["Error"],
            color=colors[i],
            label=clases[i],
            linestyle=linestyles[i],
        )

    mpl.xlabel("Bins")
    mpl.ylabel("Errors")
    mpl.legend()

    mpl.show()



def plot_weights(df_weights, title):
    mpl.title(title)
    mpl.plot(
        df_weights["Weight"],
        df_weights["Épocas"],
        color="blue",
        label="Weights",
        linestyle="-",
    )

    mpl.xlabel("Weights")
    mpl.ylabel("Errors")
    mpl.legend()

    mpl.show()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def fst_function(theta):
    return theta / (4 * np.pi)


def snd_function(theta):
    return (theta + np.pi) / (4 * np.pi)


def random_points(n):
    points = []
    for i in range(n):
        [x, y] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        while dist([0, 0], [x, y]) > 1:
            (x, y) = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        points.append([x, y])
    return points


def change_to_polars(distribution):
    polar_distribution = []
    for p in distribution:
        (r, theta) = polar(complex(p[0], p[1]))
        polar_distribution.append([r, theta])
    return polar_distribution


def espiral_dataframe_with_class(points):
    polar_points = change_to_polars(points)
    i = 0

    for [rho, theta] in polar_points:
        class_0 = (
            fst_function(theta) < rho < snd_function(theta)
            or fst_function(theta) + 0.5 < rho < snd_function(theta) + 0.5
            or fst_function(theta) + 1 < rho < snd_function(theta) + 1
        )

        if class_0:
            points[i].append(0)
        else:
            points[i].append(1)
        i += 1

    return pd.DataFrame(points, columns=[0, 1, "Class"])


def generar_espirales(n):
    points = random_points(n)
    dataframe = espiral_dataframe_with_class(points)
    return dataframe


def test_ej_1a():
    d = 2
    n = 200
    c = 0.75
    centros = centros_eja(d)
    dataframe = generar_valores(centros, c * sqrt(d), d, n)
    print(dataframe)
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
    dataframe = generar_valores(centros, c * sqrt(d), d, n)
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


def test_espirales():
    n = 20000
    points = random_points(n)
    dataframe = espiral_dataframe_with_class(points)
    plot(dataframe)
