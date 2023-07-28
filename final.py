import pandas as pd
import numpy as np
from matplotlib import pyplot as mpl
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

# from google.colab import files


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


def preprocess_cifar10_dataset():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    # Reduce the pixel values
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Flatten the label values (Es necesario? No se aplana despues?)
    # y_train, y_test = y_train.flatten(), y_test.flatten()

    return X_train, y_train, X_test, y_test


def ejercicio_1():
    X_raw, y_raw, X_test, y_test = preprocess_cifar10_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, random_state=0, test_size=0.2
    )

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(
        layers.Flatten()
    )  # Yo asumo que este Flatten es suficiente para reemplazar el que iria en preprocess. Creo que da decente.
    model.add(layers.Dense(64))
    model.add(layers.Dense(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(10))

    epocas = 30

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(X_train, y_train, epochs=epocas, validation_data=(X_val, y_val))

    errors = []

    for i in range(epocas):
        errors.append([1 - history.history["accuracy"][i], i + 1, "Train error"])
        errors.append(
            [1 - history.history["val_accuracy"][i], i + 1, "Validation error"]
        )

    df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
    plot_errors(df_errors, "Errores por época")

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    error_train = 1 - history.history["accuracy"][epocas - 1]
    error_val = 1 - history.history["val_accuracy"][epocas - 1]
    error_test = 1 - test_acc
    print(
        "Train Error: "
        + str(error_train)
        + " Val error: "
        + str(error_val)
        + " Test error: "
        + str(error_test)
    )


# Train Error: 0.04407501220703125 Val error: 0.30239999294281006 Test error: 0.31470000743865967


def ejercicio_1_print():
    df_errors = pd.read_csv("TP_FINAL/ejercicio1.csv")
    plot_errors(df_errors, "Errores por época")


def ejercicio_2():
    X_raw, y_raw, X_test, y_test = preprocess_cifar10_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, random_state=0, test_size=0.2
    )

    errors = []
    test_errors = []
    epocas = 30  # No realicé el doble porque tomaba infinito

    for d1 in [0, 0.2, 0.5]:
        for d2 in [0, 0.2, 0.5]:
            model = models.Sequential()
            model.add(
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(32, 32, 3),
                    padding="same",
                )
            )
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
            model.add(
                layers.Dropout(d1, seed=42)
            )  # No se si sale bien el 42, pero al menos es consistente
            model.add(layers.Flatten())
            model.add(layers.Dense(64))
            model.add(layers.Dropout(d2, seed=42))
            model.add(layers.Dense(128))
            model.add(layers.Dense(128))
            model.add(layers.Dense(10))

            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            history = model.fit(
                X_train, y_train, epochs=epocas, validation_data=(X_val, y_val)
            )
            _, error_test = model.evaluate(X_test, y_test, verbose=0)
            test_errors.append([d1, d2, 1 - error_test])

            for i in range(epocas):
                errors.append(
                    [
                        1 - history.history["accuracy"][i],
                        i + 1,
                        "Train error on d1: " + str(d1) + ", d2: " + str(d2),
                    ]
                )
                errors.append(
                    [
                        1 - history.history["val_accuracy"][i],
                        i + 1,
                        "Validation error on d1: " + str(d1) + ", d2: " + str(d2),
                    ]
                )

            df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
            df_test_errors = pd.DataFrame(
                test_errors, columns=["d1", "d2", "Test error"]
            )

            errors_file = "df_errors-" + str(d1) + "-" + str(d2) + ".csv"
            test_errors_file = "df_test_errors-" + str(d1) + "-" + str(d2) + ".csv"


#      For Google to download the files
#
#      df_errors.to_csv(errors_file)
#      files.download(errors_file)
#      df_test_errors.to_csv(test_errors_file)
#      files.download(test_errors_file)


def plot_error_lines(error_dataframe):
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
        "firebrick",
        "firebrick",
        "slategrey",
        "slategrey",
        "cadetblue",
        "cadetblue",
        "chocolate",
        "chocolate",
    ]
    line = [
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
        ":",
        "-",
    ]

    types = list(pd.unique(error_dataframe["Clase"]))

    for i in range(len(types)):
        df = error_dataframe[error_dataframe["Clase"] == types[i]]
        mpl.plot(
            df["Épocas"],
            df["Error"],
            color=colors[i],
            label=types[i],
            linestyle=line[i],
        )

    mpl.xlabel("Épocas")
    mpl.ylabel("Error")
    mpl.legend()

    mpl.show()


def ejercicio_2_comb():
    df_errors_0_0 = pd.read_csv("TP_FINAL/df_errors-0-0.csv")
    df_errors_0_2 = pd.read_csv("TP_FINAL/df_errors-0-0.2.csv")
    df_errors_0_5 = pd.read_csv("TP_FINAL/df_errors-0-0.5.csv")
    df_errors_2_0 = pd.read_csv("TP_FINAL/df_errors-0.2-0.csv")
    df_errors_2_2 = pd.read_csv("TP_FINAL/df_errors-0.2-0.2.csv")
    df_errors_2_5 = pd.read_csv("TP_FINAL/df_errors-0.2-0.5.csv")
    df_errors_5_0 = pd.read_csv("TP_FINAL/df_errors-0.5-0.csv")
    df_errors_5_2 = pd.read_csv("TP_FINAL/df_errors-0.5-0.2.csv")
    df_errors_5_5 = pd.read_csv("TP_FINAL/df_errors-0.5-0.5.csv")
    df_errors = pd.concat(
        [
            df_errors_0_0,
            df_errors_0_2,
            df_errors_0_5,
            df_errors_2_0,
            df_errors_2_2,
            df_errors_2_5,
            df_errors_5_0,
            df_errors_5_2,
            df_errors_5_5,
        ]
    )
    df_errors.to_csv("TP_FINAL/errors_ej_2.csv", index=False)

    df_test_errors_0_0 = pd.read_csv("TP_FINAL/df_test_errors-0-0.csv")
    df_test_errors_0_2 = pd.read_csv("TP_FINAL/df_test_errors-0-0.2.csv")
    df_test_errors_0_5 = pd.read_csv("TP_FINAL/df_test_errors-0-0.5.csv")
    df_test_errors_2_0 = pd.read_csv("TP_FINAL/df_test_errors-0.2-0.csv")
    df_test_errors_2_2 = pd.read_csv("TP_FINAL/df_test_errors-0.2-0.2.csv")
    df_test_errors_2_5 = pd.read_csv("TP_FINAL/df_test_errors-0.2-0.5.csv")
    df_test_errors_5_0 = pd.read_csv("TP_FINAL/df_test_errors-0.5-0.csv")
    df_test_errors_5_2 = pd.read_csv("TP_FINAL/df_test_errors-0.5-0.2.csv")
    df_test_errors_5_5 = pd.read_csv("TP_FINAL/df_test_errors-0.5-0.5.csv")
    df_test_errors = pd.concat(
        [
            df_test_errors_0_0,
            df_test_errors_0_2,
            df_test_errors_0_5,
            df_test_errors_2_0,
            df_test_errors_2_2,
            df_test_errors_2_5,
            df_test_errors_5_0,
            df_test_errors_5_2,
            df_test_errors_5_5,
        ]
    )
    df_test_errors.drop(index=0)
    df_test_errors.to_csv("TP_FINAL/test_errors_ej_2.csv", index=False)


def ejercicio_2_print():
    df_errors = pd.read_csv("TP_FINAL/errors_ej_2.csv")
    plot_error_lines(df_errors)

    df_test_errors = pd.read_csv("TP_FINAL/test_errors_ej_2.csv")
    print(df_test_errors)


def ejercicio_3():
    X_raw, y_raw, X_test, y_test = preprocess_cifar10_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, random_state=0, test_size=0.2
    )

    errors = []
    epocas = 50

    img_height = 32
    img_width = 32
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    model = models.Sequential()
    model.add(data_augmentation)
    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Dropout(0.5, seed=42))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.2, seed=42))
    model.add(layers.Dense(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(10))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(X_train, y_train, epochs=epocas, validation_data=(X_val, y_val))
    _, error_test = model.evaluate(X_test, y_test, verbose=0)

    for i in range(epocas):
        errors.append([1 - history.history["accuracy"][i], i + 1, "Train error"])
        errors.append(
            [1 - history.history["val_accuracy"][i], i + 1, "Validation error"]
        )

    df_errors = pd.DataFrame(errors, columns=["Error", "Épocas", "Clase"])
    print(1 - error_test)

    # Google Colab stuff
    # df_errors.to_csv("errors_ejercicio_3.csv")
    # files.download("errors_ejercicio_3.csv")


# error test: 0.269900023937


def ejercicio_3_print():
    df_errors = pd.read_csv("TP_FINAL/errors_ejercicio_3.csv")
    plot_errors(df_errors, "Errores por época")
