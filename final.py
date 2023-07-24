import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as mpl


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
  #y_train, y_test = y_train.flatten(), y_test.flatten()

  return X_train, y_train, X_test, y_test


def ejercicio_1():
    X_raw, y_raw, X_test, y_test = preprocess_cifar10_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, random_state=0, test_size=0.2)
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten()) # Yo asumo que este Flatten es suficiente para reemplazar el que iria en preprocess. Creo que da decente.
    model.add(layers.Dense(64))
    model.add(layers.Dense(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(10))

    epocas = 30

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epocas, 
                        validation_data=(X_val, y_val))
    
    errors = []

    for i in range(epocas):
        errors.append([1 - history.history['accuracy'][i], i + 1, "Train error" ])
        errors.append([1 - history.history['val_accuracy'][i], i + 1, "Validation error" ])

    df_errors = pd.DataFrame(errors, columns = ["Error", "Épocas", "Clase"])
    plot_errors(df_errors, "Errores por época")

    _, test_acc = model.evaluate(X_test,  y_test, verbose=0)
    error_train = 1 - history.history['accuracy'][epocas - 1]
    error_val = 1 - history.history['val_accuracy'][epocas - 1]
    error_test = 1 - test_acc
    print("Train Error: " + str(error_train) + " Val error: " + str(error_val) + " Test error: " + str(error_test) )

def ejercicio_2():
  X_raw, y_raw, X_test, y_test = preprocess_cifar10_dataset()
  X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, random_state=0, test_size=0.2)

  errors = []
  test_errors = []
  epocas = 30 # No realicé el doble porque tomaba infinito
  
  for d1 in [0, 0.2, 0.5]:
     for d2 in [0, 0.2, 0.5]:
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
      model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
      model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
      model.add(layers.Dropout(d1, seed=42)) # No se si sale bien el 42, pero al menos es consistente
      model.add(layers.Flatten())
      model.add(layers.Dense(64))
      model.add(layers.Dropout(d2, seed=42))
      model.add(layers.Dense(128))
      model.add(layers.Dense(128))
      model.add(layers.Dense(10))

 
      model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
      history = model.fit(X_train, y_train, epochs=epocas, 
                          validation_data=(X_val, y_val))
      _, error_test = model.evaluate(X_test,  y_test, verbose=0)
      test_errors.append([d1, d2, error_test])        

      for i in range(epocas):
          errors.append([1 - history.history['accuracy'][i], i + 1, "Train error on d1: " + str(d1) + ", d2: " + str(d2)])
          errors.append([1 - history.history['val_accuracy'][i], i + 1, "Validation error on d1: " + str(d1) + ", d2: " + str(d2)])
      print(errors)


  df_errors = pd.DataFrame(errors, columns = ["Error", "Épocas", "Clase"])
  df_test_errors = pd.DataFrame(errors, columns = ["d1", "d2", "Test error"])
  print(df_test_errors)
  plot_errors(df_errors, "Errores por época")
