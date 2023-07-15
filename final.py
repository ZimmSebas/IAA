import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from data_gen import *


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

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(10))

    epocas = 100

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