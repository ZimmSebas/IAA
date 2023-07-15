import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def preprocess_cifar10_dataset():
  (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

  # Reduce the pixel values
  X_train, X_test = X_train / 255.0, X_test / 255.0

  # Flatten the label values
  y_train, y_test = y_train.flatten(), y_test.flatten()

  return X_train, y_train, X_test, y_test

