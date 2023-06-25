from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

def ejercicio_1_classifier(data, test):

  X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
  X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

  best_error_train = 1
  best_n = 0

  errors = []

  for n in range(2,15):
      knclf = KNeighborsClassifier(n)

      knclf.fit(X_train, np.ravel(y_train))

      results_test = knclf.predict(X_test)
      results_train = knclf.predict(X_train)

      error_train = 1 - accuracy_score(y_train, results_train)
      error_test = 1 - accuracy_score(y_test, results_test)

      errors.append([n ,error_train, error_test])

      if error_train < best_error_train:
         best_error_train = error_train
         best_n = n
  
  df_errors = pd.DataFrame(errors, columns=["N", "Error Train", "Error Test"])
  print(df_errors)
  print("The best N: " + str(best_n) + " with error: " + str(best_error_train))


def ejercicio_1():
  data = pd.read_csv(
      "TP_4/c_0.data",
      header=None,
  )

  test = pd.read_csv(
      "TP_4/c_0.test",
      header=None,
  )

  ejercicio_1_classifier(data, test)

  data_noisy = pd.read_csv(
      "TP_4/c_2.data",
      header=None,
  )

  test_noisy = pd.read_csv(
      "TP_4/c_2.test",
      header=None,
  )

  ejercicio_1_classifier(data_noisy, test_noisy)

def ejercicio_2():
   return 
  