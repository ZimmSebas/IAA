from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from tree import entrenar, clasificar

def ejercicio_1_no_noise(data, test):

  X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
  X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

  X_train, X_val, y_train, y_val = train_test_split(
      X_raw, y_raw, test_size=0.2, random_state=42
  )

  best_error_val = 1
  best_n = 0

  errors = []

  for n in range(2,15):
      knclf = KNeighborsClassifier(n)

      knclf.fit(X_train, np.ravel(y_train))

      results_test = knclf.predict(X_test)
      results_val = knclf.predict(X_val)
      results_train = knclf.predict(X_train)

      error_train = 1 - accuracy_score(y_train, results_train)
      error_val = 1 - accuracy_score(y_val, results_val)
      error_test = 1 - accuracy_score(y_test, results_test)

      errors.append([n ,error_train, error_val, error_test])

      if error_val < best_error_val:
         best_error_val = error_val
         best_n = n
         best_results = results_test
  
  df_errors = pd.DataFrame(errors, columns=["N", "Error Train", "Error Val", "Error Test"])
  print(df_errors)
  print("The best N: " + str(best_n) + " with error: " + str(best_error_val))

  plot(test, title="Original")

  # Tree Clasifier
  tree_clf = entrenar(data)
  df_results_tree = clasificar(test, tree_clf)
  plot(df_results_tree, title="Tree")

  # KNN Clasifier
  df_results_knn = X_test.copy()
  df_results_knn["Class"] = best_results
  plot(df_results_knn, title="KNN")


def ejercicio_1_noise(data, test):

  X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
  X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

  X_train, X_val, y_train, y_val = train_test_split(
      X_raw, y_raw, test_size=0.2, random_state=42
  )

  best_error_val = 1
  best_n = 0

  errors = []

  for n in range(2,15):
      knclf = KNeighborsClassifier(n)

      knclf.fit(X_train, np.ravel(y_train))

      results_test = knclf.predict(X_test)
      results_val = knclf.predict(X_val)
      results_train = knclf.predict(X_train)

      error_train = 1 - accuracy_score(y_train, results_train)
      error_val = 1 - accuracy_score(y_val, results_val)
      error_test = 1 - accuracy_score(y_test, results_test)

      errors.append([n ,error_train, error_val, error_test])

      if error_val < best_error_val:
         best_error_val = error_val
         best_n = n
         best_results = results_test
  
  df_errors = pd.DataFrame(errors, columns=["N", "Error Train", "Error Val", "Error Test"])
  print(df_errors)
  print("The best N: " + str(best_n) + " with error: " + str(best_error_val))

  test.drop(["2", "3"], axis=1)

  plot(test, title="Original")

  # Tree Clasifier
  tree_clf = entrenar(data)
  df_results_tree = clasificar(test, tree_clf)
  df_results_tree.drop(["2", "3"], axis=1)
  plot(df_results_tree, title="Tree")

  # KNN Clasifier
  df_results_knn = X_test.copy()
  df_results_knn["Class"] = best_results
  df_results_knn.drop(["2", "3"], axis=1)
  plot(df_results_knn, title="KNN")

def ejercicio_1():
  data = pd.read_csv(
      "TP_4/c_0.data",
      names=["0", "1", "Class"],
      header=None,
  )

  test = pd.read_csv(
      "TP_4/c_0.test",
      names=["0", "1", "Class"],
      header=None,
  )

  ejercicio_1_no_noise(data, test)

  data_noisy = pd.read_csv(
      "TP_4/c_2.data",
      names=["0", "1", "2", "3", "Class"],
      header=None,
  )

  test_noisy = pd.read_csv(
      "TP_4/c_2.test",
      names=["0", "1", "2", "3", "Class"],
      header=None,
  )

  ejercicio_1_noise(data_noisy, test_noisy)

def ejercicio_2():
   return 
  