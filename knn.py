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

      errors.append([n, error_train, error_val, error_test])

      if error_val < best_error_val:
         best_error_val = error_val
         best_n = n
         best_results = results_test
  
  df_errors = pd.DataFrame(errors, columns=["N", "Error Train", "Error Val", "Error Test"])
  print("The best N: " + str(best_n) + " with error: " + str(best_error_val))

  print(df_errors)
  plot_knn_errors(df_errors)
  
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

      errors.append([n, error_train, error_val, error_test])

      if error_val < best_error_val:
         best_error_val = error_val
         best_n = n
         best_results = results_test
  
  df_errors = pd.DataFrame(errors, columns=["N", "Error Train", "Error Val", "Error Test"])
  print("The best N: " + str(best_n) + " with error: " + str(best_error_val))
  
  print(df_errors)
  plot_knn_errors(df_errors)

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

def knn_train(n, X_train, y_train, X_val, y_val, X_test, y_test):
  knclf = KNeighborsClassifier(n)

  knclf.fit(X_train, np.ravel(y_train))

  results_test = knclf.predict(X_test)
  results_val = knclf.predict(X_val)
  results_train = knclf.predict(X_train)

  error_train = 1 - accuracy_score(y_train, results_train)
  error_val = 1 - accuracy_score(y_val, results_val)
  error_test = 1 - accuracy_score(y_test, results_test)

  return error_train, error_val, error_test

def ejercicio_2():
  c = 0.78
  n = 250

  d_values = [2, 4, 8, 16, 32]

  errors = []

  for d in d_values:
      centros_a = centros_eja(d)

      test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)
      X_test, y_test = test_case_a.iloc[:, :-1], test_case_a.iloc[:, -1:]

      test_error_para_1 = 0.0
      values_error_para_1 = 0.0
      test_error_para_k = 0.0
      values_error_para_k = 0.0
      test_error_diag_1 = 0.0
      values_error_diag_1 = 0.0
      test_error_diag_k = 0.0
      values_error_diag_k = 0.0

      for j in range(20):
          values = generar_valores(centros_a, c * sqrt(d), d, n)
          X_raw, y_raw = values.iloc[:, :-1], values.iloc[:, -1:]

          X_train, X_val, y_train, y_val = train_test_split(
              X_raw, y_raw, test_size=0.2, random_state=42
          )

          _, error_val_1, error_test_1 = knn_train(1, X_train, y_train, X_val, y_val, X_test, y_test)
          _, error_val_k, error_test_k = knn_train(2, X_train, y_train, X_val, y_val, X_test, y_test)

          values_error_para_1 += error_val_1
          test_error_para_1 += error_test_1
          values_error_para_k += error_val_k
          test_error_para_k += error_test_k

      test_error_para_1 = test_error_para_1 / 20
      values_error_para_1 = values_error_para_1 / 20
      test_error_para_k = test_error_para_k / 20
      values_error_para_k = values_error_para_k / 20

      centros_b = centros_ejb(d)
      test_case_b = generar_valores(centros_b, c, d, 10000)
      X_test, y_test = test_case_b.iloc[:, :-1], test_case_b.iloc[:, -1:]

      for j in range(20):
          values = generar_valores(centros_b, c, d, n)
          X_raw, y_raw = values.iloc[:, :-1], values.iloc[:, -1:]

          X_train, X_val, y_train, y_val = train_test_split(
              X_raw, y_raw, test_size=0.2, random_state=42
          )

          _, error_val_1, error_test_1 = knn_train(1, X_train, y_train, X_val, y_val, X_test, y_test)
          _, error_val_k, error_test_k = knn_train(2, X_train, y_train, X_val, y_val, X_test, y_test)


          test_error_diag_1 += error_test_1
          values_error_diag_1 += error_val_1
          test_error_diag_k += error_test_k
          values_error_diag_k += error_val_k

      test_error_diag_1 = test_error_diag_1 / 20
      values_error_diag_1 = values_error_diag_1 / 20
      test_error_diag_k = test_error_diag_k / 20
      values_error_diag_k = values_error_diag_k / 20

      errors.append([test_error_para_1, d, "Test_Parallel_KN1"])
      errors.append([values_error_para_1, d, "Val_Parallel_KN1"])
      errors.append([test_error_diag_1, d, "Test_Diagonal_KN1"])
      errors.append([values_error_diag_1, d, "Val_Diagonal_KN1"])

  df_errors = pd.DataFrame(errors, columns=["Error", "D", "Type"])
  df_errors.to_csv("TP_4/errors_ej_2.csv", index=False)


def ejercicio_2_print():
    df_errors_tree = pd.read_csv("TP_1/errors_ej_4.csv")
    df_errors_nn = pd.read_csv("TP_2/errors_ej_5.csv")
    df_errors_bayes = pd.read_csv("TP_3/errors_ej_1.csv")
    df_errors_knn = pd.read_csv("TP_4/errors_ej_2.csv")
    df_errors = pd.concat([df_errors_tree, df_errors_nn, df_errors_bayes, df_errors_knn])
    print(df_errors)
    plot_error_lines_with_dimensions(df_errors)

