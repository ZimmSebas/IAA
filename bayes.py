from sklearn.metrics import mean_squared_error, zero_one_loss
from sklearn.naive_bayes import GaussianNB
from data_gen import *
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.model_selection import train_test_split
from csv import reader
from sys import maxsize


def bayes_train(X_train, y_train, X_test, y_test):
  clf = GaussianNB()

  clf.fit(X_train, np.ravel(y_train))

  results_train = clf.predict(X_train)
  results_test = clf.predict(X_test)

  error_train = zero_one_loss(results_train, y_train)
  error_test = zero_one_loss(results_test, y_test)

  #print(error_train)

  return error_train, error_test


def ejercicio_1():
  c = 0.78
  n = 250

  d_values = [2, 4, 8, 16, 32]

  errors = []

  for d in d_values:
    centros_a = centros_eja(d)

    test_case_a = generar_valores(centros_a, c * sqrt(d), d, 10000)        
    X_test, y_test = test_case_a.iloc[:, :-1], test_case_a.iloc[:, -1:]

    test_error_para = 0.0
    values_error_para = 0.0
    test_error_diag = 0.0
    values_error_diag = 0.0

    for j in range(20):
      values = generar_valores(centros_a, c * sqrt(d), d, n)
      X_train, y_train = values.iloc[:, :-1], values.iloc[:, -1:]

      t_errors, v_errors = bayes_train(X_train, y_train, X_test, y_test)

      test_error_para += t_errors
      values_error_para += v_errors

    test_error_para = test_error_para / 20
    values_error_para = values_error_para / 20


    centros_b = centros_ejb(d)
    test_case_b = generar_valores(centros_b, c, d, 10000)
    X_test, y_test = test_case_b.iloc[:, :-1], test_case_b.iloc[:, -1:]

    for j in range(20):
      values = generar_valores(centros_b, c, d, n)
      X_train, y_train = values.iloc[:, :-1], values.iloc[:, -1:]

      t_errors, v_errors = bayes_train(X_train, y_train, X_test, y_test)

      test_error_diag += t_errors
      values_error_diag += v_errors

    test_error_diag = test_error_diag / 20
    values_error_diag = values_error_diag / 20

    errors.append([test_error_para, d, "Test_Parallel_BY"])
    errors.append([values_error_para, d, "Val_Parallel_BY"])
    errors.append([test_error_diag, d, "Test_Diagonal_BY"])
    errors.append([values_error_diag, d, "Val_Diagonal_BY"])
  
  df_errors = pd.DataFrame(errors, columns=["Error", "D", "Type"])
  df_errors.to_csv("TP_3/errors_ej_1.csv", index=False)

def ejercicio_1_print():
  df_errors_tree = pd.read_csv("TP_1/errors_ej_4.csv")
  df_errors_nn = pd.read_csv("TP_2/errors_ej_5.csv")
  df_errors_bayes = pd.read_csv("TP_3/errors_ej_1.csv")
  df_errors = pd.concat([df_errors_tree,df_errors_nn,df_errors_bayes])
  print(df_errors)
  plot_error_lines_with_dimensions(df_errors)
