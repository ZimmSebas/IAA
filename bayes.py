from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from data_gen import *
from neural_net import crear_red, entrenar_red
from copy import deepcopy
from matplotlib import pyplot as mpl
from sklearn.model_selection import train_test_split

def bayes_train(X_train, y_train, X_test, y_test):
    clf = GaussianNB()

    clf.fit(X_train, np.ravel(y_train))

    results_train = clf.predict(X_train)
    results_test = clf.predict(X_test)

    error_train = zero_one_loss(results_train, y_train)
    error_test = zero_one_loss(results_test, y_test)

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

            v_errors, t_errors = bayes_train(X_train, y_train, X_test, y_test)

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

            v_errors, t_errors = bayes_train(X_train, y_train, X_test, y_test)

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
    df_errors = pd.concat([df_errors_tree, df_errors_nn, df_errors_bayes])
    print(df_errors)
    plot_error_lines_with_dimensions(df_errors)


def ejercicio_2_elipses():
    columns = ["0","1","Class"]

    data = pd.read_csv(
        "TP_2/dos_elipses.data",
        names=columns,
        header=None,
    )

    test = pd.read_csv(
        "TP_2/dos_elipses.test",
        names=columns,
        header=None,
    )

    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    clf = GaussianNB()
    clf.fit(X_train, np.ravel(y_train))

    results_test = clf.predict(X_test)

    df_results_bayes = X_test
    df_results_bayes["Class"] = results_test

    df_results_nn = pd.read_csv("TP_3/nn_results_elipses_2.csv")

    plot(test, "Original results")
    plot(df_results_nn, "NN results")
    plot(df_results_bayes, "Bayes results")

def ejercicio_2_espirales():

    test = generar_espirales(2000)
    data = generar_espirales(600)

    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]


    clf = GaussianNB()
    clf.fit(X_train, np.ravel(y_train))

    results_test = clf.predict(X_test)

    df_results_bayes = X_test
    df_results_bayes["Class"] = results_test

    df_results_nn = pd.read_csv("TP_3/nn_results_espirales_2.csv")

    plot(test, "Original results")
    plot(df_results_nn, "NN results")
    plot(df_results_bayes, "Bayes results")

def ejercicio_3():
    n_bins = [2,3,4,5,6,7,8]
    for bins in n_bins:
        kbdisc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        kbdisc.fit(X_train)

    X_train_disc = kbdisc.transform(X_train.copy())
    X_val_disc = kbdisc.transform(X_val.copy())
    X_test_disc = kbdisc.transform(X_test.copy())

    clf = CategoricalNB(min_categories=bins)
    clf.fit(X_train_disc, y_train)

    predict_train = clf.predict(X_train_disc)
    predict_val = clf.predict(X_val_disc)
    predict_test= clf.predict(X_test_disc)

    actual_train_error = 1 - accuracy_score(y_train, predict_train)
    actual_val_error = 1 - accuracy_score(y_val, predict_val)
    actual_test_error = 1 - accuracy_score(y_test, predict_test)

    errors.append([actual_train_error, bins, "Train error"])
    errors.append([actual_val_error, bins, "Validation error"])
    errors.append([actual_test_error, bins, "Test error"])

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_bins = bins
      best_clf = deepcopy(clf)
      best_kbdisc = deepcopy(kbdisc)

    errors_df = pd.DataFrame(errors, columns = ["Error", "Bins", "Class"])

    return best_bins, best_clf, best_kbdisc, errors_df 
