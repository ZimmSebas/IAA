from sklearn.metrics import zero_one_loss, accuracy_score
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

    error_train = 1-accuracy_score(results_train, y_train)
    error_test = 1-accuracy_score(results_test, y_test)

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

def entrenar_CategoricalNB(list_bins, X_train, y_train, X_val, y_val, X_test, y_test ):  
    errors = []
    best_val_error = 1

    for bins in list_bins:
        kbdisc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        kbdisc.fit(X_train)

        X_train_discrete = kbdisc.transform(X_train.copy())
        X_val_discrete = kbdisc.transform(X_val.copy())
        X_test_discrete = kbdisc.transform(X_test.copy())

        bnet = CategoricalNB(min_categories=bins)
        bnet.fit(X_train_discrete, np.ravel(y_train))

        results_train = bnet.predict(X_train_discrete)
        results_val = bnet.predict(X_val_discrete)
        results_test= bnet.predict(X_test_discrete)

        train_error = 1 - accuracy_score(y_train, results_train)
        val_error = 1 - accuracy_score(y_val, results_val)
        test_error = 1 - accuracy_score(y_test, results_test)

        errors.append([train_error, bins, "Train error"])
        errors.append([val_error, bins, "Validation error"])
        errors.append([test_error, bins, "Test error"])

        if val_error < best_val_error:
            best_val_error = val_error
            best_bins = bins
            best_bnet = deepcopy(bnet)
            best_kbdisc = deepcopy(kbdisc)

    df_errors = pd.DataFrame(errors, columns = ["Error", "Bins", "Clase"])

    return best_bins, df_errors, best_bnet, best_kbdisc

def ejercicio_4_elipses():
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
    
    X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    # Classic Bayes
    
    clf = GaussianNB()
    clf.fit(X_train, np.ravel(y_train))

    results_test = clf.predict(X_test)

    df_results_bayes = X_test.copy()
    df_results_bayes["Class"] = results_test

    ## Bins discrete Bayes

    list_bins = range(2,100,2)

    best_bins, df_errors, best_bnet, best_kbdisc = entrenar_CategoricalNB(list_bins, X_train, y_train, X_val, y_val, X_test, y_test )

    X_test_discrete = best_kbdisc.transform(X_test.copy())
    results_bins_test = best_bnet.predict(X_test_discrete)

    df_results_bins_bayes = X_test
    df_results_bins_bayes["Class"] = results_bins_test

    print("Best result was "+ str(best_bins))
    plot(test, "Original results")
    plot(df_results_bayes, "Bayes results")
    plot(df_results_bins_bayes, "Bayes bins results")

    plot_error_bins(df_errors, title="Errors with bins")


def ejercicio_4_espirales():

    test = generar_espirales(2000)
    data = generar_espirales(600)

    X_raw, y_raw = data.iloc[:, :-1], data.iloc[:, -1:]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    # Classic Bayes
    
    clf = GaussianNB()
    clf.fit(X_train, np.ravel(y_train))

    results_test = clf.predict(X_test)

    df_results_bayes = X_test.copy()
    df_results_bayes["Class"] = results_test

    ## Bins discrete Bayes

    list_bins = range(2,100,2)

    best_bins, df_errors, best_bnet, best_kbdisc = entrenar_CategoricalNB(list_bins, X_train, y_train, X_val, y_val, X_test, y_test )

    X_test_discrete = best_kbdisc.transform(X_test.copy())
    results_bins_test = best_bnet.predict(X_test_discrete)

    df_results_bins_bayes = X_test
    df_results_bins_bayes["Class"] = results_bins_test

    print("Best result was "+ str(best_bins))
    plot(test, "Original results")
    plot(df_results_bayes, "Bayes results")
    plot(df_results_bins_bayes, "Bayes bins results")

    plot_error_bins(df_errors, title="Errors with bins")