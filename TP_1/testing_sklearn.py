from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from data_gen import *


def test_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_impurity_decrease=0.005,
        random_state=0,
        min_samples_leaf=5,
    )
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
