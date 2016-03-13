# -*- coding: utf-8 -*-

import os
import pydot
import graphviz
import sys

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image

def download_tree_model_2_pdf():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
    ##数据缓存变量
    dot_data = StringIO()
    #把树模型信息以dot格式输出到dot_data变量中
    tree.export_graphviz(clf, out_file=dot_data)
    #把dot格式的数据转变成graph格式的数据
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #通过Graphviz executor 把graph转换成pdf文档  在转换之前要手动安装Graphviz 该安装包可以在Graphviz官网下载
    graph.write_pdf('iris.pdf')
    # print clf.predict([[1.0, 5.0, 6.0, 5.]])


def display_tree_model_by_image():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    Image(graph.)
    a = input("Please input your name2")
    print "hello"

if __name__ == "__main__":
    # download_tree_model_2_pdf()
    display_tree_model_by_image()
    # os.unlink()