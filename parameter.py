from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


def change_parameter(selectmodel,X,y):
    # Number of random trials
    NUM_TRIALS = 30

    if selectmodel == '逻辑回归':
        p_grid = {"penalty": ['l1', 'l2'], "C": [0.1, 1, 10, 100, 1000]}
        e = LogisticRegression()
    elif selectmodel == '线性判别分析':
        p_grid = {}
        e = LinearDiscriminantAnalysis()
    elif selectmodel == 'K近邻':
        l = []
        for i in range(73):
            l.append(i)
        p_grid = {"n_neighbors": l}
        e = KNeighborsClassifier()
    elif selectmodel == '决策树':
        p_grid = {"max_depth": [5, 8, 15, 25, 30]}
        e = DecisionTreeClassifier()
    elif selectmodel=='朴素贝叶斯':
        p_grid = {}
        e = GaussianNB()
    elif selectmodel == '支持向量机':
        p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
        e = SVC()
    else:
        p_grid = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
        e = RandomForestClassifier()

    # Arrays to store scores
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=e, param_grid=p_grid, cv=outer_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=e, param_grid=p_grid, cv=inner_cv)
        nested_score = cross_val_score(clf, X, y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores

    print(
        "Average difference of {:6f} with std. dev. of {:6f}.".format(
            score_difference.mean(), score_difference.std()
        )
    )

    # Plot scores on each trial for nested and non-nested CV
    plt.figure()
    plt.subplot(211)
    (non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
    (nested_line,) = plt.plot(nested_scores, color="b")
    plt.ylabel("score", fontsize="14")
    plt.legend(
        [non_nested_scores_line, nested_line],
        ["Non-Nested CV", "Nested CV"],
        bbox_to_anchor=(0, 0.4, 0.5, 0),
    )
    plt.title(
        "Non-Nested and Nested Cross Validation on Iris Dataset",
        x=0.5,
        y=1.1,
        fontsize="15",
    )

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
    plt.xlabel("Individual Trial #")
    plt.legend(
        [difference_plot],
        ["Non-Nested CV - Nested CV Score"],
        bbox_to_anchor=(0, 1, 0.8, 0),
    )
    plt.ylabel("score difference", fontsize="14")

    st.pyplot()