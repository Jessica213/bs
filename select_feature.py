from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


def select_feature(selectmodel,X,y):
    if selectmodel == '逻辑回归':
        model = LogisticRegression(max_iter=10000)
    elif selectmodel == '线性判别分析':
        model = LinearDiscriminantAnalysis()
    elif selectmodel == '决策树':
        model = DecisionTreeClassifier(max_depth=25)
    elif selectmodel == '支持向量机':
        model = SVC(kernel="linear")
    else:
        model = RandomForestClassifier(max_depth=5, n_estimators=1200)
    # 分类
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(X, y)
    st.write("RFE挑选了"+str(rfecv.n_features_)+"个特征")
    # Plot number of features VS. cross-validation scores
    # 画出不同特征数量下交叉认证验证得分
    plt.figure()
    #  选择的特征数量
    plt.xlabel("Number of features selected")
    # 交叉验证得分
    plt.ylabel("Cross validation score (nb of correct classifications)")
    # 画出各个特征的得分
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    st.pyplot()