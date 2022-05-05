import streamlit as st
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from itertools import cycle
from sklearn.datasets import base
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV

from fill_dependents_missing import fill_dependents_missing
from increase_data import increase_data
from parameter import change_parameter
from select_feature import select_feature

st.title("不孕症中医分型机器学习平台")
#一些说明
st.markdown("本平台采用的数据集来自中日友好医院中医妇科门诊2018年10月——2022年3月期间诊断为不孕症的女性患者，年龄为25-45岁，平均（31.23±5.43)岁。"
            "有效数据600例，根据78种症候将其分类为6种常见不孕症中医证型，分别为："
            "肾气虚证、肾阳虚证、肾阴虚证、肝气郁结证、 瘀滞胞宫证、痰湿内阻证。")

#探索数据模块
if st.sidebar.checkbox('探索数据'):
    st.header("一、探索数据")
    #导入数据集
    df=pd.read_excel('E:\document\Bishe\data\不孕症数据总表.xlsx')
    df=pd.DataFrame(df)
    #检查有无重复行
    st.write("1.检查有无重复行")
    st.write(df[df.duplicated()])
    #筛选出有缺失值的行
    st.markdown("2.筛选出有缺失值的行")
    st.write(df[df.isna().T.any()])
    #用模型填补缺失值
    fill_dependents_missing(df, '舌无苔')
    fill_dependents_missing(df, '舌厚苔')
    fill_dependents_missing(df, '舌腻苔')
    fill_dependents_missing(df, '舌苔白')
    fill_dependents_missing(df, '舌苔黄')
    fill_dependents_missing(df, '舌苔黄白相兼')
    fill_dependents_missing(df, '舌苔灰黑')
    fill_dependents_missing(df, '舌体胖')
    st.success("填补缺失值完毕！")
    st.dataframe(df)

    classinformation=df['分型'].tolist()
    df1=df.drop(columns=['分型'])
    array=np.array(df1)
    feature=array.tolist()
    class_names=["1","2","3","4","5","6"]
    feature_name=['月经不调', '月经停闭', '月经周期异常', '经量异常', '经期延长', '月经色淡', '月经黯红', '经色鲜红',
           '经色紫黯', '经行血块', '经行腹痛', '经行不畅', '带下量少', '带下量多', '带下粘稠', '带下清稀', '阴中干涩',
           '性欲冷淡', '胞宫发育异常', '眼眶暗', '面色黯沉', '小腹冷', '畏寒', '神疲乏力', '头晕耳鸣', '腰膝酸软',
           '形体消瘦', '形体肥胖', '纳差', '盗汗', '失眠多梦', '五心烦热', '面部痤疮', '肌肤失润', '烦躁易怒',
           '精神抑郁', '胸胁乳房胀痛', '善太息', '腹胀', '胸闷泛恶', '面目虚浮', '咽干口渴', '性交痛', '肛门坠胀不适',
           '小便清长', '大便不成形', '大便干燥', '舌淡白', '舌淡红', '舌红', '舌绛', '舌紫暗', '舌暗红', '舌淡紫',
           '舌边尖红', '舌有瘀斑瘀点', '少苔', '舌无苔', '舌厚苔', '舌腻苔', '舌苔白', '舌苔黄', '舌苔黄白相兼',
           '舌苔灰黑', '舌体胖', '舌体瘦', '舌有齿痕', '脉浮', '脉沉', '脉弱', '脉缓', '脉涩', '脉迟', '脉滑',
           '脉结', '脉虚', '脉细', '脉弦']
    samples=base.Bunch(data=feature,target=classinformation,target_names=class_names,feature_names=feature_name)
    X = pd.DataFrame(samples.data, columns=samples.feature_names)
    Y = pd.DataFrame(samples.target, columns=['分型']).iloc[:, 0]
    # 分离数据集
    validation_size = 0.2
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, shuffle=True)

#创建模型模块
if st.sidebar.checkbox('快速建模'):
    st.header("二、创建模型")
    # 多选框
    selectmodel = st.sidebar.selectbox(
        '请选择一个模型',
        ('逻辑回归', '线性判别分析', 'K近邻', '决策树', '朴素贝叶斯', '支持向量机', '随机森林')
    )
    st.success("您选择了"+selectmodel+"模型！")
    if selectmodel=='逻辑回归':
        p = st.sidebar.slider('请选择参数max_iter的大小', 0, 20000, 10000)
        model = LogisticRegression(max_iter=p)
    elif selectmodel=='线性判别分析':
        model = LinearDiscriminantAnalysis()
    elif selectmodel=='K近邻':
        p = st.sidebar.slider('请选择参数n_neighbors的大小', 0, 130, 50)
        model = KNeighborsClassifier(n_neighbors=p)
    elif selectmodel=='决策树':
        p = st.sidebar.slider('请选择参数max_depth的大小', 0, 130, 8)
        model = DecisionTreeClassifier(max_depth=p)
    elif selectmodel=='朴素贝叶斯':
        model = GaussianNB()
    elif selectmodel=='支持向量机':
        model = SVC()
    elif selectmodel=='随机森林':
        p1 = st.sidebar.slider('请选择参数max_depth的大小', 0, 130, 5)
        p2 = st.sidebar.slider('请选择参数n_neighbors的大小', 0, 2000, 1200)
        model= RandomForestClassifier(max_depth=p1,n_estimators=p2)

    warnings.filterwarnings("ignore")
    kflod = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kflod, scoring='accuracy')
    st.write("该模型的平均准确率为：")
    st.write(cv_results.mean())
    st.write("该模型的标准差为：")
    st.write(cv_results.std())

    # 预测结果可视化
    st.write("ROC曲线如下：")
    # 加载数据
    x = samples.data
    y = samples.target
    # 将标签二值化
    # y = label_binarize(y, classes=['肾气虚', '肾阳虚', '肾阴虚', '肝气郁结', '瘀滞胞宫', '痰湿内阻'])
    y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6])
    # 设置种类
    n_classes = y.shape[1]

    # 训练模型并预测
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=0)

    # Learn to predict each class against the other
    # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
    #                                 random_state=random_state))
    if selectmodel == '逻辑回归':
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=p))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    elif selectmodel == '线性判别分析':
        classifier = OneVsRestClassifier(LinearDiscriminantAnalysis())
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    elif selectmodel == 'K近邻':
        classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=p))
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    elif selectmodel == '决策树':
        classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=p))
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    elif selectmodel == '朴素贝叶斯':
        classifier = OneVsRestClassifier(GaussianNB())
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    elif selectmodel == '支持向量机':
        classifier = OneVsRestClassifier(SVC())
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        classifier = OneVsRestClassifier(RandomForestClassifier(max_depth=p1, n_estimators=p2))
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#调优模块
if st.sidebar.checkbox('模型调优'):
    st.header("三、模型调优")
    st.write("1.运用超参数调优")
    X=samples.data
    y=samples.target
    change_parameter(selectmodel,X,y)

    st.write("2.运用特征选择")
    x = pd.DataFrame(samples.data)
    y = pd.DataFrame(samples.target)
    st.write("原始数据集的容量：")
    st.write(x.shape)
    selector = VarianceThreshold()
    x1 = selector.fit_transform(x)
    st.write("移除零方差特征后的容量：")
    st.write(x1.shape)

    X=x1
    y = pd.DataFrame(samples.target, columns=['分型']).iloc[:, 0]

    st.write("特征选择曲线如下：")
    select_feature(selectmodel, X, y)

    st.write("3.运用数据增广")
    increase_data(samples)
