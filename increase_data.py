# 引入必要的库
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

def load_dataset(samples):
    feature = samples.data
    dataset = pd.DataFrame(feature)
    dataset.columns = samples.feature_names
    dataset["label"] = samples.target
    return dataset, samples.feature_names


# 根据每一个类的均值、协方差求更正后的均值、协方差
def distribution_calibration(query, base_means, base_cov, k, alpha=0.5):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query - base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov


# 选取小样本，基于此生成数据
def get_min_sample(dataset, feature_name):
    base_means = []
    base_cov = []

    df_group = dataset.groupby("label")
    result = pd.DataFrame(columns=feature_name + ["label"])
    for label, df in df_group:
        # 抽取少量的一部分样本, 10%的比例
        df_temp = df.sample(frac=0.1, axis=0)
        result = result.append(df_temp, ignore_index=True)

        feature = df[feature_name].values
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
    return result, base_means, base_cov


# 生成数据

def generate_dataset(result, base_means, base_cov, feature_name, num_sampled=10, return_origion=False):
    df_group = result.groupby("label")

    sampled_data = []
    sampled_label = []

    sample_num = result.shape[0]
    feature = result[feature_name].values
    label = result["label"].values
    for i in range(sample_num):
        mean, cov = distribution_calibration(feature[i], base_means, base_cov, k=1)
        sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
        sampled_label.extend([label[i]] * num_sampled)

    sampled_data = np.concatenate([sampled_data[:]]).reshape(result.shape[0] * num_sampled, -1)
    result_aug = pd.DataFrame(sampled_data)
    result_aug.columns = feature_name
    result_aug["label"] = sampled_label

    if return_origion:
        result_aug = result_aug.append(result, ignore_index=True)
    return result_aug

def increase_data(samples):
    # 获取iris原数据集
    base_class_dataset, feature_name = load_dataset(samples)

    # 获取数据分布统计量, 参数k和alpha就是超参, k和类别数有关，如果总类别数(num_label)多，这个
    # 可以调大一点，总之k<num_label
    result, base_means, base_cov = get_min_sample(base_class_dataset, feature_name)
    # 生成数据,
    result_aug = generate_dataset(result, base_means, base_cov, feature_name, num_sampled=10)

    list = [base_class_dataset, result_aug]
    new_samples = pd.concat(list)

    X0 = base_class_dataset.drop(columns='label')
    Y0 = base_class_dataset['label']
    X = new_samples.drop(columns='label')
    Y = new_samples['label']

    validation_size = 0.2
    X0_train, X0_validation, Y0_train, Y0_validation = train_test_split(X0, Y0, test_size=validation_size, shuffle=True)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, shuffle=True)
    # 创建模型
    models = {}
    models['LR'] = LogisticRegression(max_iter=10000)
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier(n_neighbors=11)
    models['CART'] = DecisionTreeClassifier(max_depth=25)
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()
    models['RF'] = RandomForestClassifier(max_depth=5, n_estimators=1200)

    # 评估模型
    import warnings
    warnings.filterwarnings("ignore")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    results0 = []
    results = []
    m0 = []
    m = []
    for key in models:
        kflod = KFold(n_splits=10, shuffle=True)
        cv_results0 = cross_val_score(models[key], X0_train, Y0_train, cv=kflod, scoring='accuracy')
        cv_results = cross_val_score(models[key], X_train, Y_train, cv=kflod, scoring='accuracy')
        results0.append(cv_results0)
        results.append(cv_results)
        m0.append(cv_results0.mean())
        m.append(cv_results.mean())
        print('%s:%f(%f)' % (key, cv_results0.mean(), cv_results0.std()))
        print('%s:%f(%f)' % (key, cv_results.mean(), cv_results.std()))
    n = ['逻辑回归', '线性判别', 'K近邻', '决策树', '朴素贝叶斯', '支持向量机', '随机森林']
    plt.ylim(ymin=0, ymax=1)
    plt.plot(n, m0, c='red', label='原始数据集')
    plt.plot(n, m, c='blue', label='增广数据集')
    plt.legend()
    st.pyplot()
