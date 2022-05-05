from sklearn.ensemble import RandomForestRegressor

def fill_dependents_missing(data, to_fill):
    # 构建新的特征和标签
    # df = data.copy()
    df = data
    # 对第0行到第599行进行修改
    df.loc[0:99, ('分型')] = [[1]]
    df.loc[100:199, ('分型')] = [[2]]
    df.loc[200:299, ('分型')] = [[3]]
    df.loc[300:399, ('分型')] = [[4]]
    df.loc[400:499, ('分型')] = [[5]]
    df.loc[500:599, ('分型')] = [[6]]
    columns = [*df.columns]
    # columns.remove(to_fill)    # 将需要预测的标签移除
    # 移除有缺失值的列
    list = ['舌无苔', '舌厚苔', '舌腻苔', '舌苔白', '舌苔黄', '舌苔黄白相兼', '舌苔灰黑', '舌体胖']
    for i in list:
        columns.remove(i)
    x = df.loc[:, columns]
    y = df.loc[:, to_fill]
    x_train = x.loc[df[to_fill].notnull()]
    x_pred = x.loc[df[to_fill].isnull()]
    y_train = y.loc[df[to_fill].notnull()]
    model = RandomForestRegressor(random_state=0,
                                  n_estimators=200,
                                  max_depth=3,
                                  n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_pred)
    df.loc[df[to_fill].isnull(), to_fill] = pred
    return df