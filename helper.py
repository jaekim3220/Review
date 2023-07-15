import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer


def getIq(field):
    q1 = field.quantile(q=0.25)
    q3 = field.quantile(q=0.75)
    iqr = q3 - q1
    하한 = q1 - 1.5 * iqr
    상한 = q3 + 1.5 * iqr
    결측치경계 = [하한, 상한]
    return 결측치경계


def replaceMissingValue(df):
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imr = imr.fit_transform(df.values)
    re_df = DataFrame(df_imr, index=df.index, columns=df.columns)
    return re_df


def replaceOutlier(df, fieldName):
    cdf = df.copy()

    # fieldName이 List가 아니면 List로 변환
    if not isinstance(fieldName, list):
        fieldName = [fieldName]

    for f in fieldName:
        결측치경계 = getIq(cdf[f])
        cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
        cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

    return cdf


def clearStopwords(nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"):
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        
        for i, v in enumerate(stopwords):
            stopwords[i] = v.strip()

    data_set = []

    for v in nouns:
        if v not in stopwords:
            data_set.append(v)

    return data_set