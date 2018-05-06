#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>
import gensim
import pandas as pd
import numpy as np
import datetime
import time
import jieba

DATE_FORMAT = '%Y-%m-%d'

def __to_categorical(pv, is_multi_class):
    if is_multi_class:
        pv = pv if pv < 1e+6 else (1e+6 - 1)
        pv = pv / 100
        pv = pv if pv >= 1 else 1
        return np.floor(np.log10(pv))
    else:
        return 0 if pv < 10000 else 1

def process_post_time(post_time, create_time):
    # * days
    if create_time < post_time:
        return np.zeros(shape=12)
    survival_time = (create_time - post_time) / (6 * 24 * 60 * 60)
    survival_time = 3.0 if survival_time > 3 else survival_time
    dd = datetime.datetime.fromtimestamp(post_time)
    weed_index = dd.weekday()
    ret = np.zeros(shape=32)
    ret[weed_index] = 1.0
    hour_index = np.floor(dd.hour - 1)
    ret[int(7 + hour_index)] = 1.0
    ret[31] = survival_time
    return pd.Series(ret)

def get_date_interval(post_date, start_date):  # input timestamp
    return (datetime.datetime.fromtimestamp(post_date) - start_date).days

def get_distinct_value(texts, separator=None):
    words = set()
    for text in texts:
        if separator is not None:
            text = text.split(separator)
        for x in text:
            words.add(x)
    return words

def __preprocess_hot_count(x):
    x = x if x > 0 else 1
    x = x if x < 10000000 else 10000000
    return round(np.log10(x) / 2, 4)

def load_hot_trend_detail():
    file_path = "E:/data/sohu_news/hot_trend/hot_trend_1.csv"
    df = pd.read_csv(file_path, header=None)
    startDate = datetime.datetime.strptime('2017-12-26', '%Y-%m-%d')
    date_list = [(startDate + datetime.timedelta(i)).strftime('%Y-%m-%d') for i in range(90)]
    df.columns = ['word', 'wid'] + date_list
    print(df.shape)
    df = df.drop_duplicates(keep='last')
    print(df.shape)

    word_index_dict = {word: index for index, word in enumerate(df['word'])}
    df = df.drop(labels=['word', 'wid'], axis=1)
    for column in df.columns:
        df[column] = df[column].apply(lambda x: __preprocess_hot_count(x))
    return word_index_dict, np.array(df)

def load_hot_trend_detail_done():
    t1 = time.time()
    print("load hot trend data....")
    file_path = "../data/hot_trend_done.csv"
    df = pd.read_csv(file_path)
    word_index_dict = {word: index for index, word in enumerate(df['word'])}
    df = df.drop(labels=['word', 'wid'], axis=1)
    print("load hot trend data success, cost: %s, df shape: %s" % (time.time() - t1, str(df.shape)))
    return word_index_dict, np.array(df)

def load_sohunews_2(is_multi_class=False):
    t1 = datetime.datetime.now()
    print("load news data....")
    file_path = "../data/account_user_dict_for_jieba.txt"
    jieba.load_userdict(file_path)

    file_path = "../data/sohu_news_article.csv"
    ori_df = pd.read_csv(file_path)
    ori_df.columns = ['account','source', 'newsid', 'pv', 'commentCnt', 'post_time', 'title', 'tags', 'create_time', 'brief']
    ori_df['post_time'] = ori_df['post_time'] / 1000
    ori_df = ori_df.dropna()
    print('before filter post time, ori_df shape: %s' % str(ori_df.shape))
    # 只保留 2017-12-27 到 2018-03-02之间的文章
    start_date = datetime.datetime.strptime('2017-12-26', DATE_FORMAT).timestamp()
    end_date = datetime.datetime.strptime('2018-03-26', DATE_FORMAT).timestamp()
    ori_df = ori_df[(ori_df.create_time - ori_df.post_time) > (3 * 24 * 60 * 60)]
    ori_df = ori_df[ori_df.post_time > start_date]
    ori_df = ori_df[ori_df.post_time < end_date]
    print('after filter post time, ori_df shape: %s' % str(ori_df.shape))

    df = pd.DataFrame()
    df['account'] = ori_df['account']
    # text = ori_df['source'] + ori_df['tags'] + ori_df['title'] + ori_df['brief']
    text = ori_df['title'] + ori_df['brief']
    text = text.apply(lambda st: st.replace("\n", '').replace(" ", '').replace("\t", '').replace("\r", '')
                      .replace('...', '').replace("&quot;", '').replace('…', ''))
    df['segmented'] = text.apply(lambda st: " ".join(jieba.cut(st)))
    df['text'] = text
    df['tags'] = ori_df['tags']
    df['post_time'] = ori_df['post_time']
    df['create_time'] = ori_df['create_time']
    df['labels'] = ori_df['pv'].apply(lambda pv: __to_categorical(pv, is_multi_class))
    df['pv'] = ori_df['pv']
    print("load data success, cost: %s " % (datetime.datetime.now() - t1))
    print("sample size: %s" % (len(df)))
    return df

def load_toutiaonews_2(is_multi_class=False):
    t1 = datetime.datetime.now()
    print("load data....")
    file_path = "../data/account_user_dict_for_jieba.txt"
    jieba.load_userdict(file_path)

    file_path = "../data/toutiao_news_article.csv"
    ori_df = pd.read_csv(file_path)
    ori_df.columns = ['account', 'source', 'newsid', 'pv', 'post_time', 'title', 'tags', 'create_time', 'brief']
    # ori_df['post_time'] = ori_df['post_time'] / 1000
    ori_df = ori_df.dropna()
    print('before filter post time, ori_df shape: %s' % str(ori_df.shape))
    start_date = datetime.datetime.strptime('2017-12-26', DATE_FORMAT).timestamp()
    end_date = datetime.datetime.strptime('2018-03-26', DATE_FORMAT).timestamp()
    ori_df = ori_df[(ori_df.create_time - ori_df.post_time) > (3 * 24 * 60 * 60)]
    ori_df = ori_df[ori_df.post_time > start_date]
    ori_df = ori_df[ori_df.post_time < end_date]
    print('after filter post time, ori_df shape: %s' % str(ori_df.shape))

    df = pd.DataFrame()
    df['account'] = ori_df['account'].apply(lambda x: str(x))
    # text = ori_df['source'] + ori_df['tags'] + ori_df['title'] + ori_df['brief']
    text = ori_df['title'] + ori_df['brief']
    text = text.apply(lambda st: st.replace("\n", '').replace(" ", '').replace("\t", '').replace("\r", '')
                      .replace('...', '').replace("&quot;", '').replace('…', ''))
    df['segmented'] = text.apply(lambda st: " ".join(jieba.cut(st)))
    df['text'] = text
    df['tags'] = ori_df['tags']
    df['post_time'] = ori_df['post_time']
    df['create_time'] = ori_df['create_time']
    df['labels'] = ori_df['pv'].apply(lambda pv: __to_categorical(pv, is_multi_class))
    df['pv'] = ori_df['pv']
    print("load data success, cost: %s " % (datetime.datetime.now() - t1))
    print("sample size: %s" % (len(df)))
    return df

def load_data(data_set='sohu', is_multi_class=False):
    if data_set == 'sohu':
        return load_sohunews_2(is_multi_class)
    else:
        return load_toutiaonews_2(is_multi_class)

def load_zh_word_vector():
    t1 = datetime.datetime.now()
    print("load zhwiki word embedding_index start...")
    embeddings_index = {}
    f = open('../data/zhwiki_2017_03.sg_50d.word2vec', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print("load zhwiki word embedding_index done, cost: %s" % (datetime.datetime.now() - t1))
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index

def load_zh_word_vector_2():
    start = time.time()
    print("load zhwiki word embedding_index start...")
    file_path = '../data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

    cost = time.time() - start
    print("load word vector dict success. cost time: %s" % cost)
    # word_vectors.word_vec()
    return word_vectors