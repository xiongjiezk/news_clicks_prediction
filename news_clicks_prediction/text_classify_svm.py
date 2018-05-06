#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>
import json
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC

from news_clicks_prediction.utils import load_data


class TextClassifySVM:

    def __init__(self, dataset='sohu', is_multi_class=False):
        self.is_multi_class = is_multi_class
        self.dataset = dataset
        df = load_data(self.dataset, self.is_multi_class)
        _data = self.text_process(df['segmented'])
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(_data, df['labels'])
        self.classifier = self.build_model()
        pass

    def train_test_split(self, data, label):
        X_train, X_test, y_train, y_test = train_test_split(data, label, shuffle=True, test_size=0.3)
        return X_train, X_test, y_train, y_test

    def text_process(self, balanced_texts):
        print("text_process start...")
        # t1 = datetime.now()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=10000)  # max_features = None ,最高
        data = vectorizer.fit_transform(balanced_texts)
        print("data shape: ", data.shape)
        return data

    def build_model(self):
        # use default param
        return LinearSVC()

    def train(self):
        print("svm train start...")
        self.classifier.fit(self.X_train, self.y_train)
        score = cross_val_score(self.classifier, self.X_train, self.y_train, cv=2)  # n_jobs=-1
        # print('cost time: %s ' % (datetime.now() - t1))
        print(score)
        predicted = self.classifier.predict(self.X_test)
        acc = accuracy_score(self.y_test, predicted)
        print(acc)
        result = OrderedDict()
        result['dataset'] = self.dataset
        result['multiclass'] = 4 if self.is_multi_class else 2
        result['acc'] = acc
        self.log_info(json.dumps(result))
        # print(sum(score) / len(score))

    def log_info(self, message):
        with open("logs/debug_log_svm", "a+") as f:
            f.write(message + "\n")

model = TextClassifySVM('sohu', is_multi_class=False)
model.train()