#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>
from collections import Counter, OrderedDict

from keras import Input, Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Flatten, Dropout, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight

from news_clicks_prediction.FMLayer import FMLayer
from news_clicks_prediction.attention_base import Attention
from news_clicks_prediction.attention_hot_term import AttentionHotTerm
from news_clicks_prediction.utils import load_hot_trend_detail_done, load_data, get_distinct_value, get_date_interval, \
    process_post_time, load_zh_word_vector, load_zh_word_vector_2
import pandas as pd
import numpy as np
import datetime
import time
import json
import math
import matplotlib.pyplot as plt
import scipy.sparse as sps



EPOCHS = 30
ACCOUNT_SIZE = 1000
ACCOUNT_EMBEDDING_DIM = 20

VOCABULARY_SIZE = 70000
# VOCA_EMBEDDING_DIM = 100
VOCA_EMBEDDING_DIM = 50
VOCA_PADDING_SIZE = 60

CHAR_SIZE = 5000
CHAR_EMBEDDING_DIM = 50
CHAR_PADDING_SIZE = 100

TEST_SIZE = 2000

DATE_FORMAT = '%Y-%m-%d'

class TextClassifyDNN:
    punc_dict = {'？': 3.0, '！': 3.0, '“': 2.0, '”': 2.0, '：': 1.0, '。': 1.0, '，': 1.0, '；': 1.0, '、': 1.0}

    def __init__(self, model_name, dataset='sohu', is_regress=False, is_multi_class=False, is_omit_em=False, is_omit_tr=False):

        self.model_name = model_name
        self.epochs = EPOCHS
        self.is_regress = is_regress
        self.is_multi_class = is_multi_class
        self.dataset = dataset
        self.trend_word_index_dict, self.trend_matrix = load_hot_trend_detail_done()

        df = load_data(self.dataset, self.is_multi_class)

        self.account_size = len(np.unique(df['account']))
        self.account_padding_size = 1
        self.account_embedding_dim = 20
        self.char_size = len(get_distinct_value(df['text']))
        self.char_embedding_dim = 50
        self.char_padding_size = 100
        self.tag_size = len(get_distinct_value(df['tags']))
        self.tag_embedding_dim = 50
        self.tag_padding_size = 3
        self.word_size = len(get_distinct_value(df['segmented'], separator=' '))
        # self.word_size = len(self.voca_tokenizer.word_index) + 1
        self.word_embedding_dim = 50
        self.word_padding_size = 80
        self.sample_size = len(df)
        self.end_date_str = '2018-03-25'
        self.start_date_str = '2017-12-26'  # interval 67 days  span 时间跨度， 从span中选取30天
        self.end_date = datetime.datetime.strptime(self.end_date_str, DATE_FORMAT)
        self.start_date = datetime.datetime.strptime(self.start_date_str, DATE_FORMAT)
        # self.hot_trend_span = (datetime.datetime.strptime(self.end_date, DATE_FORMAT) - datetime.datetime.strptime(self.start_date, DATE_FORMAT)).days + 1
        self.hot_trend_span = 90
        self.hot_trend_dim = 80
        self.decay_factor = 0.97
        self.trend_lambda = 0.5

        self._text = np.array(df['text'])
        self._pv = np.array(df['pv'])
        # self.hot_index_data = self.hot_index_process(_segmented_text)
        _index_df = pd.DataFrame()
        _index_df['start_index'] = df['post_time'].apply(lambda post_time: get_date_interval(post_time, self.start_date))
        _index_df['end_index'] = df['create_time'].apply(lambda create_time: get_date_interval(create_time, self.start_date))
        self.hot_index_data = self.hot_trend_process(np.array(df['segmented']), np.array(_index_df), is_omit=is_omit_tr)

        self.post_time_data = np.array(df.apply(lambda row: process_post_time(row["post_time"], row['create_time']), axis=1))
        self.labels = to_categorical(df['labels'])
        self.num_classes = self.labels.shape[1]
        self.class_weight = self.build_class_weight(df['labels'])

        print("class count: {}".format(Counter(df['labels']).items()))
        print("sample size: %s, num_classes: %s" % (self.labels.shape[0], self.num_classes))

         # todo
        self.voca_data, self.voca_tokenizer = self.voca_process(df['segmented'])
        self.account_data, self.account_tokenizer = self.convert_account(df['account'])
        self.tag_data, self.tag_tokenizer = self.tag_process(df['tags'])
        self.shuffle_data()

        self.data = {'word_input': self.voca_data,
                     'hot_index_input': self.hot_index_data,
                     'account_input': self.account_data,
                     'post_time_input': self.post_time_data,
                     'tag_input': self.tag_data,
                     '_text': self._text,
                     '_pv': self._pv}
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = self.train_valid_split(self.data, size=TEST_SIZE)

        self.vocabulary_size = len(self.voca_tokenizer.word_index) + 1
        self.embedding_matrix = self.build_embedding_matrix(is_omit=is_omit_em)
        # self.embedding_matrix = self.build_news_embedding_matrix()

        self.print_model_config()
        self.last_layer_activation = 'softmax' if is_multi_class else 'sigmoid'
        self.model = self.build_model()

    def print_model_config(self):
        for item in self.__dict__.items():
            if isinstance(item[1], int) or isinstance(item[1], str):
                print('%s:%s' % item)

    def build_class_weight(self, _labels):
        cw = class_weight.compute_class_weight('balanced', np.unique(_labels), _labels)
        return {index: weight for index, weight in enumerate(cw)}

    def shuffle_data(self,  seed=17):
        np.random.seed(seed)
        indices = np.arange(self.sample_size)
        np.random.shuffle(indices)

        self.hot_index_data = self.hot_index_data[indices]
        self.voca_data = self.voca_data[indices]
        self.account_data = self.account_data[indices]
        self.tag_data = self.tag_data[indices]
        self.post_time_data = self.post_time_data[indices]
        self.labels = self.labels[indices]
        self._text = self._text[indices]
        self._pv = self._pv[indices]

    def plot_result(self, history):
        plt.style.use("ggplot")
        plt.figure()
        N = len(history.history["loss"])
        plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on %s" % self.model_name)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()
        plt.savefig("models/acc_cnn.png")  # http://www.cnblogs.com/skyfsm/p/8051705.html
        pass

    def train_valid_split(self, data, size=500):
        if isinstance(data, dict):
            train_data = dict()
            valid_data = dict()
            for key, value in data.items():
                if sps.issparse(value):
                    valid_data[key] = value[:size, :]
                    train_data[key] = value[size:, :]
                else:
                    valid_data[key] = value[:size]
                    train_data[key] = value[size:]
        else: # list
            if sps.issparse(data):
                train_data = data[size:, :]
                valid_data = data[:size, :]
            else:
                train_data = data[size:]
                valid_data = data[:size]
        train_labels = self.labels[size:]
        valid_labels = self.labels[:size]
        return train_data, train_labels, valid_data, valid_labels

    def get_valid_data(self, i):
        item_values = [np.array2string(self.valid_labels[i])]
        keys = ['post_time_input', '_text', '_pv']
        if isinstance(self.valid_data, dict):
            for key in keys:
                item_values.append(key)
                if isinstance(self.valid_data[key][i], np.ndarray):
                    item_values.append(np.array2string(self.valid_data[key][i]))
                else:
                    item_values.append(str(self.valid_data[key][i]))
        else:
            item_values.append(str(self.valid_data[i]))
        return item_values


    def voca_process(self, texts):
        t1 = time.time()
        tokenizer = Tokenizer(num_words=round(0.85 * self.word_size), filters='\t\n')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=self.word_padding_size)
        print("voca_process done, cost: %s" % (time.time() - t1))
        print("data type: %s, data shape: %s" % (type(data), data.shape))
        return data, tokenizer

    def tag_process(self, tags):
        t1 = time.time()
        tokenizer = Tokenizer(num_words=self.tag_size)
        tokenizer.fit_on_texts(tags)
        sequences = tokenizer.texts_to_sequences(tags)
        data = pad_sequences(sequences, maxlen=self.tag_padding_size)
        print("tag_process done, cost: %s" % (time.time() - t1))
        print("data type: %s, data shape: %s" % (type(data), data.shape))
        return data, tokenizer

    def build_embedding_matrix(self, is_omit=False):
        if is_omit:
            return np.random.random((self.vocabulary_size, self.word_embedding_dim))
        t1 = time.time()
        word_vectors = load_zh_word_vector()
        embedding_matrix = np.random.random((self.vocabulary_size, self.word_embedding_dim))
        for word, i in self.voca_tokenizer.word_index.items():
            embedding_vector = word_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print("build embedding matrix success, cost: ", (time.time() - t1))
        return embedding_matrix

    def build_news_embedding_matrix(self, is_omit=False):
        if is_omit:
            return np.random.random((self.vocabulary_size, self.word_embedding_dim))
        t1 = time.time()
        word_vectors = load_zh_word_vector_2()
        embedding_matrix = np.random.random((self.vocabulary_size, self.word_embedding_dim))
        loss_hit_num = 0
        for word, i in self.voca_tokenizer.word_index.items():
            embedding_vector = None
            try:
                embedding_vector = word_vectors.word_vec(word)
            except KeyError as e:
                print('error: %s ' % e)
                loss_hit_num += 1

            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print("build embedding matrix success, cost: %s, word index size : %s, loss hit size: %s",
              ((time.time() - t1), self.vocabulary_size, loss_hit_num))
        return embedding_matrix

    def convert_account(self, account):
        t1 = time.time()
        tokenizer = Tokenizer(num_words=self.account_size)
        tokenizer.fit_on_texts(account)
        sequences = tokenizer.texts_to_sequences(account)
        data = pad_sequences(sequences, maxlen=1)
        print("char_process done, cost: %s" % (time.time() - t1))
        print("data type: %s, data shape: %s" % (type(data), data.shape))
        return data, tokenizer

    def __zip_ngrams(self, arr, n):
        z = [iter(arr[i:]) for i in range(n)]
        return zip(*z)

    def punctuation_process(self, s):
        return TextClassifyDNN.punc_dict[s]

    def number_process(self, s):
        return 3.0

    def is_punctuation(self, s):
        return s in TextClassifyDNN.punc_dict

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            if len(s) > 1:
                for c in s:
                    unicodedata.numeric(c)
                return True
            else:
                unicodedata.numeric(s)
                return True
        except (TypeError, ValueError):
            pass

        return False

    def hot_trend_process(self, texts, _index_array, is_omit=False):
        """
        combine tfidf, clac hot index
        :param texts:
        :return:  np.array  (sample_size, padding_size)
        """
        if is_omit:
            return np.zeros(shape=(self.sample_size, self.word_padding_size, self.hot_trend_dim))

        t1 = time.time()
        vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w*\\b')
        tf_idf_matrix = vectorizer.fit_transform(texts)
        words = vectorizer.get_feature_names()

        data_index = 0
        hot_array = np.zeros(shape=(self.sample_size, self.word_padding_size, self.hot_trend_dim))  # ?*60*30
        hit_stat = []
        # for row_index in range(len(texts)):
        #     col = 0
        #     for word in texts[row_index].split(' '):
        #         if col >= VOCA_PADDING_SIZE:
        #             break
        #         if word in hot_index_dict:
        #             hot_index = hot_index_dict[word]
        #             hot_array[row_index][col] = hot_index[0] * 2
        #         elif self.is_number(word):
        #             hot_array[row_index][col] = self.number_process(word)
        #         elif self.is_punctuation(word):
        #             hot_array[row_index][col] = self.punctuation_process(word)
        #         else:
        #             hot_array[row_index][col] = 1.0
        #         col += 1
            # print(texts[row_index])
            # print(hot_array[row_index])
        for row_index, row_point in enumerate(self.__zip_ngrams(tf_idf_matrix.indptr, 2)):  # 每一行,起止index
            indices = tf_idf_matrix.indices
            row_tf_idf_dict = {}
            for i in indices[row_point[0]:row_point[1]]:
                word = words[i]
                row_tf_idf_dict[word] = tf_idf_matrix.data[data_index]
                data_index += 1

            col = 0
            not_hit = 0
            start_end = _index_array[row_index]
            if start_end[1] - start_end[0] > self.hot_trend_dim:
                start_index = 0
                generate_normal_size = self.hot_trend_dim
            else:
                start_index = start_end[0] - start_end[1]
                generate_normal_size = abs(start_index)

            for word in texts[row_index].split(' '):
                if col >= self.word_padding_size:
                    break
                hit = False
                if word in self.trend_word_index_dict:
                    hit = True
                    word_index = self.trend_word_index_dict[word]
                    hot_array[row_index][col][start_index:] = self.trend_matrix[word_index][start_end[0]: start_end[0] + generate_normal_size]
                    trans = lambda x: np.random.normal(x, 0.5, size=1)
                    hot_array[row_index][col][start_index:] = np.squeeze(list(map(trans, hot_array[row_index][col][start_index:])))
                    hot_array[row_index][col][start_index:] *= row_tf_idf_dict.get(word, 0.5)
                elif self.is_number(word):
                    hot_array[row_index][col][start_index:] = self.number_process(word) * 0.5 \
                                                              * np.random.normal(1.0, 0.5, size=generate_normal_size)
                elif self.is_punctuation(word):
                    hot_array[row_index][col][start_index:] = self.punctuation_process(word) * 0.5 \
                                                              * np.random.normal(1.0, 0.5, size=generate_normal_size)
                else:
                    hot_array[row_index][col][start_index:] = row_tf_idf_dict.get(word, 0.5) * 2 \
                                                              * np.random.normal(1.0, 0.5, size=generate_normal_size)
                    not_hit += 1
                hot_array[row_index][col][start_index:] *= [math.pow(self.decay_factor, day) * self.trend_lambda for day in range(generate_normal_size)]
                # print(word + ('*%s*' % hit) + texts[row_index])
                # print(hot_array[row_index][col][:])
                col += 1
            hit_stat.append(str(round(1 - not_hit / col, 2)))
            # print(texts[row_index])
            # print(hot_array[row_index])
        print("build hot trend matrix success, cost: ", (time.time() - t1))
        print(' '.join(hit_stat[1:300]))
        return hot_array

    def build_model(self):

        if self.model_name == 'rnn':
            return self.build_rnn_model()
        elif self.model_name == 'cnn':
            return self.build_cnn_model()
        elif self.model_name == 'rcnn':
            return self.build_rcnn_model()
        elif self.model_name == 'deep_fm':
            return self.build_deep_fm_model()
        elif self.model_name =='attention':
            return self.build_attention_model()
        elif self.model_name =='trend':
            return self.build_trend_model()
        elif self.model_name == 'attention_trend':
            return self.build_attention_trend()
        else:
            return self.build_ensemble_model()

    def build_rnn_model(self):
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = Bidirectional(GRU(self.word_padding_size, dropout=0.3,
                                     recurrent_dropout=0.1, return_sequences=True))(word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)
        rnn_word = Flatten()(rnn_word)
        out = Dense(64, activation='relu')(rnn_word)
        out = Dropout(0.5)(out)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_rcnn_model(self):
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        conv1 = Conv1D(filters=self.word_padding_size, kernel_size=5, padding='valid',
                       activation='relu', kernel_initializer='uniform')(word_embedding)
        pooling1 = MaxPooling1D(50)(conv1)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(
            pooling1)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)
        rnn_word = Flatten()(rnn_word)
        out = Dense(64, activation='relu')(rnn_word)
        out = Dropout(0.5)(out)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_cnn_model(self):
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        conv1 = Conv1D(filters=self.word_padding_size, kernel_size=5, padding='valid',
                       activation='relu', kernel_initializer='uniform')(word_embedding)
        maxpool = GlobalMaxPooling1D()(conv1)
        out = Dense(64, activation='relu')(maxpool)
        out = Dropout(0.5)(out)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_trend_model(self):
        return self.build_attention_model()

    def build_attention_model(self):  
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)
        # hot_index_input = BatchNormalization()(hot_index_input)
        att = Attention()(rnn_word)
        out = Dense(64, activation='relu')(att)
        # out = Dropout(0.4)(out)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
        model.summary()
        return model

    def build_attention_trend(self):
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(
            word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)

        hot_index_input = Input(shape=(self.hot_index_data.shape[1], self.hot_index_data.shape[2]), dtype='float32', name="hot_index_input")
        hot_index_td = TimeDistributed(Dense(50))(hot_index_input)
        # hot_index_bn = BatchNormalization()(hot_index_input)
        merge_layer = Concatenate(axis=2)([rnn_word, hot_index_td])
        att = AttentionHotTerm(50, name='attention_layer')(merge_layer)
        out = Dense(64, activation='relu')(att)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input, hot_index_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
        model.summary()
        return model

    def build_ensemble_model(self):
        # word input
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(
            word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)

        hot_index_input = Input(shape=(self.hot_index_data.shape[1], self.hot_index_data.shape[2]), dtype='float32',
                                name="hot_index_input")
        hot_index_td = TimeDistributed(Dense(50))(hot_index_input)
        # hot_index_bn = BatchNormalization()(hot_index_input)
        merge_layer = Concatenate(axis=2)([rnn_word, hot_index_td])
        att = AttentionHotTerm(50)(merge_layer)

        # account_data
        account_input = Input(shape=(self.account_padding_size,), dtype='int32', name="account_input")
        account_embedding = Embedding(self.account_size, self.account_embedding_dim,
                                      input_length=self.account_padding_size, trainable=True)(account_input)
        account_out = Flatten()(account_embedding)
        account_out = Dropout(0.3)(account_out)
        # tag_input
        tag_input = Input(shape=(self.tag_padding_size,), dtype='int32', name="tag_input")
        tag_embedding = Embedding(self.tag_size, self.tag_embedding_dim,
                                  input_length=self.tag_padding_size, trainable=True)(tag_input)
        tag_out = Flatten()(tag_embedding)
        tag_out = Dropout(0.3)(tag_out)
        # post_time
        post_time_input = Input(shape=(self.post_time_data.shape[1],), dtype='float32', name="post_time_input")
        meta_feature__merge = Concatenate(axis=1)([account_out, tag_out, post_time_input])

        fm_out = FMLayer(100, activation='relu')(meta_feature__merge)
        fm_out = Dropout(0.5)(fm_out)
        # fm_out = FMLayer(100, activation='relu')(fm_out)
        # fm_out = Dropout(0.5)(fm_out)

        out = Concatenate(axis=1)([att, fm_out])
        # out = Dense(64)(out)
        # out = Dropout(0.3)(out)

        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)
        model = Model(inputs=[word_input, hot_index_input, account_input, tag_input, post_time_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_deep_fm_model(self):
        # account_data
        account_input = Input(shape=(self.account_padding_size,), dtype='int32', name="account_input")
        account_embedding = Embedding(self.account_size, self.account_embedding_dim,
                                      input_length=self.account_padding_size, trainable=True)(account_input)
        account_out = Flatten()(account_embedding)
        account_out = Dropout(0.3)(account_out)
        # tag_input
        tag_input = Input(shape=(self.tag_padding_size,), dtype='int32', name="tag_input")
        tag_embedding = Embedding(self.tag_size, self.tag_embedding_dim,
                                  input_length=self.tag_padding_size, trainable=True)(tag_input)
        tag_out = Flatten()(tag_embedding)
        tag_out = Dropout(0.3)(tag_out)
        # post_time
        post_time_input = Input(shape=(self.post_time_data.shape[1],), dtype='float32', name="post_time_input")
        meta_feature__merge = Concatenate(axis=1)([account_out, tag_out, post_time_input])

        fm_out = FMLayer(100, activation='relu')(meta_feature__merge)
        fm_out = Dropout(0.5)(fm_out)
        fm_out = FMLayer(100, activation='relu')(fm_out)
        fm_out = Dropout(0.5)(fm_out)

        dense_out = Dense(64, activation='relu')(meta_feature__merge)
        dense_out = Dropout(0.5)(dense_out)
        dense_out = Dense(64, activation='relu')(dense_out)
        dense_out = Dropout(0.5)(dense_out)
        dense_out = Dense(64, activation='relu')(dense_out)
        dense_out = Dropout(0.5)(dense_out)

        out = Concatenate(axis=1)([fm_out, dense_out])
        l_out = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[account_input,tag_input, post_time_input], outputs=l_out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def save_test_result(self, predict_labels):
        print("******save test result********")

        with open("models/test_result", "w+", encoding='utf8') as f:
            for i in range(TEST_SIZE):
                tmp = [np.array2string(predict_labels[i])] + self.get_valid_data(i)
                result = ' '.join(tmp)
                f.write(result + "\n")



    def save_test_result2(self, y_test, y_pred):
        print('***********save test result custom*****')
        with open("models/test_result2", "w+", encoding='utf8') as f:
            for i in range(TEST_SIZE):
                f.write(' '.join([str(y_test[i]), str(y_pred[i])]) + '\n')

    def get_valid_data_2(self, i):
        item_values = [np.array2string(self.valid_labels[i])]
        keys = ['_text']
        if isinstance(self.valid_data, dict):
            for key in keys:
                item_values.append(key)
                if isinstance(self.valid_data[key][i], np.ndarray):
                    item_values.append(np.array2string(self.valid_data[key][i]))
                else:
                    item_values.append(str(self.valid_data[key][i]))
        else:
            item_values.append(str(self.valid_data[i]))
        return item_values


    def load_old_model(self):
        # with tf.Session() as sess:
        #     # model = keras.models.load_model('current_model.h5')
        #     sess.run(tf.global_variables_initializer())
        #     try:
        #         model_1.load_weights("model_1_weights.hdf5")
        #     except IOError as ioe:
        #         print("no checkpoints available !")
        #     model_1.fit(x_train_34_T, x_train_34_C,
        #                 validation_data=(x_val_34_T, x_val_34_C),
        #                 epochs=2, batch_size=16, shuffle=True,
        #                 callbacks=[tb_callback, checkpointer])
        #     # model.save('current_sent_model.h5')
        pass

    def log_info(self, message):
        with open("models/debug_log_plot", "a+") as f:
            f.write(message + "\n")

    def train(self):
        print("model train start...")
        start = time.time()
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        tb_callback = TensorBoard(log_dir='logs/keras_log', histogram_freq=0, write_graph=True,
                                                  write_images=True)
        # check_pointer = ModelCheckpoint(filepath="./models/model_1_weights.hdf5",
        #                                verbose=1,
        #                                monitor="val_acc",
        #                                save_best_only=True,
        #                                mode="max")
        history = self.model.fit(self.train_data, self.train_labels, validation_split=0.2, batch_size=1000, shuffle=True,
                                 epochs=self.epochs, verbose=1, class_weight=None,
                                 callbacks=[earlyStopping, tb_callback])

        # history = self.model.fit_generator(generator=self.batch_generator(self.train_data, self.train_labels, batch_size=500),
        #                                    steps_per_epoch=np.ceil(self.train_labels.shape[1] / 500),
        #                                    validation_data=(self.valid_data, self.valid_labels),
        #                                    epochs=1, verbose=1)
        predict_labels = self.model.predict(self.valid_data)
        self.save_test_result(predict_labels)
        test_score, test_acc = self.model.evaluate(self.valid_data, self.valid_labels)
        print('test score: %s, acc: %s' % (test_score, test_acc))
        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(np.argmax(self.valid_labels, axis=1), np.argmax(predict_labels, axis=1)))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(np.argmax(self.valid_labels, axis=1), np.argmax(predict_labels, axis=1))
        print(cm)
        # if self.is_regress:
        #     y_test, y_pred = self.inverse_label_process(self.valid_labels, predict_labels)
        #     self.save_test_result2(y_test, y_pred)
        cost = time.time() - start
        print("model train done, cost: %s" % cost)
        # print(history.history.keys())
        plot_dict = OrderedDict()
        plot_dict['train_model'] = self.model_name
        plot_dict['dataset'] = self.dataset
        plot_dict['test_acc'] = test_acc
        plot_dict['test_score'] = test_score
        plot_dict['multi_class'] = 4 if self.is_multi_class else 2
        plot_dict['history'] = history.history
        self.log_info(json.dumps(plot_dict))
        # self.plot_result(history=history)

# model_name = 'rnn'
# model_name = 'cnn'
model_name = 'rcnn'
# model_name = 'deep_fm'
# model_name = 'cnn_voca'
# model_name = 'cnn_char'
# model_name = 'plain_cnn'
# model_name = 'attention'
# model_name = 'attention_trend'
# model_name = 'ensemble'
model = TextClassifyDNN(model_name, dataset='toutiao', is_multi_class=False, is_omit_em=False, is_omit_tr=True)
model.train()
#