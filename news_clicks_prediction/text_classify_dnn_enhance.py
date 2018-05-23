#!/usr/bin/env python
# -*- coding: utf-8 -*-  

# Copyright (c) 2017 - xiongjiezk <xiongjiezk@163.com>
from keras import Input, Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Flatten, Dropout, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Concatenate

import json
from collections import OrderedDict

import time

import jieba
import numpy as np
from sklearn import metrics

from news_clicks_prediction.FMLayer import FMLayer
from news_clicks_prediction.attention_base_sum import AttentionSum
from news_clicks_prediction.attention_base_weight import AttentionWeight
from news_clicks_prediction.attention_hot_term_sum import AttentionHotTermSum
from news_clicks_prediction.attention_hot_term_weight import AttentionHotTermWeight
from news_clicks_prediction.text_classify_dnn import TextClassifyDNN


class TextClassifyDNNEnhance(TextClassifyDNN):

    def save_test_result3(self, attention_weights):
        print("******save test result********")

        max_attention = attention_weights.max()
        html_list = []
        with open("models/test_result3", "a+", encoding='utf8') as f:
            for i in [-2, -1]:
                attention_pair = []
                for index, word in enumerate(jieba.cut(str(self.valid_data['_text'][i]))):
                    if index < attention_weights.shape[1]:
                        # weight = attention_weights[i, index] / max(attention_weights[i, :])
                        attention_pair.append((word, str(attention_weights[i, index])))
                        alpha_1 = attention_weights[i, index] / max_attention
                        html_ele = '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha_1, word)
                        html_list.append(html_ele)
                    else:
                        break
                txt = json.dumps(attention_pair)
                f.write(txt + "\n")
        file_name = "result/visualization_%s_%s_%s.html" % (self.model_name, self.dataset, (4 if self.is_multi_class else 2))
        with open(file_name, "a+") as html_file:
            for html_ele in html_list:
                html_file.write(html_ele)

    def build_attention_model(self):
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)
        # hot_index_input = BatchNormalization()(hot_index_input)
        # att = Attention2()(rnn_word)
        att_word_1 = AttentionWeight(name='attention_inter_one')(rnn_word)
        con_1 = Concatenate(axis=2)([rnn_word, att_word_1])
        att = AttentionSum(partition=50, name='attention_inter_two')(con_1)
        out = Dense(64, activation='relu')(att)
        # out = Dropout(0.4)(out)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def build_attention_trend(self):
        # ht_attention  0302 0.44, 0325 0.474, as 0.51, 0.55/0.56
        word_input = Input(shape=(self.word_padding_size,), dtype='int32', name="word_input")
        word_embedding = Embedding(self.vocabulary_size, self.word_embedding_dim, input_length=self.word_padding_size,
                                   weights=[self.embedding_matrix], trainable=True)(word_input)
        rnn_word = GRU(self.word_padding_size, dropout=0.3, recurrent_dropout=0.1, return_sequences=True)(
            word_embedding)
        rnn_word = TimeDistributed(Dense(50))(rnn_word)

        hot_index_input = Input(shape=(self.hot_index_data.shape[1], self.hot_index_data.shape[2]), dtype='float32', name="hot_index_input")
        hot_index_td = TimeDistributed(Dense(50))(hot_index_input)
        merge_layer = Concatenate(axis=2)([rnn_word, hot_index_td])
        # att = AttentionWithHotIndex(50, name='attention_layer')(merge_layer)
        att_weight_1 = AttentionHotTermWeight(partition=50, name='attention_inter_one')(merge_layer)
        con_1 = Concatenate(axis=2)([rnn_word, att_weight_1])
        att = AttentionHotTermSum(partition=50, name='attention_inter_two')(con_1)
        out = Dense(64, activation='relu')(att)
        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)

        model = Model(inputs=[word_input, hot_index_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        # att = AttentionWithHotIndex(50)(merge_layer)
        att_weight_1 = AttentionHotTermWeight(partition=50, name='attention_inter_one')(merge_layer)
        con_1 = Concatenate(axis=2)([rnn_word, att_weight_1])
        att = AttentionHotTermSum(partition=50, name='attention_inter_two')(con_1)

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

        out = Concatenate(axis=1)([att, fm_out])
        # out = Dense(64)(out)
        # out = Dropout(0.3)(out)

        output_layer = Dense(self.num_classes, activation=self.last_layer_activation, name='output')(out)
        model = Model(inputs=[word_input, hot_index_input, account_input, tag_input, post_time_input], outputs=[output_layer])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        print("model train start...")
        start = time.time()
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        tb_callback = TensorBoard(log_dir='E:/tmp/keras_log', histogram_freq=0, write_graph=True,
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

        # output attention weight
        layer_name = 'attention_inter_one'
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.valid_data)
        intermediate_output_s = np.squeeze(intermediate_output)
        self.save_test_result3(intermediate_output_s)
        print('intermediate_output save success')

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



# model_name = 'trend'
# model_name = 'attention'
# model_name = 'attention_trend'
model_name = 'ensemble'
model = TextClassifyDNNEnhance(model_name, dataset='sohu', is_multi_class=False, is_omit_em=False, is_omit_tr=False)
model.train()