# -*- coding:utf-8 -*-
import numpy as np
from konlpy.tag import Twitter
from sklearn.externals import joblib
import pandas as pd
import json


class wrapper:
    @staticmethod
    def pos_tagging_with_stem(text):
        text_pos = twitter.pos(text, norm=False, stem=True)
        return ' '.join([token for token, pos in text_pos
                         if pos not in ['Punctuation']])


twitter = Twitter()


class Classifier:
    def __init__(self, pipe_file_name='pipe-svm.sav', level_file_name='levels.sav', wrapper_file_name='wrapper.sav'):
        self.pipe = joblib.load(pipe_file_name)
        self.levels = joblib.load(level_file_name)
        self.wrapper = joblib.load(wrapper_file_name)

    def inference(self, text):
        values = self.pipe.predict_proba([self.wrapper.pos_tagging_with_stem(text)])[0]
        top_score_map = sorted([(self.levels[i], values[i]) for i in range(len(values)) if values[i] > 0.12],
                               key=lambda x: -x[1])
        json_dict = {'deep_outputs': top_score_map, 'raw_sentence': text}
        return json.dumps(json_dict)


clf = Classifier()
Rbf_svm = clf

data_df = pd.read_csv('total.csv', encoding='utf-8')
labels, levels = pd.factorize(data_df['classification'])

"""
print(clf.pipe.score(data_df['tokens'], labels))
print(clf.inference('배가 아파요'))
print(Rbf_svm.inference('사실 요즘 배가 아파서 왔어요.'))
print(Rbf_svm.inference('설사가 너무 심한거 같네요.'))
print(Rbf_svm.inference('속이 계속 더부룩한 느낌이에요'))
print(Rbf_svm.inference('명치 쪽이 계속 땡기는거 같아요'))
print(Rbf_svm.inference('어제 속이 메스꺼워서 구토를 오랬동안 했던거 같아요'))
"""

print(clf.inference('사실 요즘 배가 아프고 구토가 심해서 왔어요.. 제가 너무 잘생겨서 그런것이겠죠 원지운님이 저를 보고 웃고있어요 김동민 화이팅'))
print(clf.inference('어젯밤에 설사를 하다가 명치가 아팠어요.'))
print(clf.inference('설사가 심한데 약을 먹어야 할까요?'))
print(clf.inference('배에 가스가 찬 기분이에요.'))
print(clf.inference('어머니가 위암으로 돌아가셨어요.'))
