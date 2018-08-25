# -*- coding:utf-8 -*-
from konlpy.tag import Twitter
from sklearn.externals import joblib
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
