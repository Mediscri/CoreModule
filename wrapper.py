# -*- coding:utf-8 -*-
import numpy as np
from konlpy.tag import Twitter
from sklearn.externals import joblib
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

twitter = Twitter()


def pos_tagging_with_stem(text):
    text_pos = twitter.pos(text, norm=False, stem=True)
    return ' '.join([token for token, pos in text_pos
                     if pos not in ['Punctuation']])


class Classifier:
    def __init__(self, pipe_file_name='pipe.sav', level_file_name='levels.sav'):
        self.pipe = joblib.load(pipe_file_name)
        self.levels = joblib.load(level_file_name)

    def inference(self, text):
        print([pos_tagging_with_stem(text)])
        return self.levels[self.pipe.predict([pos_tagging_with_stem(text)])][0], \
               np.max(self.pipe.predict_proba([pos_tagging_with_stem(text)]))


clf = Classifier()
print(clf.inference('배가 아파요'))
