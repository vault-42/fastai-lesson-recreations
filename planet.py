from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings


def f2(preds, targs, start=.17, end=.24, step=.01):
    return max([fbeta_score(targs, (preds>threshold), 2, average='samples')
               for threshold in np.arange(start, end, step)])


def get_data(path, tfms, n, bs=64):
    val_idxs = get_cv_idxs(n)
    return ImageClassifierData.from_csv(path, 'train-jpg', f'{path}/train_v2.csv', bs, tfms=tfms, suffix='.jpg',
                                   val_idxs=val_idxs, test_name='test-jpg')
