import pandas as pd
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from lightfm import  LightFM
import re
import numpy as np
import pickle
from collections import OrderedDict

test = pd.read_csv('~/test.csv')
mapping = pd.read_csv('~/mapping.csv')
with open('~/model.pickle', 'rb') as f:
  model = pickle.load(f)
with open('~/dataset.pickle', 'rb') as f:
  dataset = pickle.load(f)

def get_recommendation(user_id, model, dataset, df, mapping, n_rec=5):
  user_item_map = dataset.mapping()
  out = OrderedDict()
  history = df[['reader_id', 'book_id', 'author', 'title']]
  history = history[history['reader_id'] == user_id].drop('reader_id', axis=1)
  history.reset_index(inplace=True, drop=True)

  user_id_model = user_item_map[0][user_id]
  preds = model.predict(user_id_model, np.arange(dataset.interactions_shape()[1]))
  books_ids_map = [user_item_map[2][i] for i in history['book_id'].tolist()]
  preds = np.delete(preds, books_ids_map, axis=0)

  item_ids = []
  while n_rec >= 0:
    ind = np.where(preds == np.max(preds))[0][0]
    item_ids.append(list(user_item_map[2].keys())[list(user_item_map[2].values()).index(ind)])
    preds = np.delete(preds, ind, axis=0)
    n_rec -= 1

  mapping = mapping[mapping['book_id'].isin(item_ids)]
  mapping.reset_index(inplace=True, drop=True)
  bks = []
  for i in range(len(mapping)):
    temp = OrderedDict()
    temp['id'] = mapping.loc[i, 'book_id']
    temp['title'] = mapping.loc[i, 'title']
    temp['author'] = mapping.loc[i, 'author']
    bks.append(temp)
  out['recommendations'] = bks

  hist = []
  for i in range(len(history)):
    temp = OrderedDict()
    temp['id'] = history.loc[i, 'book_id']
    temp['title'] = history.loc[i, 'title']
    temp['author'] = history.loc[i, 'author']
    hist.append(temp)
  out["history"] = hist
  
  return out

rec = get_recommendation(user_id, model, dataset, test, mapping)