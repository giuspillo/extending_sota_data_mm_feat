from sentence_transformers import SentenceTransformer
# from transformers import BertTokenizer, BertModel
# from transformers import RobertaTokenizer, RobertaModel
# from transformers import DistilBertTokenizer, DistilBertModel
# import torch
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm

texts = pd.read_csv('_text/text_ml1m.tsv')
texts['text'] = texts[['description', 'genres', 'tags', 'reviews']].fillna('').agg(' '.join, axis=1)
text_dict = dict(zip(texts['item'], texts['text']))

# sentence encoders
models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text_feat_dict = dict()
for item, text in tqdm(text_dict.items(), total=len(text_dict)):
    
    text = text.replace(',', ', ')

    emb = model.encode(text)
    text_feat_dict[item] = emb
    # print(f'text: {text}')

dict_name = 'all-MiniLM-L6-v2'
pkl.dump(dict_name, open(f'text/{dict_name}.pkl', 'wb'))

print(f'Text encoded with {dict_name}')





model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

text_feat_dict = dict()
for item, text in tqdm(text_dict.items(), total=len(text_dict)):

    text = text.replace(',', ', ')

    emb = model.encode(text)
    text_feat_dict[item] = emb
    # print(f'text: {text}')

dict_name = 'all-mpnet-base-v2'
pkl.dump(dict_name, open(f'text/{dict_name}.pkl', 'wb'))

print(f'Text encoded with {dict_name}')