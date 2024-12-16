import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import data_processing
from data_processing import merge_title_abstract,preprocess_text_column,generate_label_and_id_mappings, apply_new_id_mapping
from sklearn.metrics import f1_score,accuracy_score,recall_score,hamming_loss,precision_score

# import the train and the val dataset
train_base_dir = os.path.dirname('train.parquet')
val_base_dir = os.path.dirname('val.parquet')
train_file_path = os.path.join(train_base_dir, 'data', 'train.parquet')
val_file_path = os.path.join(train_base_dir, 'data', 'val.parquet')
df_train = pd.read_parquet(train_file_path)
df_val = pd.read_parquet(val_file_path)

# set up the train
df_train = merge_title_abstract(df_train)
df_train = preprocess_text_column(df_train)
label_new_id, old_new_ids = generate_label_and_id_mappings(df_train)
df_train = apply_new_id_mapping(df_train,old_new_ids)

# set up the val 
df_val = merge_title_abstract(df_val)
df_val = preprocess_text_column(df_val)
df_val = apply_new_id_mapping(df_val,old_new_ids)

# vectorize 
vectorizer3 = TfidfVectorizer(
    max_features=10000,
    max_df=0.8,
    min_df=0.01
)
TFIDF_train = vectorizer3.fit_transform(df_train['text']) # fit transform
TFIDF_val = vectorizer3.transform(df_val['text']) # transform to apply the vocabulary of the train on the val

# target
y_true = df_val['new_ids']
y_train = df_train['new_ids']

# convert target Y to binary matrix form with the mapping
all_classes = sorted({id_ for ids in df_train['new_ids'] for id_ in ids})
mlb = MultiLabelBinarizer(classes=all_classes)
y_train_matrix = mlb.fit_transform(y_train)
y_true_matrix = mlb.transform(y_true)

# import models
model3_tfidf = joblib.load('models/model3_tfidf.pkl')

# predict
prediction = model3_tfidf.predict(TFIDF_val)

# metrics
f1 = f1_score(y_true_matrix, prediction,average='samples')
print("Exact Match f1_score for model 3 :", f1)

precision = precision_score(y_true_matrix, prediction,average='samples',zero_division=0)
print("Exact Match precision_score for model 3 :", precision)

loss = hamming_loss(y_true_matrix, prediction)
print("Exact Match hamming_loss for model 3 :", loss)