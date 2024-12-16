import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def merge_title_abstract(df):
    """
    Combines the `title` and `abstract` columns of a DataFrame into a new column named `text`.

    Parameters:
    - df (DataFrame): The DataFrame containing the `title` and `abstract` columns.

    Returns:
    - DataFrame: A copy of the input DataFrame with the new `text` column.
    """
    df_copy = df.copy()
    df_copy['text'] = df_copy.apply(lambda row: row['title'] if pd.isnull(row['abstract']) else row['title'] + " " + row['abstract'], axis=1)
    return df_copy

def preprocess_text_column(df, column_name='text'):
    """
    Preprocesses the text data in a specified column of a DataFrame:
    - Converts to lowercase
    - Removes special characters
    - Removes stop words
    - Lemmatizes the text

    Parameters:
    - df (DataFrame): The input DataFrame containing the text column.
    - column_name (str): The name of the column to preprocess. Default is 'text'.

    Returns:
    - DataFrame: A copy of the input DataFrame with the processed text column.
    """
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Load stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Helper functions
    def remove_special_characters(text):
        return re.sub(r'[^a-z\s\-]', '', text)  # keep "-"

    def remove_stopwords(text):
        return ' '.join([word for word in text.split() if word not in stop_words])

    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].apply(remove_special_characters)
    df_copy[column_name] = df_copy[column_name].apply(remove_stopwords)
    df_copy[column_name] = df_copy[column_name].apply(lemmatize_text)
    return df_copy

def generate_label_and_id_mappings(df, label_column='verified_uat_labels', id_column='verified_uat_ids'):
    """
    Generate a set of label-ID pairs and the old-new IDs pairs for the modified IDs

    Parameters:
    - df (DataFrame): The DataFrame containing labels and IDs.
    - label_column (str): The name of the column containing labels.
    - id_column (str): The name of the column containing IDs.

    Returns:
    - dict: A dictionary label-ID pairs consecutively from 0 to n-1
    - dict: A dictionary (old_id, new_id) for the mapping
    """
    # Collect unique label-ID pairs
    label_id = set()
    for labels, ids in zip(df[label_column], df[id_column]):
        label_id.update(zip(labels, ids))
    
    # Sort label-ID pairs by ascending ID
    label_id_sorted = sorted(label_id, key=lambda item: item[1])
    
    # Create mappings
    # Order the ascendind label-ID pairs consecutively from 0 to n-1
    label_new_id = {label: idx for idx, (label, _) in enumerate(label_id_sorted)}
    # Create a set that contains (old_id, new_id) for the mapping
    old_new_ids = {old_id: idx for idx, (_, old_id) in enumerate(label_id_sorted)}
    
    return label_new_id, old_new_ids

def apply_new_id_mapping(df, id_mapping, id_column='verified_uat_ids', new_column='new_ids'):
    """
    Applies ID mappings to the DataFrame to create a new column with remapped IDs.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - id_mapping (dict): Mapping of original IDs to new IDs.
    - id_column (str): The name of the column containing original IDs.
    - new_column (str): The name of the new column containing remapped IDs.

    Returns:
    - DataFrame: A copy of the DataFrame with a new column containing remapped IDs.
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Function to remap IDs
    def remap_ids(ids, remapping):
        return [remapping[old_id] for old_id in ids if old_id in remapping]
    
    # Apply the remapping
    df_copy[new_column] = df_copy[id_column].apply(lambda ids: remap_ids(ids, id_mapping))
    
    return df_copy
