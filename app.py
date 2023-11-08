import openai

from embedchain.llm.cohere_llm import CohereLlm  # Consider using a more descriptive module name

import os

import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity

import pandas as pd  # Add missing import statement for pandas


os.environ["COHERE_API_KEY"] = "your-cohere-api-key"


# Function to load and inspect a dataset

def load_and_inspect_dataset(input_datapath):

    _, file_extension = os.path.splitext(input_datapath)

    

    if file_extension == '.csv':

        df = pd.read_csv(input_datapath, index_col=0)

    elif file_extension == '.json':

        df = pd.read_json(input_datapath)

    else:

        raise ValueError(f'Unsupported file type {file_extension}')


    print(f"Columns in the dataset: {df.columns}")


    return df



# Function to get the embedding of a text

def get_embedding(text, model_name='cohere'):

    if model_name == 'cohere':

        try:

            cohere_llm = CohereLlm()  # Consider moving this line outside the function for a potentially faster initialization

            return cohere_llm.get_llm_model_answer(text)

        except (ValueError, ModuleNotFoundError):

            print("Cohere model not available, falling back to ada model")

    

    # If the cohere model is not available or if the model_name is ada, use the ada model

    return openai.Embedding.create(input=text, model='text-embedding-ada-002')["data"][0]["embedding"]



# Function to calculate the cosine similarity between two vectors

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



# Function to search a dataset based on a query

def search_dataset(df, query, column_to_search):

    # Get the embedding of the query

    query_embedding = get_embedding(query)


    # Calculate the similarity between the query and each row in the specified column

    similarities = df[column_to_search].apply(lambda text: cosine_similarity(get_embedding(text), query_embedding))


    # Add the similarities as a new column in the dataframe

    df['similarity'] = similarities


    # Sort the dataframe by similarity in descending order

    df = df.sort_values(by='similarity', ascending=False)


    return df
