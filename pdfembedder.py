import textract
import os
import openai
import tiktoken
from itertools import islice

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]

long_text = 'AGI ' * 5000
try:
    get_embedding(long_text)
except openai.InvalidRequestError as e:
    print(e)
import tiktoken

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]

truncated = truncate_text_tokens(long_text)
len(get_embedding(truncated))


# Extract the raw text from each PDF using textract

# changes must be made so that the pdf used as input is specified in the cli
text = textract.process('data/fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf', method='pdfminer').decode('utf-8')
clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')




# Split a text into smaller chunks of size n, preferably ending at the end of a sentence. The sizes must be determined by token size. Verify if the tokenizer function works with the following functions



def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch
def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator

def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def extract_chunk

# extract from pdf file to a format that can undergo embedding by ada from openai



import numpy as np

def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings
average_embedding_vector = len_safe_get_embedding(long_text, average=True)
chunks_embedding_vectors = len_safe_get_embedding(long_text, average=False)

print(f"Setting average=True gives us a single {len(average_embedding_vector)}-dimensional embedding vector for our long text.")
print(f"Setting average=False gives us {len(chunks_embedding_vectors)} embedding vectors, one for each of the chunks.")

# the output files should be json and csv and must consist of these parameters:
# When creating an embeddings file, the optimal categories for creating a JSON or CSV that will serve as a dataset would typically include:

# 1. `id`: A unique identifier for each data point. This could be a hash of the content or any unique value.

# 2. `content`: The actual text content that you want to create embeddings for.

# 3. `embedding`: The embedding vector for the content. This would be a list of numbers.

# . `metadata`: Any additional information about the content. This could include things like the source of the data, the data type, or any other relevant information.
