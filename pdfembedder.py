
# ==============================================================================

# This script performs data manipulation, embedding generation, and searches for

# pdf files. It optimizes for

# performance, clarity, and integrates tiktoken functionality for better handling

# of tokens in data preprocessing steps.

# ==============================================================================


# Required libraries

import pandas as pd

import argparse

from tiktoken import tiktoken

from openai.embeddings_utils import get_embedding, cosine_similarity


# Constants and configurations for the embedding model

EMBEDDING_MODEL = "text-embedding-ada-002"

EMBEDDING_ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")

MAX_TOKENS = 8000


# Function to load the dataset and preprocess it

def load_and_preprocess(input_path, top_n=1000):

    df = pd.read_csv(input_path, index_col=0)

    # Selects relevant columns and drops any rows with missing values

    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]].dropna()


    # Combine 'Summary' and 'Text' columns to create a single text input

    df["combined"] = df.apply(lambda row: f"Title: {row['Summary'].strip()}; Content: {row['Text'].strip()}", axis=1)


    # Sorting by 'Time' and taking the most recent entries up to 2x top_n to account for potential dropouts

    # during token count filtering.

    df = df.sort_values("Time").tail(top_n * 2)


    # Drop 'Time' column as it is no longer needed after sorting

    df.drop("Time", axis=1, inplace=True)


    # Calculate token counts using tiktoken and filter out reviews exceeding the MAX_TOKENS limit.

    # We utilize tiktoken's encoding object for counting tokens.

    df["n_tokens"] = df.combined.apply(lambda x: len(EMBEDDING_ENCODING.encode(x)))

    df = df[df.n_tokens <= MAX_TOKENS].tail(top_n)

    return df


# Function to add embedding information to the dataframe

def embed_reviews(df):

    # OPTIMIZATION: Batch embedding function can be integrated once available, reducing API calls.

    # This mock function assumes an existent batch embedding function `get_embedding_batch`

    df["embeddings"] = get_embedding_batch(df["combined"].tolist(), engine=EMBEDDING_MODEL)

    return df


# Function to search reviews for a specific product

def search_reviews(df, product_description, n=3):

    # Generates embedding for the product description.

    product_embedding = get_embedding(product_description, engine=EMBEDDING_MODEL)


    # Compute cosine similarities using vectorization for performance optimization.

    df["similarity"] = cosine_similarity_bulk(df["embedding"], product_embedding)


    # Sort by similarity and take the top `n` most related reviews.

    results = df.sort_values("similarity", ascending=False).head(n)

    return results


# Main function to orchestrate data flow and handle CLI interactions

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('file_path', help='Path to the data file')

    parser.add_argument('product_description', help='Product description to search')

    parser.add_argument('-n', type=int, default=3, help='Number of results to display')

    args = parser.parse_args()


    # Load, preprocess, and embed reviews

    df = load_and_preprocess(args.file_path)

    df = embed_reviews(df)


    # Perform the search and display results

    results = search_reviews(df, args.product_description, n=args.n)

    for i, row in results.iterrows():

        print(f"{row['combined'][:200]} (Similarity: {row['similarity']:.2f})")


    # Save results, including embeddings, if needed (code commented out by default)

    # df.to_csv("data/fine_food_reviews_with_embeddings.csv")


# Entry point for script execution

if __name__ == "__main__":

    main()

```
