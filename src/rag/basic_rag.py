from dotenv import load_dotenv
import os, csv
import pathlib
import requests
from utils import DATA, ROOT
import pandas as pd


from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

import numpy as np
from tqdm.auto import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()  # Load environment variables from .env file

PROVIDER = 'gemini'
EMBED_PROVIDER = 'gemini'

CHAT_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"


# False -> call the embedding API for every chunk (small cost)
# True  -> load vectors from the cache file written by a previous run
load_embedding = True
DATA_URL = "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv"
DATA_PATH = pathlib.Path(DATA, "mini-llama-articles.csv")
print(f"DATA_PATH: {DATA_PATH}")

if not DATA_PATH.exists():
    DATA_PATH.write_bytes(requests.get(DATA_URL, timeout=30).content)
    print(f"Downloaded {DATA_PATH}")


# basic chunking with no overlapping or anything fancy
def split_into_chunks(text, chunk_size=1024):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


# read the csv file for content and perform the chunking

chunks = []

with open(DATA_PATH, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx==0:
            continue  # Skip the header row
        chunks.extend(split_into_chunks(row[1]))  # As the content is in the second column

print(f"Number of chunks: {len(chunks)}")
print(f"Number of articles: {idx}")

# transform the chunks into dataframe
df = pd.DataFrame(chunks, columns=['chunk'])
# print(f"Dataframe head: {df.head()}")

# create the embedding for each chunk and store it in a new column
MODELS = {
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-5",
}
MODELS[PROVIDER] = CHAT_MODEL

if "gemini" in (PROVIDER, EMBED_PROVIDER):
    gemini_client = genai.Client()
if "openai" in (PROVIDER, EMBED_PROVIDER):
    openai_client = OpenAI()
if "anthropic" in (PROVIDER, EMBED_PROVIDER):
    anthropic_client = Anthropic()

def generate(prompt, system=None, model=None):

    if PROVIDER == "gemini":
        response = gemini_client.models.generate_content(
            model=model or MODELS["gemini"],
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system
            )
        )
        return response.text

    if PROVIDER == "openai":
        response = openai_client.responses.create(
            model=model or MODELS["openai"],
            input=prompt,
            instructions=system,
            reasoning={"effort": "none"}
        )
        return response.output_text

    if PROVIDER == "anthropic":
        response = anthropic_client.messages.create(
            model=model or MODELS["anthropic"],
            max_tokens=4096,
            **({"system": system} if system else {}),
            messages=[{"role":"user", "content": prompt}]
        )
        return response.content[0].text

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")


EMBED_MODELS = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}
EMBED_MODELS[EMBED_PROVIDER] = EMBED_MODEL  # setup-cell selection (or typed ID) wins
EMBED_DIM = 1536  # same output size for both providers


def embed(texts, task="document"):
    """Embed a list of texts with the selected EMBED_PROVIDER.

    Returns a list of EMBED_DIM-dimensional vectors (one per input text).
    task: "document" for corpus chunks, "query" for user questions. Retrieval
    models treat the two sides differently; see above for how each model family
    wants that declared.
    """
    if isinstance(texts, str):
        texts =  [texts]
    texts = [t.replace("\n", " ") for t in texts] # embeddings work best on single lines

    if EMBED_PROVIDER == "gemini":

        model = EMBED_MODELS["gemini"]

        if model.startswith("gemini-embedding-2"):
            # Embedding 2 has no task_type: the task goes into the text as a prefix.
            # Documents carry a title slot ("none" when we have no title to give).
            prefix = "task: search result | query: " if task == "query" else "title: none | text: "
            payload = [prefix + t for t in texts]
            config = genai_types.EmbedContentConfig(output_dimensionality=EMBED_DIM)

            # Embedding 2 is multimodal: a plain list in `contents` is read as ONE
            # document of several parts and returns ONE aggregated vector. Each input
            # must be its own Content object to get one vector per input.
            result = gemini_client.models.embed_content(
                model = model,
                contents = [
                    genai_types.Content(parts=[genai_types.Part.from_text(text=t)])
                    for t in payload
                ],
                config=config,
            )
            vectors = [e.values for e in result.embeddings]

            # Trust nothing: if the batch still collapsed, embed one at a time.
            if len(vectors) != len(payload):
                vectors = [
                    gemini_client.models.embed_content(
                        model = model,
                        contents = t,
                        config = config
                    ).embeddings[0].values
                    for t in payload
                ]
            return vectors

        # Embedding 001 and earlier: an explicit task_type, and a plain list of
        # strings already returns one vector per string.
        result = gemini_client.models.embed_content(
            model = model,
            contents = texts,
            config = genai_types.EmbedContentConfig(
                task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY",
                output_dimensionality = EMBED_DIM
            )
        )
        vectors = [e.values for e in result.embeddings]

        # if len(vectors)!=len(texts):
        #     vectors = [
        #         gemini_client.models.embed_content(
        #             model = model,
        #             contents = t,
        #             config = genai_types.EmbedContentConfig(
        #                 task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY",
        #                 output_dimensionality = EMBED_DIM
        #             )
        #         ).embeddings.values
        #         for t in texts
        #     ]
        return vectors

    if EMBED_PROVIDER == 'openai':
        # dimensions= is redundant for text-embedding-3-small (1536 natively) but
        # keeps the 1536 contract if you type a different OpenAI model into the picker.
        result = openai_client.embeddings.create(
            model = EMBED_MODELS["openai"],
            input = texts,
            dimensions = EMBED_DIM
        )
        return [d.embedding for d in result.data]

    raise ValueError(f"Unknown EMBED_PROVIDER: {EMBED_PROVIDER!r}")


# test the embed
# vectors = embed(["LLMs predict tokens.", "What do LLMs predict?"])

# assert len(vectors) == 2, f"Expected 2 vectors, got {len(vectors)} — the batch collapsed." 
# assert all(len(v) == EMBED_DIM for v in vectors), "Unexpected vector dimension."

# print(f"Got {len(vectors)} vectors of dimension {len(vectors[0])}")


# The cache key includes the MODEL and dimension, not just the provider: two
# embedding models produce incomparable vectors, so they must never share a file.
MODEL_TAG = EMBED_MODELS[EMBED_PROVIDER].replace("/", "-")
CACHE_DIR = pathlib.Path(ROOT, "Cache")
if not CACHE_DIR.exists():
    # create the directory
    os.makedirs(CACHE_DIR)

CACHE_PATH = pathlib.Path(CACHE_DIR, f"mini-llama-articles-embedded-{MODEL_TAG}-{EMBED_DIM}d.csv")
BATCH_SIZE = 30


if not load_embedding:
    print(f"Generating embeddings with {EMBED_MODELS[EMBED_PROVIDER]}...")
    embeddings = []
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df['chunk'].iloc[start: start+BATCH_SIZE].tolist()
        embeddings.extend(embed(batch, task='document'))

    assert len(embeddings) == len(df), (
        f"Got {len(embeddings)} vectors for {len(df)} chunks — the embedding batch "
        "collapsed; check the embed() batching branch for your model."
    )

    df["embedding"] = embeddings
    df.to_csv(CACHE_PATH, index=False)
    print(f"Cached embeddings to {CACHE_PATH}")

else:
    assert CACHE_PATH.exists(), (
        f"No cache for {EMBED_MODELS[EMBED_PROVIDER]} at {CACHE_PATH}.\n"
        "Set load_embedding = False to build it (or switch EMBED_MODEL back to the "
        "model you embedded with)."
    )
    print(f"Loading cached embeddings from {CACHE_PATH}")
    df = pd.read_csv(CACHE_PATH)
    # Stored vectors are JSON lists — parse with json.loads (never eval() on data!)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))


# Guardrail: whatever we are about to search must be the same shape as the vector
# we will embed the question into. A silent mismatch here is the classic "no answer" bug.
cached_dim = len(df['embedding'].iloc[0])
assert cached_dim == EMBED_DIM, (
    f"Cached vectors are {cached_dim}-dim but EMBED_DIM is {EMBED_DIM} — "
    "re-embed with load_embedding = False."
)
print(f"✅ knowledge base: {len(df)} chunks × {cached_dim} dims, via {EMBED_MODELS[EMBED_PROVIDER]}")


# Finding related chunks given the QUERY
# cosine-similarity metric helps to find chunks that are close to the given QUERY
# by calculating the cosine distance between the QUERY vector and chunk vector

QUESTION = "How many parameters LLaMA 2 model has?"

QUESTION_emb = np.array(embed(QUESTION, task="query")[0])
print(f"Question vector: {QUESTION_emb.shape}")

# BAD_SOURCE_emb = embed("The sky is blue.")[0]
# GOOD_SOURCE_emb = embed("LLaMA 2 model has a total of 2B parameters.")[0]

# print("> Bad  Source Score:", cosine_similarity([QUESTION_emb], [BAD_SOURCE_emb])[0][0].round(4))
# print("> Good Source Score:", cosine_similarity([QUESTION_emb], [GOOD_SOURCE_emb])[0][0].round(4))

NUM_CHUNKS_TO_RETRIEVE = 3

# Similarity between the question and every chunk in the knowledge base
cosine_similarities = cosine_similarity([QUESTION_emb], np.stack(df["embedding"].to_numpy()))[0]

# Indices of the top-k highest-scoring chunks
top_indices = np.argsort(cosine_similarities)[::-1][:NUM_CHUNKS_TO_RETRIEVE]

# Always LOOK at what you retrieved — chunks and scores — before trusting it
for rank, i in enumerate(top_indices, start=1):
    print(f"┌─ Rank {rank} | chunk #{i} | score {cosine_similarities[i]:.4f}")
    preview = df.chunk[i][:300].replace("\n", " ")
    print(f"│  {preview}...")
    print("└" + "─" * 60)


# System Instructions for the AI assistant
SYSTEM_INSTRUCTION = (
    "You are an assistant and expert in answering questions from a chunks of content. "
    "Only answer AI-related question, else say that you cannot answer this question."
)

PROMPT_TEMPLATE = (
    "Read the following informations that might contain the context you require to "
    "answer the question. You can use the informations starting from the "
    "<START_OF_CONTEXT> tag and end with the <END_OF_CONTEXT> tag. Here is the content:\n\n"
    "<START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n\n"
    "Please provide an informative and accurate answer to the following question based "
    "on the available context. Be concise and take your time.\nQuestion: {question}\nAnswer:"
)

context = "".join(df.chunk[i] for i in top_indices)
formatted_prompt = PROMPT_TEMPLATE.format(context=context, question=QUESTION)

answer = generate(formatted_prompt, system=SYSTEM_INSTRUCTION)
print(f"Generated Response to the question: {answer}")

