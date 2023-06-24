from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df = pd.read_csv("data/titles.csv", encoding='utf-8', usecols=['description'])
df = df.dropna()
descriptions = df['description'].values


embeddings = []

for d in descriptions:
    embeddings.append(embedder.encode(d, show_progress_bar=True))
embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)