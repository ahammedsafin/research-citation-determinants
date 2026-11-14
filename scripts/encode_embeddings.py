import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"D:\cse360\cse437-citation-project")
df = pd.read_pickle(BASE / "data" / "cohort.pkl")
tr_m = (df["year"] <= 2012).values

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-MiniLM-L6-v2")
st.max_seq_length = 128

for split, mask in (("train", tr_m), ("test", ~tr_m)):
    out = BASE / "data" / f"emb_{split}.npy"
    if out.exists():
        print(split, "already encoded, skipping", flush=True)
        continue
    texts = df.loc[mask, "text"].tolist()
    parts = []
    t0 = time.time()
    done = 0
    CH = 5000
    for i in range(0, len(texts), CH):
        E = st.encode(texts[i:i + CH], batch_size=192, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        parts.append(E)
        done += len(E)
        el = time.time() - t0
        eta = el / done * (len(texts) - done)
        print(f"{split}: {done}/{len(texts)} | elapsed {el:.0f}s | ETA {eta:.0f}s", flush=True)
    np.save(out, np.vstack(parts))
    print("saved", out, flush=True)
print("ALL DONE", flush=True)
