"""Text encoders: sentence-transformers models, or openai:<model> (API)."""
import numpy as np

from trace_harness.io import l2_normalize


class Encoder:
    def __init__(self, name: str, device: str = "cuda", batch: int = 512):
        self.name, self.batch = name, batch
        if name.startswith("openai:"):
            from openai import OpenAI  # requires OPENAI_API_KEY

            self.client = OpenAI()
            self.model = name.split(":", 1)[1]
            self.st = None
        else:
            from sentence_transformers import SentenceTransformer

            self.st = SentenceTransformer(name, device=device)

    @property
    def tokenizer(self):
        return self.st.tokenizer if self.st is not None else None

    def encode(self, texts):
        if self.st is not None:
            e = self.st.encode(texts, batch_size=self.batch, convert_to_numpy=True,
                               normalize_embeddings=False, show_progress_bar=False)
        else:
            out = []
            for i in range(0, len(texts), 1024):
                r = self.client.embeddings.create(model=self.model, input=texts[i:i + 1024])
                out.extend(d.embedding for d in r.data)
            e = np.asarray(out, dtype=np.float32)
        return l2_normalize(np.asarray(e, dtype=np.float32))

    def dim(self):
        return int(self.encode(["dimension probe"]).shape[1])
