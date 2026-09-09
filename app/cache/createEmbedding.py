from sentence_transformers import SentenceTransformer
import asyncio
from typing import Tuple, List

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def get_embeddings(self, text: str) -> Tuple[bytes, List[float]]:
        """
        Runs the neural model ONCE and returns:
        1. bytes for Redis
        2. list of floats for pgvector
        """
        arr = await asyncio.to_thread(self.model.encode, text, normalize_embeddings=True)
        return arr.astype("float32").tobytes(), arr.tolist()