import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class BertEmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        batch_size: int = 8
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔹 Loading model on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate_embeddings_in_chunks(
        self,
        texts,
        save_dir,
        chunk_size=5000,
        start_chunk=0
    ):
        os.makedirs(save_dir, exist_ok=True)

        total_chunks = (len(texts) // chunk_size) + 1
        print(f"📦 Total chunks: {total_chunks}")

        for chunk_idx in range(start_chunk, total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(texts))

            if start >= len(texts):
                break

            chunk_texts = texts[start:end]
            print(f"\n➡️ Processing chunk {chunk_idx + 1}/{total_chunks} ({start}:{end})")

            all_embeddings = []

            for i in tqdm(range(0, len(chunk_texts), self.batch_size), desc="Embedding batches"):
                batch_texts = chunk_texts[i:i + self.batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    model_output = self.model(**encoded)

                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                all_embeddings.append(embeddings.cpu().numpy())

                del encoded, model_output, embeddings
                torch.cuda.empty_cache()

            all_embeddings = np.vstack(all_embeddings)

            file_path = os.path.join(save_dir, f"embeddings_chunk_{chunk_idx}.npy")
            np.save(file_path, all_embeddings)

            print(f"✅ Saved: {file_path}")
