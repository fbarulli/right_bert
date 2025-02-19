
# src/embedding/embedding_validation.py
# src/embedding/embedding_validation.py
from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging
from pathlib import Path

from src.common.managers import get_tokenizer_manager, get_model_manager

logger = logging.getLogger(__name__)

def validate_embeddings(
    model_path: str,
    tokenizer_name: str,
    output_dir: str,
    words_to_check: Optional[List[str]] = None,
    top_k: int = 10
):
    """
    Validates the quality of trained embeddings using nearest neighbors and t-SNE.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_manager = get_model_manager()
    tokenizer_manager = get_tokenizer_manager()
    tokenizer = tokenizer_manager.get_worker_tokenizer(0, tokenizer_name)
    model = model_manager.get_worker_model(0, model_path, "embedding")
    model.eval()

    if words_to_check:
        logger.info(f"Finding nearest neighbors for: {words_to_check}")
        with torch.no_grad():
            inputs = tokenizer(words_to_check, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            word_embeddings = outputs['last_hidden_state'].mean(dim=1)

            all_embeddings = model.bert.embeddings.word_embeddings.weight.data

            similarities = cosine_similarity(word_embeddings.cpu().numpy(), all_embeddings.cpu().numpy())

            top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]

            for i, word in enumerate(words_to_check):
                print(f"Nearest neighbors for '{word}':")
                for j in range(top_k):
                    idx = top_k_indices[i, j]
                    neighbor_word = tokenizer.decode([idx])
                    similarity = similarities[i, idx]
                    print(f"  {j+1}: {neighbor_word} (similarity: {similarity:.4f})")

    logger.info("Performing t-SNE visualization...")
    with torch.no_grad():
        all_embeddings = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] - 1))
        reduced_embeddings = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 12))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, s=5)
    plt.title("t-SNE Visualization of Word Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(output_dir / "embedding_tsne.png")
    plt.close()
    logger.info(f"t-SNE plot saved to {output_dir / 'embedding_tsne.png'}")