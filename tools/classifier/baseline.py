"""Baseline classifier using CLIP embeddings

Uses dot product between image embeddings and text embeddings to perform classification.
"""


import torch
import clip
from tools.dataset.cocodataset import CoCoDataset
from tools.utils import embed_class_labels


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def baseline_classifier_predict(
        dataset: CoCoDataset,
        classes: list,
        clip_model: str = 'ViT-B/32',
        device: str = DEVICE,
):
    model, _ = clip.load(clip_model, device)

    # Get embeddings
    image_embeddings, labels = dataset.get_embeddings_and_labels(model)
    text_embeddings = embed_class_labels(classes, model)

    # Normalising the embeddings
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Similarity computation
    similarity = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
    predictions = similarity.topk(1).indices.flatten()

    return predictions
