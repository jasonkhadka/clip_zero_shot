"""Baseline classifier using CLIP embeddings

Uses dot product between image embeddings and text embeddings to perform classification.
"""


import clip
from tools.dataset.cocodataset import CoCoDataset
from tools.utils import embed_class_labels, cos_similarity


def baseline_classifier_predict(
        dataset: CoCoDataset,
        classes: list,
        clip_model: str = 'ViT-B/32',
        device: str = 'cpu',
):
    model, _ = clip.load(clip_model, device)

    # Get embeddings
    image_embeddings, labels = dataset.get_embeddings_and_labels(model)
    text_embeddings = embed_class_labels(classes, model)

    # Similarity computation
    similarity = cos_similarity(image_embeddings, text_embeddings)
    similarity = (100.0 * similarity).softmax(dim=-1)
    predictions = similarity.topk(1).indices.flatten()

    return predictions
