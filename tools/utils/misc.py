from typing import List
import torch
import clip


def embed_class_labels(
        classes: List[str],
        model,
):
    device = next(model.parameters()).device
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    return text_features


def compute_cluster_means(
        image_embeddings,
        image_labels,
):
    cluster_means = []
    cluster_count = []
    for idx in range(len(image_labels.unique())):
        target_embeddings = image_embeddings[image_labels == idx]
        cluster_means.append(target_embeddings.mean(dim=0, keepdim=True))
        cluster_count.append(target_embeddings.shape[0])

    cluster_means = torch.cat(cluster_means)
    cluster_count = image_labels.unique(return_counts=True)[1]
    return cluster_means, cluster_count
