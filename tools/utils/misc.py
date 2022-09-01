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


def cos_similarity(a, b, eps=1e-8):
    """Cosine similarity between tensors
    """
    a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim = a_norm @ b_norm.T
    return sim


def update_means_and_class_count(new_embedding, new_class, means, class_counts):
    if new_class < 0:
        # If new class is proposed, use the new embedding as class mean

        means = torch.cat((means, new_embedding))
        class_counts.append(1)
        return means, class_counts

    # Update new mean for the class
    sum_ = (means[new_class] * class_counts[new_class]) + new_embedding

    # Update class count for the detected class
    class_counts[new_class] += 1

    means[new_class] = sum_ / class_counts[new_class]

    return means, class_counts
