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
