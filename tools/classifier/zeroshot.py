"""ZeroShot classifier."""

import torch
from tools.utils import cos_similarity, update_means_and_class_count


class ZeroShot:
    def __init__(
            self,
            base_classifier,
            transformation,
            classes,
            classification_threshold=0.5,
            new_class_threshold=0.6,
    ):
        self.base_classifier = base_classifier
        self.transformation = transformation
        self.classes = classes
        self.classification_threshold = classification_threshold
        self.new_class_threshold = new_class_threshold

        self._new_class_means = None
        self._new_classes = []
        self._new_class_counts = []

    def __call__(self, image_embedding):

        # Compute prediction class and probability
        if self.transformation:
            reduced_embedding = self.transformation.transform(image_embedding)
        else:
            reduced_embedding = image_embedding.numpy()

        prediction_prob = self.base_classifier.predict_proba(reduced_embedding)
        prediction_prob, prediction_class = torch.tensor(prediction_prob).type(torch.float32).topk(1)

        if prediction_prob > self.classification_threshold:
            return self.classes[int(prediction_class)]

        return self._detect_out_of_distribution_class(image_embedding)

    def _generate_new_class(self):
        new_name = f'new_object_{len(self._new_classes) + 1}'
        self._new_classes.append(new_name)

    def _detect_out_of_distribution_class(self, image_embedding):
        if self._new_class_means is None:
            self._new_class_means = torch.tensor(image_embedding)
            self._generate_new_class()
            self._new_class_counts.append(1)
            return self._new_classes[0]

        # Similarity between the image_embedding and previous found embeddings
        similarity_value, index = cos_similarity(image_embedding, self._new_class_means).topk(1)
        similarity_value = similarity_value.item()
        index = index.item()

        if similarity_value < self.new_class_threshold:
            # Not similar to any previously detected new (out of distribution) object
            index = -1
            self._generate_new_class()

        # Update means and class count: mean of image embeddings are computed if same class is detected
        self._new_class_means, self._new_class_counts = update_means_and_class_count(
            image_embedding,
            index,
            self._new_class_means,
            self._new_class_counts,
        )
        return self._new_classes[index]
