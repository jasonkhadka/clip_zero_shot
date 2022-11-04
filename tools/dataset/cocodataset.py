"""Dataset class for CoCo"""

from pathlib import Path
import PIL
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


class CoCoDataset:
    def __init__(self, data_dir, fold=None, image_format='jpg', preprocess=None):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.image_format = image_format
        if fold is None:
            self._image_path = self.data_dir
        else:
            self._image_path = self.data_dir / fold
        self.files, self.classes = self._get_files()
        self.preprocess = preprocess
        self.class_index = {c: index for index, c in enumerate(self.classes)}

    def _get_files(self):
        image_paths = list(self._image_path.rglob(f'**/*.{self.image_format}'))
        classes = list(set([im.parent.name for im in image_paths]))

        return image_paths, classes

    def __getitem__(self, index):
        im_path = self.files[index]
        im = PIL.Image.open(im_path)
        if self.preprocess:
            im = self.preprocess(im)
        data_class = self.class_index[im_path.parent.name]
        return im, data_class

    def __len__(self):
        return len(self.files)

    def get_embeddings_and_labels(self, model, batch_size=100):
        all_features = []
        all_labels = []

        device = next(model.parameters()).device
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(self, batch_size=batch_size)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu(), torch.cat(all_labels).cpu()




