import shutil
from pathlib import Path
from tqdm import tqdm
import click

import clip
from tools.classifier.baseline import baseline_classifier_predict
from tools.dataset.cocodataset import CoCoDataset


@click.command()
@click.option('-s', '--src-dir', help='source directory for images')
@click.option('-o', '--output-dir', help='output directory for organized images')
@click.option('-c', '--candidates', help='candidates for classes', multiple=True)
@click.option('-m', '--clip-model', help='name of clip model', default='ViT-B/32')
@click.option('-d', '--device', help='device to use', default='cpu')
@click.option('-f', '--image-format', help='format of the images', default='png')
def organize_directory(
        src_dir,
        output_dir,
        candidates,
        clip_model,
        device,
        image_format,
):
    src_dir = Path(src_dir)

    model, preprocess = clip.load(clip_model, device)
    dataset = CoCoDataset(src_dir, preprocess=preprocess, image_format=image_format)

    prediction_classes = baseline_classifier_predict(dataset, candidates)
    prediction_classes = list(map(lambda x: candidates[x], prediction_classes))

    if output_dir is None:
        output_dir = src_dir.parent/(src_dir.name + '_organized')
    output_dir.mkdir(exist_ok=True)

    for class_ in set(prediction_classes):
        (output_dir/class_).mkdir(exist_ok=True)

    for class_, file in tqdm(zip(prediction_classes, dataset.files)):
        shutil.copyfile(file, output_dir/class_/file.name)

    print('done.')
    return


if __name__ == '__main__':
    organize_directory()