# Clip zero shot classification
This project uses CLIP embeddings from text and images to detect classes in zero shot task.
```
├── data
├── requirements.txt
└── tools
    ├── classifier
    │   ├── baseline.py
    │   ├── essemble_classifier.py
    │   └── zeroshot.py
    ├── dataset
    │   └── cocodataset.py
    └── utils
        ├── __init__.py
        └── misc.py
```

* tools.classifier.baseline: Includes baseline classifier for few shot learning
* tools.classifier.essemble_classifier: Includes ensemble classifier that is trained to learn on CLIP embeddings
* tools.classifier.zeroshot: Includes Zero-Shot classifier
* utils: Some utility functions.

## Installations

* Install clip as given on the CLIP repo: https://github.com/openai/CLIP
* Install requirements from `requirement.txt`

## Usages

### 1. Script to organize a random collection of images
`clip_organize` script is able to organize your images by classifying given images using CLIP embeddings.

##### Use:
```
python clip_organize.py -s path/to/target_directory/ -o path/to/output_directory/ -c class1 -c class2 -c class3
```

##### Parameters:
```
-s/--src-dir: directory of images to organize.
-o/--output-dir: directory to store the images (by default same directory with '_organized' suffix).
-c/--candidates: use `-c` repeatedly to pass multiple classes. Candidate classes are classes that are used to classify the images.
-m/--clip-model: clip model to use (default is ViT-B/32).
-d/--device: device to use (default is cpu).
-f/--image-format: format of the image (default is png).
```

#### Example with data in the repo
First, unzip the file `boot_shoe_sandal.zip` in data directory, and then run the following script.
```
 python clip_organize.py -s 'data/boot_shoe_sandal/scrambled/' -c cow -c shoes -c sneaker -c sandals -c boots -c coat -c plane  -c tiger -c lion
```
Initial data directory tree is as given below. `boot_shoe_sandal` contains `scrambled` directory, which in turn contains 600 images of boots, shoes and sandals from coco-dataset.
```
└── data
     └── boot_shoe_sandal
         └── scrambled
```
The script classifies target images in the closest candidate class and reorganizes them in a new directory as shown below.
```
└── data
     └── boot_shoe_sandal
         ├── scrambled
         └── scrambled_organized
             ├── boots
             ├── sandals
             ├── shoes
             └── sneaker
```
Thus, if you have a directory with unclassified/unlabelled images, but you know what are the possible labels, you can use this script to organize your directory and label your data.
