# Clip Few shot and Zero shot learning
This project uses CLIP embeddings from text and images to detect classes in few shot and zero shot task.
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

* tools.classifier.baseline: Includes baseline classifier for few shot learnign
* tools.classifier.essemble_classifier: Includes essemble classifier that is trained to learn on CLIP embeddings
* tools.classifier.zeroshot: Includes Zero-Shot classifier
* utils: Some utility functions.

## Installations

* Install clip as given on the CLIP repo: https://github.com/openai/CLIP
* Install requirements from `requirement.txt`

## Report
Report is in `report.ipynb` notebook. It shows the training and uses of the detectors.

