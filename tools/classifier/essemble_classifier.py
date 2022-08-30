"""Essemble Classifier"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def create_essemble_classifier(
        svm_config=None,
        rf_config=None,
        lr_config=None,
):
    if svm_config is None:
        svm_config = dict(probability=True)
    if rf_config is None:
        rf_config = dict(max_depth=None, n_estimators=11)
    if lr_config is None:
        lr_config = dict(C=0.3, max_iter=1000)

    svm_model = svm.SVC(**svm_config)
    rf_classifier = RandomForestClassifier(**rf_config)
    lr_classifier = LogisticRegression(**lr_config)

    clf = VotingClassifier(estimators=[
        ('lr', lr_classifier), ('rf', rf_classifier), ('gnb', svm_model)],
        voting='soft', weights=[1, 1, 1],
        flatten_transform=True)

    return clf


def ensemble_classifier_fit(
        classifier,
        train_embeddings,
        train_labels,
        pca=True,
        n_components=None,
):
    pca_model = None

    if pca:
        if n_components is None:
            n_components = min(train_embeddings.shape)
        pca_model = PCA(n_components=n_components)
        pca_model.fit(train_embeddings)
        train_embeddings = pca_model.transform(train_embeddings)

    classifier.fit(train_embeddings, train_labels)

    return classifier, pca_model


def ensemble_classifier_predict(
        classifier,
        test_embeddings,
        pca_model=None,
):
    if pca_model:
        test_embeddings = pca_model.transform(test_embeddings)
    predictions = classifier.predict(test_embeddings)
    return predictions
