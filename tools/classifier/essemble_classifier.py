"""Essemble Classifier"""

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def create_essemble_classifier():
    svm_model = svm.SVC(probability=True)
    rf_classifier = RandomForestClassifier(max_depth=None, n_estimators=11)
    lr_classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)

    clf = VotingClassifier(estimators=[
        ('lr', lr_classifier), ('rf', rf_classifier), ('gnb', svm_model)],
        voting='hard', weights=[1, 1, 1],
        flatten_transform=True)

    return clf


def ensemble_classifier_fit(
        classifier,
        train_embeddings,
        train_labels,
        pca=True,
        n_components=None,
):
    pca_model = lambda x: x

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
        pca_model=lambda x: x,
):
    test_embeddings = pca_model(test_embeddings)
    predictions = classifier.predict(test_embeddings)
    return predictions





