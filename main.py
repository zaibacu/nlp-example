import pandas as pd
import numpy as np

from argparse import ArgumentParser

from gensim.corpora import Dictionary
from gensim.utils import tokenize
from gensim.models import TfidfModel

from sklearn.naive_bayes import GaussianNB


def main(args):
    # Read CSV and select what we will actually need
    data = pd.read_csv(args.file)
    df = data[["articleID", "newDesk", "headline", "keywords", "snippet"]]

    # Text -> Tokens
    tokens = [list(tokenize(row)) for row in df["snippet"].values]
    # Building Corpus
    d = Dictionary(tokens)
    corpus = [d.doc2bow(row) for row in tokens]
    # tf-idf
    tf = TfidfModel(corpus)

    # Preview how single sentence can be converted into vector
    print(tf[corpus[0]])

    # Building Training Matrix
    M = len(tokens)
    N = len(d)

    X = np.zeros((M, N))
    for i, doc in enumerate(corpus):
        for idx, val in tf[doc]:
            X[(i, idx)] = val

    # Give labels
    id_by_label = dict([(v, i) for i, v in enumerate(set(df["newDesk"].values))])
    label_by_id = dict([(v, k) for k, v in id_by_label.items()])
    y = [id_by_label[label] for label in df["newDesk"].values]

    # Most of sklearn models should work - can experiment yourself
    model = GaussianNB()

    model.fit(X, y)

    # Testing model

    text = "The Trump is calling for war"  # Give any kind of text

    test_corpus = d.doc2bow(list(tokenize(text)))

    Xs = np.zeros((1, N))
    for idx, val in tf[test_corpus]:
        Xs[(0, idx)] = val

    model.predict(Xs)
    print(label_by_id[6])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", help="Give CSV file with data", default="data/ArticlesApril2017.csv")
    main(parser.parse_args())
