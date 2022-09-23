from pprint import pprint

from flair.data import Sentence
from flair.models import TextClassifier

sia = TextClassifier.load('en-sentiment')


def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    label = sentence.labels[0]
    print(label.score)
    print(label.value)


def main():

    flair_prediction("@SawyerMerritt @Tesla For now, supply is too low, but ordering a Powerwall by itself should be possible end of year")

if __name__ == '__main__':
    main()
