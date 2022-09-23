import csv

from flair.data import Sentence
from flair.models import TextClassifier

sia = TextClassifier.load('en-sentiment')


def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    return sentence.labels[0]


def main():
    with open("elonmusk.csv", encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            text = row[3]
            if "tesla" in text.lower():
                label = flair_prediction(text)
                print("----------------")
                print(text)
                print(label.score)
                print(label.value)
                print("----------------")


if __name__ == '__main__':
    main()
