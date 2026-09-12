import csv
import operator
import random
import math


class KNN:
    """
    Implémentation KNN from scratch.
    Basée sur l'implémentation originale de l'auteur, généralisée pour
    fonctionner avec n'importe quel jeu de points (pas seulement des fleurs)
    et étendue avec une deuxième métrique de distance.
    """

    def __init__(self, x_train, y_train, k=3, distance_fn="euclidian"):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.distance_fn = distance_fn

    def euclidian_distance(self, training_point, point):
        diff_sum = 0
        for x1, x2 in zip(training_point, point):
            diff = x1 - x2
            diff_sum += math.pow(diff, 2)
        return math.sqrt(diff_sum)

    def manhattan_distance(self, training_point, point):
        return sum(abs(x1 - x2) for x1, x2 in zip(training_point, point))

    def distance(self, training_point, point):
        if self.distance_fn == "euclidian":
            return self.euclidian_distance(training_point, point)
        elif self.distance_fn == "manhattan":
            return self.manhattan_distance(training_point, point)
        else:
            return None

    def find_neighbors(self, point):
        # neighbors: [(distance, label), ...]
        neighbors = []
        for training_point, label in zip(self.x_train, self.y_train):
            d = self.distance(training_point, point)
            neighbors.append((d, label))
        neighbors.sort(key=lambda n: n[0])
        return neighbors

    def test(self, x_test):
        predictions = []
        for point in x_test:
            neighbors = self.find_neighbors(point)[: self.k]
            response = self.vote(neighbors)
            predictions.append(response)
        return predictions

    def accuracy(self, y_pred, y_test):
        correct = 0
        for y1, y2 in zip(y_pred, y_test):
            correct += 1 if y1 == y2 else 0
        return (correct / float(len(y_test))) * 100

    def vote(self, neighbors):
        votes = {}
        for neighbor in neighbors:
            prediction = neighbor[1]
            votes[prediction] = votes.get(prediction, 0) + 1

        votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return votes[0][0]

    @staticmethod
    def load_data(filename, split, pop_header=True):
        x_train, y_train, x_test, y_test = [], [], [], []
        with open(filename) as file:
            datas = list(csv.reader(file))
            if pop_header:
                datas.pop(0)

            for data in datas:
                for i, v in enumerate(data[:-1]):
                    data[i] = float(v)

                if random.random() < split:
                    x_train.append(data[:-1])
                    y_train.append(data[-1])
                else:
                    x_test.append(data[:-1])
                    y_test.append(data[-1])

            return x_train, y_train, x_test, y_test
