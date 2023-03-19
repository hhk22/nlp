import numpy as np
import pandas as pd
from typing import Tuple, List


def random_generator(n_data: int, n_features: int = 4) -> Tuple[pd.DataFrame, np.ndarray]:
    rand_generator = np.random.default_rng()
    weights = rand_generator.random((1, n_features))[0]
    data = rand_generator.random((n_data, n_features))
    targets = np.random.choice([0,1], n_data)

    data = pd.DataFrame(data, columns=["n1", "n2", "n3", "n4"])
    data['target'] = targets

    return data, weights

class dnn:
    def __init__(self, learning_rate:float = 0.001, bias: float = 0.7, epochs:int = 50) -> None:
        self.learning_rate = learning_rate
        self.bias= bias
        self.epochs = epochs
        self.final_epoch_loss = []
    
    def train(self, data: pd.DataFrame, weights: np.ndarray):
        for epoch in range(self.epochs):
            individual_loss = []
            for i in range(len(data)):
                features = data.loc[i][:-1]
                target = data.loc[i][-1]

                weight_sum = self._get_weighted_sum(features, weights)
                prediction = self._sigmoid(weight_sum)
                loss = self._calc_cross_entropy_loss(target, prediction)
                individual_loss.append(loss)
                weights = self._update_weights(weights, target, prediction, features)
                self.bias = self._update_bias(self.bias, target, prediction)
            
            avg_loss = sum(individual_loss) / len(individual_loss)
            self.final_epoch_loss.append(avg_loss)
            print(f"* * * * * * Epoch: {epoch} , Loss: {avg_loss}")
                
    
    def _get_weighted_sum(self, features: np.ndarray, weights: np.ndarray) -> np.float64:
        return np.dot(features, weights) + self.bias
    
    def _sigmoid(self, x: np.float64) -> np.float64:
        return 1 / (1 + np.exp(-x))

    def _calc_cross_entropy_loss(self, target: np.float64, prediction: np.float64) -> np.float64:
        loss = target * np.log10(prediction) + (1-target) * np.log10(1-prediction)
        return -loss

    def _update_weights(self, weights: np.ndarray, target: np.ndarray, prediction: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        new_weights = []
        for x, old_weight in zip(features, weights):
            new_weight = old_weight + self.learning_rate * (target - prediction) * x
            new_weights.append(new_weight)
        return new_weights
    
    def _update_bias(self, bias: float, target: np.ndarray, prediction: np.ndarray) -> float:
        return bias + self.learning_rate * (target - prediction)

data, weights = random_generator(500, 4)
model = dnn()
model.train(data, weights)

