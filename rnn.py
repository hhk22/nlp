
import numpy as np
import pandas as pd
from typing import *

def random_generator(n_data: int, n_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    rand_generator = np.random.default_rng()
    data = rand_generator.random((n_data, n_features))
    targets = rand_generator.random((n_data))
    return data, targets

class RNN:
    def __init__(self) -> None:
        self.global_weight = [1, 1]
        self.local_weight = [0.0001, 0.001]
        self.W_sign = [0, 0]

        self.eta_p = 1.2
        self.eta_n = 0.5
    
    def _state_handler(self, X: np.ndarray, previous_state: np.ndarray) -> np.ndarray:
        return X * self.global_weight[0] + previous_state * self.global_weight[1]
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        S = np.zeros((X.shape[0], X.shape[1]+1))
        for k in range(0, X.shape[1]):
            next_state = self._state_handler(X[:,k], S[:,k])
            S[:, k+1] = next_state
        return S
    
    def _backward_propagation(self, X: np.ndarray, S: np.ndarray, grad_out: np.ndarray) -> Tuple[List[float], np.ndarray]:
        grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
        grad_over_time[:,-1] = grad_out

        wx_grad = 0
        wy_grad = 0
        for k in range(X.shape[1], 0, -1):
            wx_grad += np.sum(grad_over_time[:, k] * X[:, k-1])
            wy_grad += np.sum(grad_over_time[:, k] * S[:, k-1])
        
            grad_over_time[:, k-1] = grad_over_time[:,k] * self.global_weight[1]
        
        return (wx_grad, wy_grad), grad_over_time
    
    def _update_rprop(self, X: np.ndarray, y: np.ndarray, W_prev_sign: List[float], local_weight: List[float]) -> None:
        S = self.forward_propagation(X)
        grad_out = 2*(S[:,-1] - y) / 500
        W_grads, _ = self._backward_propagation(X, S, grad_out)
        self.W_sign = np.sign(W_grads)
        
        for i, _ in enumerate(self.global_weight):
            if self.W_sign[i] == W_prev_sign[i]:
                local_weight[i] *= self.eta_p
            else:
                local_weight[i] *= self.eta_n
        
        self.local_weight = local_weight

    def train(self, X: np.ndarray, y: np.ndarray, epoches: int=500):
        self._update_rprop(X, y, self.W_sign, self.local_weight)
        for epoch in range(epoches):
            self._update_rprop(X, y, self.W_sign, self.local_weight)

            for i, _ in enumerate(self.global_weight):
                self.global_weight[i] -= self.W_sign[i] * self.local_weight[i]

trainX, trainY = random_generator(500, 4)
testX, testY = random_generator(5, 4)
model = RNN()
model.train(trainX, trainY)
y = model.forward_propagation(testX)[:, -1]
print(f"target: {testY}")
print(f"predicted: {y}")