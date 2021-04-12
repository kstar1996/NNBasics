#신경망

import numpy as np
from sigmoid import sigmoid


def activation(x):  # 활성화 함수
    return x


def forward(x):
    X = np.array([1.0, 0.5])   # 입력 신호
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 가중치
    B1 = np.array([0.1, 0.2, 0.3])  # 편향
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    a1 = np.dot(X, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3

    y = activation(a3)

    return y


x = np.array([1.0, 0.5])
y = forward(x)
print(y)









