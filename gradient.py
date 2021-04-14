import numpy as np
from differentiation import numerical_gradient
from softmax import softmax
from activation_func import sigmoid
from loss_function import cross_entropy_error


# lr is learning rate
# step_num is the number of times we will do this
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    # x is input data and t is the answer label
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


# net = SimpleNet()
# 가중치 매개변
# print(net.W)
