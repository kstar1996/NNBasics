import numpy as np
from DeepLearning.functions import sigmoid
from DeepLearning.softmax import softmax
from DeepLearning.differentiation import numerical_gradient
from DeepLearning.functions import cross_entropy_error
from DeepLearning.gradient import sigmoid_grad


class TwoLayerNet:

    # class initialization
    # in order of 입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수
    # hidden size 는 적당한 값 설정
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 각 손실 함수에 대한 기울기 계산
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {'W1': numerical_gradient(loss_w, self.params['W1']),
                 'b1': numerical_gradient(loss_w, self.params['b1']),
                 'W2': numerical_gradient(loss_w, self.params['W2']),
                 'b2': numerical_gradient(loss_w, self.params['b2'])}

        return grads

    # back propagation
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(net.params['W1'].shape)
# net.params['b1'].shape
# net.params['W2'].shape
# net.params['b2'].shape

