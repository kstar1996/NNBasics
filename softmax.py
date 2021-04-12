import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # 오버플로우 대책 pg.94
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# a = np.array([1010, 1000, 990])
# print(softmax(a))

