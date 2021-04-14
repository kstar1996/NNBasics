import numpy as np


def numerical_diff(f, x):
    h = 1e-4   # 0.0001
    return(f(x+h) - f(x-h)) / (2*h)


# x should be numpy array
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) calculation
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) calculation
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

