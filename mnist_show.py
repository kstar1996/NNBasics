import sys, os
sys.path.append(os.pardir)
import numpy as np
from get_data import load_mnist
from PIL import Image  # Python Image Library: PIL
# code for showing the mnist image

# changing the numpy image data to PIL data
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# flatten True makes a 1 dimensional numpy array
# so you have to change it into a 28*28 image
(x_train, t_train), (x_test, t_test) \
    = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

#print(img.shape)  # (784,)
img = img.reshape(28, 28)  # changing into the numpy format you want
#print(img.shape)  # (28, 28)

img_show(img)
