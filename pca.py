import sys
# import sklearn
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

filename = ['A1.jpg', 'A2.jpg', 'A3.jpg', 'B1.jpg', 'B2.jpg', 'B3.jpg', 'C1.jpg', 'C2.jpg', 'C3.jpg', 'D1.jpg', 'D2.jpg', 'D3.jpg', 'E1.jpg', 'E2.jpg', 'E3.jpg', 'F1.jpg', 'F2.jpg', 'F3.jpg',]


for i in range(1, len(filename)):
    img = mpimg.imread('training_set/' + filename[i])
    print(img.shape)
    plt.axis('off')
    plt.imshow(img)

    # reshape the image
    img_r = np.reshape(img, (img.shape[0], img.shape[1]* img.shape[2]))
    print(img_r.shape)

    ipca = PCA(20).fit(img_r)
    img_c = ipca.transform(img_r)
    print(img_c.shape)
    print(np.sum(ipca.explained_variance_ratio_))

    temp = ipca.inverse_transform(img_c)
    print(temp.shape)

    temp = np.reshape(temp, img.shape)

    print(temp.shape)

    plt.axis('off')
    plt.imshow(temp)
    mpimg.imsave('compressedImages/'+ filename[i],temp)
    plt.show()