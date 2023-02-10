import sys
from sklearn.decomposition import RandomizedPCA
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

filename = []

for i in range(1, len(sys.argv)):
    img = mpimg.imread('input/' + sys.argv[i])
    print(img.shape)
    plt.axis('off')
    plt.imshow(img)

    # reshape the image
    img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
    print(img_r.shape)

    ipca = RandomizedPCA(int(sys.argv[1]).fit(img_r))
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