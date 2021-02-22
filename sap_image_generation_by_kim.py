import numpy as np
import random
import cv2

def sp_noise(img, prob):
    count_all=0 #전체 noise-free 픽셀 개수
    count_mis = 0  # 전체 noise-free 중 실제 픽셀값이 0-255인 경우 (texture pixel)
    output = np.zeros(img.shape, np.uint8)
    s = np.zeros(img.shape, np.uint8)
    threshold = 1 - prob/2
    row=img.shape[0]
    col=img.shape[1]
    count_0=0
    count_255=0
    for i in range(row):
        for j in range(col):
            rdn = random.random() #0~1
            if rdn < prob/2:
                output[i][j] = 0
                count_0+=1
            elif rdn > threshold:
                output[i][j] = 255
                count_255+=1
            else:
                output[i][j] = img[i][j]
                s[i][j]=1
                count_all+=1
                if img[i][j]==0 or img[i][j]==255:
                    count_mis+=1
    print('전체 noisy 픽셀 수 : 0(pepper 개수 : {0}/{1}={2:.2f}, 255(salt) 개수: {3}/{4}={5:.2f} => 전체 {6:.2f}'.format(count_0, row*col, count_0/(row*col), count_255, row*col, count_255/(row*col), (count_0+count_255)/(row*col)))
    print('전체 noiseless 픽셀 수 중 noise 픽셀로 잘못 판명된 비율 {0}/{1}={2:.4f}'.format(count_mis, count_all, count_mis/count_all))
    return output #


def sp_noise_to_file(img, prob, outfilename):

    output = np.zeros(img.shape, np.uint8)
    s = np.zeros(img.shape, np.uint8)
    threshold = 1 - prob/2
    row=img.shape[0]
    col=img.shape[1]

    for i in range(row):
        for j in range(col):
            rdn = random.random() #0~1
            if rdn < prob/2:
                output[i][j] = 0
            elif rdn > threshold:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
                s[i][j]=1

    cv2.imwrite("noiseimage/{0}_{1:.2f}.png".format(outfilename, prob), output)


    return output #

def generation_noise_images():
    noise_probs=[0.1, 0.3, 0.5, 0.7, 0.9]
    #filenames=[]
    for i in range(1,53):
        for noise_prob in noise_probs:
            filename="[test{0:02d}]_origin".format(i)
            #filenames.append(filename)

            inputfile = "testset/{0}.bmp".format(filename)
            print('inputfile',inputfile)
            img = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)

            outputfile = "{0}".format(filename)
            print('outputfile',outputfile)

            sp_noise_to_file(img,noise_prob, outputfile)


