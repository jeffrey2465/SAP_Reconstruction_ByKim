import cv2
import numpy as np
import random

def integralImgSqDiff(PaddedImg, Ds, t1, t2):
    row_end_of_PaddedImg, col_end_of_PaddedImg=np.array(PaddedImg.shape)+1 #matlab에서는 1:N이면 N이 포함이지만 파이선에서는 N-1까지라 이에 대한 변환 보정
    Dist2 = (PaddedImg[Ds: row_end_of_PaddedImg - Ds-1, Ds: col_end_of_PaddedImg - Ds-1]-PaddedImg[Ds + t1: row_end_of_PaddedImg - Ds + t1-1, Ds + t2: col_end_of_PaddedImg - Ds + t2-1])**2
    Sd=np.cumsum(Dist2,axis=0)
    Sd = np.cumsum(Sd, axis=1)
    return Sd

def amf_bykim(x, Ds, ds, B): #AMF

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter

    return M

def namf_bykim(x, Ds, ds, B):

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter

    y=M
    noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]

    I=y
    PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    #print('PaddedV', PaddedV.shape)

    average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    wmax=average #-----------원이미지 크기의 0행렬
    sweight=average #-----------원이미지 크기의 0행렬

    h2=hs*hs
    d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2

    for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
        for t2 in range(-Ds,Ds+1):
            if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
                continue
            Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
            #print('Sd', Sd.shape)
            row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
            SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
                      -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
            SqDist2 = SqDist2 / d;
            w = np.exp(-SqDist2 / h2)

            row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
            v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
            #print('w and v shape', w.shape, v.shape)

            average=average+w*v;
            wmax=np.maximum(wmax,w);
            sweight = sweight + w;
    average=average+0*I
    average=average/sweight
    DenoisedImage=average


    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                DenoisedImage[i,j]=y[i,j]

    DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)

    return DenoisedImage

def proposed1_bykim(x, Ds, ds, B): #amf+거리 기반 weighted sum

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    #weighted average를 위해 수정한 부분
                    if(Ws.shape[0]==Ws.shape[1]):
                        kernel1d = cv2.getGaussianKernel(Ws.shape[0], 1)
                        kernel2d = np.outer(kernel1d, kernel1d.transpose())
                        kernel2d=kernel2d*Ws/np.sum(kernel2d*Ws)

                        M[i, j] = np.sum(tmp * kernel2d)  # 윈도우 내 정상픽셀의 weighted 평균
                    else:
                        M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter
    #
    # y=M
    # noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    # #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    # hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]
    #
    # I=y
    # PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    # #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    # PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    # #print('PaddedV', PaddedV.shape)
    #
    # average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    # wmax=average #-----------원이미지 크기의 0행렬
    # sweight=average #-----------원이미지 크기의 0행렬
    #
    # h2=hs*hs
    # d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2
    #
    # for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
    #     for t2 in range(-Ds,Ds+1):
    #         if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
    #             continue
    #         Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
    #         #print('Sd', Sd.shape)
    #         row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
    #         SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
    #                   -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
    #         SqDist2 = SqDist2 / d;
    #         w = np.exp(-SqDist2 / h2)
    #
    #         row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
    #         v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
    #         #print('w and v shape', w.shape, v.shape)
    #
    #         average=average+w*v;
    #         wmax=np.maximum(wmax,w);
    #         sweight = sweight + w;
    # average=average+0*I
    # average=average/sweight
    # DenoisedImage=average
    #
    #
    # for i in range(xlen):
    #     for j in range(ylen):
    #         if f[i,j]==0:
    #             DenoisedImage[i,j]=y[i,j]
    #
    # DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)
    #
    # # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    # restored_image = cv2.bilateralFilter(DenoisedImage, 5, 8, 8)
    # for i in range(xlen):
    #     for j in range(ylen):
    #         if f[i,j]==0:
    #             restored_image[i,j]=y[i,j]
    # #-------------------------------------------------------------------


    return M


def proposed2_bykim(x, Ds, ds, B): #proposed1 + bilateral

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    #weighted average를 위해 수정한 부분
                    if(Ws.shape[0]==Ws.shape[1]):
                        kernel1d = cv2.getGaussianKernel(Ws.shape[0], 1)
                        kernel2d = np.outer(kernel1d, kernel1d.transpose())
                        kernel2d=kernel2d*Ws/np.sum(kernel2d*Ws)

                        M[i, j] = np.sum(tmp * kernel2d)  # 윈도우 내 정상픽셀의 weighted 평균
                    else:
                        M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter
    #
    # y=M
    # noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    # #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    # hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]
    #
    # I=y
    # PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    # #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    # PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    # #print('PaddedV', PaddedV.shape)
    #
    # average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    # wmax=average #-----------원이미지 크기의 0행렬
    # sweight=average #-----------원이미지 크기의 0행렬
    #
    # h2=hs*hs
    # d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2
    #
    # for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
    #     for t2 in range(-Ds,Ds+1):
    #         if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
    #             continue
    #         Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
    #         #print('Sd', Sd.shape)
    #         row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
    #         SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
    #                   -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
    #         SqDist2 = SqDist2 / d;
    #         w = np.exp(-SqDist2 / h2)
    #
    #         row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
    #         v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
    #         #print('w and v shape', w.shape, v.shape)
    #
    #         average=average+w*v;
    #         wmax=np.maximum(wmax,w);
    #         sweight = sweight + w;
    # average=average+0*I
    # average=average/sweight
    # DenoisedImage=average
    #
    #
    # for i in range(xlen):
    #     for j in range(ylen):
    #         if f[i,j]==0:
    #             DenoisedImage[i,j]=y[i,j]
    #
    # DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)
    #
    # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    y=M.copy()
    restored_image = cv2.bilateralFilter(M, 5, 8, 8)
    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                restored_image[i,j]=y[i,j]
    #-------------------------------------------------------------------


    return restored_image


def proposed3_bykim(x, Ds, ds, B): #namf + weighted + bilateral

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    #weighted average를 위해 수정한 부분
                    if(Ws.shape[0]==Ws.shape[1]):
                        kernel1d = cv2.getGaussianKernel(Ws.shape[0], 1)
                        kernel2d = np.outer(kernel1d, kernel1d.transpose())
                        kernel2d=kernel2d*Ws/np.sum(kernel2d*Ws)

                        M[i, j] = np.sum(tmp * kernel2d)  # 윈도우 내 정상픽셀의 weighted 평균
                    else:
                        M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter
    #
    y=M
    noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]

    I=y
    PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    #print('PaddedV', PaddedV.shape)

    average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    wmax=average #-----------원이미지 크기의 0행렬
    sweight=average #-----------원이미지 크기의 0행렬

    h2=hs*hs
    d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2

    for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
        for t2 in range(-Ds,Ds+1):
            if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
                continue
            Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
            #print('Sd', Sd.shape)
            row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
            SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
                      -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
            SqDist2 = SqDist2 / d;
            w = np.exp(-SqDist2 / h2)

            row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
            v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
            #print('w and v shape', w.shape, v.shape)

            average=average+w*v;
            wmax=np.maximum(wmax,w);
            sweight = sweight + w;
    average=average+0*I
    average=average/sweight
    DenoisedImage=average

    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                DenoisedImage[i,j]=y[i,j]

    DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)
    #
    # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    restored_image = cv2.bilateralFilter(DenoisedImage, 5, 8, 8)
    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                restored_image[i,j]=y[i,j]
    #-------------------------------------------------------------------

    return restored_image

def proposed4_bykim(x, Ds, ds, B): #namf + weighted + bilateral + 자신 weight 포함

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    #weighted average를 위해 수정한 부분
                    if(Ws.shape[0]==Ws.shape[1]):
                        kernel1d = cv2.getGaussianKernel(Ws.shape[0], 1)
                        kernel2d = np.outer(kernel1d, kernel1d.transpose())
                        kernel2d=kernel2d*Ws/np.sum(kernel2d*Ws)

                        M[i, j] = np.sum(tmp * kernel2d)  # 윈도우 내 정상픽셀의 weighted 평균
                    else:
                        M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter
    #
    y=M
    noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]

    I=y
    PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    #print('PaddedV', PaddedV.shape)

    average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    wmax=average #-----------원이미지 크기의 0행렬
    sweight=average #-----------원이미지 크기의 0행렬

    h2=hs*hs
    d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2

    for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
        for t2 in range(-Ds,Ds+1):
            # if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
            #     continue
            Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
            #print('Sd', Sd.shape)
            row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
            SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
                      -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
            SqDist2 = SqDist2 / d;
            w = np.exp(-SqDist2 / h2)

            row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
            v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
            #print('w and v shape', w.shape, v.shape)

            average=average+w*v;
            wmax=np.maximum(wmax,w);
            sweight = sweight + w;
    average=average+0*I
    average=average/sweight
    DenoisedImage=average

    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                DenoisedImage[i,j]=y[i,j]

    DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)
    #
    # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    restored_image = cv2.bilateralFilter(DenoisedImage, 5, 8, 8)
    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                restored_image[i,j]=y[i,j]
    #-------------------------------------------------------------------

    return restored_image


def proposed5_bykim(x, Ds, ds, B): #proposed1 + pixel 신뢰도 weight average

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    weighted_map = np.ones(x.shape)*8 # ----------noise가 없는 픽셀은 최대 weight 8
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s = 1;  # ------------논문에서의 윈도우 크기를 나타내는 w
                g = N[max(i - s, 0):min(i + s + 1, xlen),
                    max(j - s, 0):min(j + s + 1, ylen)]  # --------------논문에 Sij에 해당하는 window
                num = np.sum(g)  # -------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num == 0 and s < smax:  # ----------s max가 3이기 때문에 최대 크기는 7x7
                    s += 1
                    g = N[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    num = np.sum(g)
                    if s==1:
                        weighted_map[i,j]=num

                if s <= smax and np.sum(
                        g) > 0:  # ----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    Ws = 1 - f[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1,
                                                                                     ylen)]  # Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    #weighted average를 위해 수정한 부분
                    if(Ws.shape[0]==Ws.shape[1]):
                        kernel1d = cv2.getGaussianKernel(Ws.shape[0], 1)
                        kernel2d = np.outer(kernel1d, kernel1d.transpose())
                        kernel2d=kernel2d*Ws/np.sum(kernel2d*Ws)

                        M[i, j] = np.sum(tmp * kernel2d)  # 윈도우 내 정상픽셀의 weighted 평균
                    else:
                        M[i, j] = np.sum(tmp * Ws) / np.sum(Ws)  # 윈도우 내 정상픽셀의 평균
                else:  # ----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i - s, 0):min(i + s + 1, xlen), max(j - s, 0):min(j + s + 1, ylen)]
                    pd = (tmp == x[i, j]).sum() / (tmp.size)

                    if pd > B:  # ---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i, j] = 0
                        continue
                    up = max(i - 1, 0)
                    left = max(j - 1, 0)
                    tmp = []
                    tmp.append(M[up, left])
                    tmp.append(M[up, j])
                    tmp.append(M[i, left])
                    # print(tmp)
                    M[i, j] = np.median(tmp)  # --------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter
    #
    # y=M
    # noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    # #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    # hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]
    #
    # I=y
    # PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    # #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    # PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    # #print('PaddedV', PaddedV.shape)
    #
    # average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    # wmax=average #-----------원이미지 크기의 0행렬
    # sweight=average #-----------원이미지 크기의 0행렬
    #
    # h2=hs*hs
    # d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2
    #
    # for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
    #     for t2 in range(-Ds,Ds+1):
    #         if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
    #             continue
    #         Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
    #         #print('Sd', Sd.shape)
    #         row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
    #         SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
    #                   -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
    #         SqDist2 = SqDist2 / d;
    #         w = np.exp(-SqDist2 / h2)
    #
    #         row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
    #         v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
    #         #print('w and v shape', w.shape, v.shape)
    #
    #         average=average+w*v;
    #         wmax=np.maximum(wmax,w);
    #         sweight = sweight + w;
    # average=average+0*I
    # average=average/sweight
    # DenoisedImage=average
    #
    #
    # 복원된 영상에 대해서 복원값에 대한 신뢰도를 weight map으로 구성하고 이를 이용한 weighted average 후처리 수행
    y=M.copy()
    for i in range(1, xlen-1):
        for j in range(1, ylen-1):
            if f[i,j]==1:
                tmp=M[i-1:i+2, j-1:j+2]
                y[i,j]=np.sum(np.array(tmp)*weighted_map[i-1:i+2, j-1:j+2])/np.sum(weighted_map[i-1:i+2, j-1:j+2])



    #DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)
    #
    # # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    # restored_image = cv2.bilateralFilter(DenoisedImage, 5, 8, 8)
    # for i in range(xlen):
    #     for j in range(ylen):
    #         if f[i,j]==0:
    #             restored_image[i,j]=y[i,j]
    # #-------------------------------------------------------------------


    return y

def amf(x, Ds, ds, B):

    #... AMF구현 (미디언 블록사이즈 가변 + noiseless pixel만 활용 + Texture pixel with 0, 255 고려
    M=x
    N=np.ones(x.shape)
    N[np.logical_or(x==0,x==255)]=0 #-------1: 0<pixel<255,    0: pixel=0 || pixel=255
    #print(N)

    f=np.zeros(x.shape)
    f[np.logical_or(x == 0, x == 255)] = 1 #------- 1: pixel=0 || pixel=255,   0: 0<pixel<255,   ==> 향후 0, 255이지만 texture의 경우 값을 1로 변경
    #print(f)

    smax=3
    xlen, ylen = x.shape
    #print(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            if N[i,j]==0: #---------- 화소값==0 || 255
                s=1; #------------논문에서의 윈도우 크기를 나타내는 w
                g=N[max(i-s, 0):min(i+s, xlen), max(j-s,0):min(j+s,ylen)] #--------------논문에 Sij에 해당하는 window
                num=np.sum(g) #-------------여기서 num: 윈도우 내에 0<pixel 값<255인 개수

                while num==0 and s<smax: #----------s max가 3이기 때문에 최대 크기는 7x7
                    s+=1
                    g=N[max(i-s, 0):min(i+s, xlen), max(j-s,0):min(j+s,ylen)]
                    num = np.sum(g)

                if s<=smax and np.sum(g)>0: #----------윈도우 내에 noiseless pixel이 하나라도 있으면, 윈도우 내의 0<pixel<255에 해당하는 화소값들의 평균 구하기 [아이디어 1 : AMF로 1차 복원시 Weight 평균을 구해볼 것]
                    tmp=x[max(i-s, 0):min(i+s, xlen), max(j-s,0):min(j+s,ylen)]
                    Ws=1-f[max(i-s, 0):min(i+s, xlen), max(j-s,0):min(j+s,ylen)] #Ws------------ 1: 0<pixel<255,    0: pixel=0 || pixel=255
                    M[i,j]=np.sum(tmp*Ws)/np.sum(Ws) # 윈도우 내 정상픽셀의 평균
                else: #----------------윈도우 내에 noiseless pixel이 하나도 없으면, 윈도우 내에 x(i,j)와 동일한 값을 가지는 비율을 구함
                    tmp = x[max(i-s, 0):min(i+s, xlen), max(j-s, 0):min(j+s, ylen)]
                    pd=(tmp==x[i,j]).sum()/(tmp.size)

                    if pd>B: #---------------------윈도우내 x(i,j)값을 가지는 화소의 비율이 B(여기서는 0.8)보다 크면 texture로 인식
                        f[i,j]=0
                        continue
                    up=max(i-1,0)
                    left=max(j-1,0)
                    tmp=[]
                    tmp.append(M[up,left])
                    tmp.append(M[up, j])
                    tmp.append(M[i,left])
                    #print(tmp)
                    M[i,j]=np.median(tmp) #--------------논문에서 평균을 구했는데 여기서는 median을 사용했네

    #non-local mean filter

    y=M
    noise=np.sum(f)/(xlen*ylen) # ------------------noise pixel의 비율 (0, 255이면서 texture가 아니라고 판명된 경우는 제외)
    #2차함수이지만 noise 비율의 범위인 0~1사이에서는 1차함수처럼 거의 직선
    hs=4.5595 + 6.0314*noise + 2.2186*(noise**2); #---------- 논문의 식(10), smoothing parameter: noise 커지면 h도 커지도록 함 (단 이 경우 2차원이라 실제 noise가 작은데 큰 hs를 가질 수도 있음) [아이디어 2 : 1차 함수로 ML 적용]

    I=y
    PaddedImg=np.pad(I, (Ds+ds+1, Ds+ds+1), 'symmetric') #------------------Ds=2, ds=20, 즉 Ds+ds+1=23을 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+46)x(이미지y+46)
    #print('PaddedImg', PaddedImg.shape) #------------------Ds=2, 즉 이미지의 상하좌우로 패딩 (패딩값은 symmetric)하게 ==> 사이즈 (이미지x+4)x(이미지y+4) :
    PaddedV=np.pad(I, (Ds, Ds), 'symmetric')
    #print('PaddedV', PaddedV.shape)

    average=np.zeros(I.shape) #----------원이미지 크기의 0행렬
    wmax=average #-----------원이미지 크기의 0행렬
    sweight=average #-----------원이미지 크기의 0행렬

    h2=hs*hs
    d=(2*ds+1)**2 # ------------식 (9)에 포함된 h^2

    for t1 in range(-Ds,Ds+1): #참고문헌 [23]에 있는 fast implementation 방법
        for t2 in range(-Ds,Ds+1):
            if t1==0 and t2==0: #non local mean을 구하는 과정에서 자기 자신의 경우 weight를 0을 할당, 나머지는 거리의 함수
                continue
            Sd = integralImgSqDiff(PaddedImg, Ds, t1, t2)
            #print('Sd', Sd.shape)
            row_end_of_Sd, col_end_of_Sd = np.array(Sd.shape)+1
            SqDist2 = Sd[2 * ds + 2-1:row_end_of_Sd - 1-1, 2 * ds + 2-1: col_end_of_Sd - 1-1]+Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 1-1: col_end_of_Sd - 2 * ds - 2-1] \
                      -Sd[2 * ds + 2-1: row_end_of_Sd - 1-1, 1-1: col_end_of_Sd - 2 * ds - 2-1]-Sd[1-1: row_end_of_Sd - 2 * ds - 2-1, 2 * ds + 2-1: col_end_of_Sd - 1-1] #여기서 2를 Ds로 바꾸어야 정확히 입력매개변수 반영할 듯함. 이렇게 되면 sqDist2는 이미지 사이즈와 동일해짐
            SqDist2 = SqDist2 / d;
            w = np.exp(-SqDist2 / h2)

            row_end_of_PaddedV, color_end_of_PaddedV = np.array(PaddedV.shape)+1
            v = PaddedV[Ds + t1:row_end_of_PaddedV - Ds + t1-1, Ds + t2: color_end_of_PaddedV - Ds + t2-1]
            #print('w and v shape', w.shape, v.shape)

            average=average+w*v;
            wmax=np.maximum(wmax,w);
            sweight = sweight + w;
    average=average+0*I
    average=average/sweight
    DenoisedImage=average


    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                DenoisedImage[i,j]=y[i,j]

    DenoisedImage = np.array(DenoisedImage, dtype=np.uint8)

    # 추가분 bilateral (주관적 화질 개선)-----------------------------------
    restored_image3 = cv2.bilateralFilter(DenoisedImage, 5, 8, 8)
    for i in range(xlen):
        for j in range(ylen):
            if f[i,j]==0:
                restored_image3[i,j]=y[i,j]
    #-------------------------------------------------------------------


    return M, DenoisedImage, restored_image3 #M=AMF, DenoisedImage=NAMF, restored_image3=+bilateral
