'''
2021.02.22
Salt And Pepper Noise 제거 알고리즘 구현
MF: opencv api
AMF: NAMF의 Step1에 해당하는 내용 구현 (논문 구현 Code 파이썬 포팅)
NAMF: NAMF 논문 Step1 + Step2 (논문 구현 Code 파이썬 포팅)
Proposed1 : AMF+거리 기반 weighted sum
Proposed2 : Proposed1 + bilateral
Proposed3 : namf + 거리 기반 weighted sum + bilateral
Proposed4 : proposed3 + 자신 weight 포함
Proposed5 : Proposed1 + pixel 신뢰도 weight average
'''
import cv2

import implemetation_bykim as imp_bykim

if __name__ == '__main__':

    #처음 다양한 noise level의 noisy image를 생성하여 저장 (실행때마다 동일한 결과가 나오도록 (rand때문에 계속 다른 결과))-------
    #generation_noise_images()
    #--------------------------------------------------------------------------------------------------------------

    #이미지 로드 (우선은 매트릭스로 테스트 (4x3)
    # img = np.arange(0,12).reshape(3,4)

    originimages = []  # 전체 테스트 영상
    for i in range(1, 53):
        fname = "testset/[test{0:02d}]_origin.bmp".format(i)
        originimages.append(fname)
    # print(filenames)


    noiseimages = []  # 전체 테스트 영상
    noise_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i in range(1, 53):
        for noise_prob in noise_probs:
            fname = "noiseimage/[test{0:02d}]_origin_{1:.2f}.png".format(i, noise_prob)
            noiseimages.append(fname)
    print(len(noiseimages))

    # img = cv2.imread('testset/[test01]_origin.bmp', cv2.IMREAD_GRAYSCALE)


    f= open('psnrs.csv','w')
    f.write("image, mf,amf,namf,proposed1,proposed2,proposed3,proposed4,proposed5\n")

    for num in range(0,2):

        img = cv2.imread(originimages[num//5], cv2.IMREAD_GRAYSCALE)
        # cv2.namedWindow('original img', cv2.WINDOW_NORMAL)
        # cv2.imshow('original img', img)
        # cv2.waitKey(0)

        #noisy_img=sp_noise(img, 0.6)

        #noisy_img=sp_noise_to_file(img, 0.6, '[test01]_origin')
        noisy_img = cv2.imread(noiseimages[num], cv2.IMREAD_GRAYSCALE)
        noisy_img_clone = noisy_img.copy()

        # 4가지 방법(mf, amf, namf, bilateral) psnr 파일로 저장하기
        # print("noisy_img PSNR = {0:.3f}".format(cv2.PSNR(img, noisy_img)))
        # restored_image0 = cv2.medianBlur(noisy_img, 3)
        # print("restored_image0 (mf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image0)))
        # restored_image1, restored_image2, restored_image3 = amf(noisy_img, 2, 20, 0.8)
        # print("restored_image1 (amf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image1)))
        # print("restored_image2 (namf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image2)))
        # print("restored_image3 (+bilateral) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image3)))

        # amf, namf, proposed = imp_bykim.amf(noisy_img, 2, 20, 0.8)
        # print("amf (amf) PSNR = {0:.3f}".format(cv2.PSNR(img, amf)))
        # print("namf (namf) PSNR = {0:.3f}".format(cv2.PSNR(img, namf)))
        # print("proposed (proposed) PSNR = {0:.3f}".format(cv2.PSNR(img, proposed)))

        #PSNR 계산하기  (함수에서 noisy_img가 변형되어 호출 전 clone으로 복사 후 진행)

        mf_byopencv=cv2.medianBlur(noisy_img, 3)
        mf_byopencv_psnr=cv2.PSNR(img, mf_byopencv)
        print("mf_byopencv (mf) PSNR = {0:.3f}".format(mf_byopencv_psnr))
        noisy_img=noisy_img_clone.copy()
        amf_bykim = imp_bykim.amf_bykim(noisy_img, 2, 20, 0.8)
        amf_bykim_psnr=cv2.PSNR(img, amf_bykim)
        print("amf_bykim (amf) PSNR = {0:.3f}".format(amf_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        namf_bykim=imp_bykim.namf_bykim(noisy_img, 2, 20, 0.8)
        namf_bykim_psnr=cv2.PSNR(img, namf_bykim)
        print("namf_bykim (namf) PSNR = {0:.3f}".format(namf_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        proposed1_bykim=imp_bykim.proposed1_bykim(noisy_img, 2, 20, 0.8)
        proposed1_bykim_psnr=cv2.PSNR(img, proposed1_bykim)
        print("proposed1_bykim (#amf+거리 기반 weighted sum) PSNR = {0:.3f}".format(proposed1_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        proposed2_bykim=imp_bykim.proposed2_bykim(noisy_img, 2, 20, 0.8)
        proposed2_bykim_psnr = cv2.PSNR(img, proposed2_bykim)
        print("proposed2_bykim (#proposed1 + bilateral) PSNR = {0:.3f}".format(proposed2_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        proposed3_bykim=imp_bykim.proposed3_bykim(noisy_img, 2, 20, 0.8)
        proposed3_bykim_psnr = cv2.PSNR(img, proposed3_bykim)
        print("proposed3_bykim (#namf + 거리 기반 weighted sum + bilateral) PSNR = {0:.3f}".format(proposed3_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        proposed4_bykim = imp_bykim.proposed4_bykim(noisy_img, 2, 20, 0.8)
        proposed4_bykim_psnr = cv2.PSNR(img, proposed4_bykim)
        print("proposed4_bykim ( #proposed3 + 자신 weight 포함) PSNR = {0:.3f}".format(proposed4_bykim_psnr))
        noisy_img = noisy_img_clone.copy()
        proposed5_bykim = imp_bykim.proposed5_bykim(noisy_img, 2, 20, 0.8)
        proposed5_bykim_psnr = cv2.PSNR(img, proposed5_bykim)
        print("proposed5_bykim (#proposed1 + pixel 신뢰도 weight average) PSNR = {0:.3f}".format(proposed5_bykim_psnr))

        #psnr file로 저장
        f.write("{8},{0:.3f},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.3f},{7:.3f}\n".format(mf_byopencv_psnr,amf_bykim_psnr,namf_bykim_psnr,proposed1_bykim_psnr,proposed2_bykim_psnr,proposed3_bykim_psnr,proposed4_bykim_psnr,proposed5_bykim_psnr,noiseimages[num]))

        '''
        # 그림 순서대로 한장씩 띄우기
        cv2.namedWindow('noisy image', cv2.WINDOW_NORMAL)
        cv2.imshow('noisy image', noisy_img)
        print("noisy_img PSNR = {0:.3f}".format(cv2.PSNR(img, noisy_img)))
        cv2.waitKey(0)

        restored_image0=cv2.medianBlur(noisy_img,3)
        cv2.namedWindow('restored_image0 (mf)', cv2.WINDOW_NORMAL)
        cv2.imshow('restored_image0 (mf)', restored_image0)
        print("restored_image0 (mf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image0)))
        cv2.waitKey(0)

        restored_image1, restored_image2, restored_image3 = amf(noisy_img, 2, 20, 0.8)

        cv2.namedWindow('restored_image1 (amf)', cv2.WINDOW_NORMAL)
        cv2.imshow('restored_image1 (amf)', restored_image1)
        print("restored_image1 (amf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image1)))
        cv2.waitKey(0)

        cv2.namedWindow('restored_image2 (namf)', cv2.WINDOW_NORMAL)
        cv2.imshow('restored_image2 (namf)', restored_image2)
        print("restored_image2 (namf) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image2)))
        cv2.waitKey(0)

        #restored_image3=cv2.bilateralFilter(restored_image2, 5, 10, 10)
        cv2.namedWindow('restored_image3 (+bilateral)', cv2.WINDOW_NORMAL)
        cv2.imshow('restored_image3 (+bilateral)', restored_image3)
        print("restored_image3 (+bilateral) PSNR = {0:.3f}".format(cv2.PSNR(img, restored_image3)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    f.close()
