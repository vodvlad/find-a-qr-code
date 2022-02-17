import sys, os
from os import path
import cv2
import numpy as np

def main(argv):
    list = []
    for filename in os.listdir(argv[1]):
        if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png']:
                list.append(filename)

    
    try:
        os.mkdir("results")
        os.mkdir("image_processing")
        os.mkdir("image_processing/close")
        os.mkdir("image_processing/blur")
        os.mkdir("image_processing/gray")
        os.mkdir("image_processing/thresh")
    except:
        print("created")

    for i in range(len(list)):
        full_name = path.basename(list[i])
        print(argv[1] + '/' + full_name)

        img = cv2.imread(argv[1] + '/' + full_name)
        img = cv2.resize(img, (int(480*2), int(640*2)))
        w, h = 480, 640
        imgWarp = img.copy()

        GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('image_processing/gray/gray_' + full_name, GrayImg)

        BlurredFrame = cv2.GaussianBlur(GrayImg, (9, 9), 0)
        cv2.imwrite('image_processing/blur/blur_' + full_name, BlurredFrame)

        thresh = cv2.threshold(BlurredFrame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imwrite('image_processing/thresh/thresh_' + full_name, thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ContourFrame = img.copy()

        cv2.imshow("close", close)
        cv2.imwrite('image_processing/close/close_' + full_name, close)
        ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (255, 0, 255), 4)
        CornerFrame = img.copy()
        maxArea = 0
        biggest = []

        for i in contours :
            area = cv2.contourArea(i)
            if area > 500 :
                peri = cv2.arcLength(i, True)
                edges = cv2.approxPolyDP(i, 0.02*peri, True)
                if area > maxArea and len(edges) == 4 :
                    biggest = edges
                    maxArea = area
                    
        if len(biggest) != 0 :
            biggest = biggest.reshape((4, 2))
            biggestNew = np.zeros((4, 1, 2), dtype= np.int32)
            add = biggest.sum(1)
            biggestNew[0] = biggest[np.argmin(add)]
            biggestNew[3] = biggest[np.argmax(add)]
            dif = np.diff(biggest, axis = 1)
            biggestNew[1] = biggest[np.argmin(dif)]
            biggestNew[2] = biggest[np.argmax(dif)]
            CornerFrame = cv2.drawContours(CornerFrame, biggestNew, -1, (255, 0, 255), 25)
            pts1 = np.float32(biggestNew)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (w, h))

        cv2.imshow("outputImage", imgWarp)
        cv2.imwrite('results/result_' + full_name, imgWarp)
        cv2.waitKey(0)
    





if __name__ == "__main__":
    main(sys.argv)
