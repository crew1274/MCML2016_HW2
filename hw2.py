import os
import numpy as np
import cv2
import sys

#顯示opencv的版本以及要測試的手勢圖片英文代碼
print ("OpenCV version :  {0}".format(cv2.__version__))
print("test:",sys.argv[1])

bin_n = 16

#----------------------------使用hog方法找出特徵------------------------------------------
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     
    return hist

#指定圖片的路徑----------------------------------
training="CSL/training/"
test="CSL/test/"

training_set = []
training_labels=[]
test_set =[]



for file in os.listdir(training):
    if file.find(str(sys.argv[1])) == 0:
        img = cv2.imread(training + file)
        h=hog(img)
        training_set.append(h)
        training_labels.append(1)
    else:
        img = cv2.imread(training + file)
        h=hog(img)
        training_set.append(h)
        training_labels.append(-1)

#svm參數設定---------------------------------------------
SVM = cv2.ml.SVM_create()
SVM.setKernel(cv2.ml.SVM_LINEAR)
SVM.setP(0.2)
SVM.setType(cv2.ml.SVM_EPS_SVR)
SVM.setC(1.0)
SVM.train(np.float32(training_set), cv2.ml.ROW_SAMPLE, np.float32(training_labels))
#------------------------------------------------------
for file in os.listdir(test):
    if file.find(str(sys.argv[1])) == 0:
        img = cv2.imread(test + file)
        h=hog(img)
        test_set.append(h)

testData = np.float32(test_set)
result = SVM.predict(testData)
error=0
count=0
for x in result[1]:
    print("value=",float(x))
    count=count+1
    if float(x) < 0:
            error=error+1
            
#算出失敗率--------------------------------------
print("判斷失敗次數:", error)
print("執行次數:", count)
print(float(error/count))



         

    

