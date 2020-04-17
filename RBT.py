import numpy as np
import argparse
import cv2
import math
import random
from PIL import Image


def rotate(image, angle, center= None, scale=1.0):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def gaussBlur(image):
    kernel_size = (9, 9)
    sigma = 40
    img = cv2.GaussianBlur(image, kernel_size, sigma)
    return img


class get_degree(object):
    def __init__(self, degree):
        self.degree = degree
        self.angle = self.__call__()

    @staticmethod
    def get_de(angle):
        degre = random.uniform(angle*-1, angle)
        return degre

    def __call__(self):
        degree = self.get_de(self.degree)
       # print(degree)
      #  print("areg")
        pi = math.radians(self.degree)
        return pi

def RGBT(image):

    ran = random.randint(-20, 20)
    #print(ran)
    #if ran < 31:
    pi = get_degree(ran)
    #print(pi)
    image = rotate(image, ran)
    #return image
   # else:
    #    image = gaussBlur(image)
    image = Image.fromarray(cv2.cvtColor(image, 1))
   # print(isinstance(image,np.ndarray))
    return image, ran

def getRotateLabelx(x1,y1,rotate_angle):
    x0=(x1-64)*math.cos(rotate_angle)-(y1-32)*math.sin(rotate_angle)+64
    return x0
def getRotateLabely(x1,y1,rotate_angle):
    y0=(x1-64)*math.sin(rotate_angle)-(y1-32)*math.cos(rotate_angle)+32
    return y0
'''
def main():
     image = cv2.imread("1400.jpg")
     #RGBT(image)
     image = rotate(image, 30)
     cv2.waitKey(1)
     for i in range(100):
        image = cv2.imread("./pic/"+str(1400+i)+".jpg")
        image = RGBT(image)
        str2 = "./RBTpics/"+str(i)+".jpg"
        cv2.imwrite(str2, image)

main()
'''