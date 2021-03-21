import Logistic_Regression as LR

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import os

from PIL import Image
from scipy import ndimage


#Testing--------------------------------------------------------------------
my_image = "y_66.jpg"
path = os.getcwd()

fname = path + "/Source/" + my_image

#They are the same
image = np.array(plt.imread(fname))
image_test = np.array(Image.open(fname))
#--------------------

#Image Data: When they are called differently(plt.imread(file_path),Image.open(file_path)), image_1.size and image_2.size return different things 
image_1 = plt.imread(fname)
image_2 = Image.open(fname)

#They are different since image_1 and image_2 are of different data type
image_1.size
image_2.size

image_3 = plt.imread(fname)
image_2 = plt.imread(fname)
image_2 = image_2.reshape((1,image_2.shape[0],image_2.shape[1],image_2.shape[2]))
image_1 = image_1.reshape((1,image_1.shape[0],image_1.shape[1],image_1.shape[2]))
data_set = np.empty((1,2884,720,3))
data_set = np.append(data_set,image_1,axis = 0)
data_set = np.append(data_set,image_2,axis = 0)


#data_set = np.array(data_set)

files_name = os.listdir(path+"/Source")

test_image = Image.open(fname)
test_image = test_image.resize((1440,1080))
#test_image.save(os.getcwd()+"/Data/"+"y_2.jpg")

#They are the same
#plt.imshow(plt.imread(fname))
#plt.imshow(image)
#-------------------
#plt.show()

"""
w, b, X, Y = np.array([[1.],[2.]]).reshape(1,2), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]]).reshape(1,3)
a = LR.Logistic_Regression(data_file_name = "Data",source_file_name = "Source")

cost_1 = a.propagate(w,b,X,Y)
list_r = a.Classification(w,b,X)

a.Optimization(w, b, X, Y, 100, 0.009, print_cost = False)
print ("w = " + str(a.para["w"]))
print ("b = " + str(a.para["b"]))
print ("dw = " + str(a.grad["dw"]))
print ("db = " + str(a.grad["db"]))

w,b =a.para["w"],a.para["b"]
cost_2 = a.propagate(w,b,X,Y)

"""


#-----------------------------------------------------------------------------

"""

Test = LR.Logistic_Regression(height=64,width = 64,data_file_name = "Data",source_file_name = "Source")


Test.Modelling(Test.data_X,Test.data_Y,print_cost = True)

"""

#np.savetxt(path+"/"+"Data_X.csv",Test.data_X,delimiter = ',',fmt='%f')
#np.savetxt(path+"/"+"Data_Y.csv",Test.data_Y,delimiter = ',',fmt='%d')


