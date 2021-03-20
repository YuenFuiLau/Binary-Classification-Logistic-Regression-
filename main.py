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

#They are different
image_1.size
image_2.size

image_2 = plt.imread(fname)
image_2 = image_2.reshape((image_2.shape[0],image_2.shape[1],image_2.shape[2],1))
image_1 = image_1.reshape((image_1.shape[0],image_1.shape[1],image_1.shape[2],1))
data_set = np.empty((2884,720,3,1))
data_set = np.append(data_set,image_1,axis = 3)
data_set = np.append(data_set,image_2,axis = 3)


#data_set = np.array(data_set)

files_name = os.listdir(path+"/Source")

test_image = Image.open(fname)
test_image = test_image.resize((1440,1080))
#test_image.save(os.getcwd()+"/Data/"+"y_2.jpg")

#They are the same
plt.imshow(plt.imread(fname))
plt.imshow(image)
#-------------------
#plt.show()

#-----------------------------------------------------------------------------

a = LR.Logistic_Regression(data_file_name = "Data",source_file_name = "Source")
