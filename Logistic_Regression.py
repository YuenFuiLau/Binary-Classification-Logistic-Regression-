
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import os

from PIL import Image
from scipy import ndimage


class Logistic_Regression:

    def __init__(self,path_name = os.getcwd(),data_file_name ="",source_file_name="",height = 1440,width =1080,depth = 3):

        self.data_X = None
        self.data_Y = []
        self.path = path_name
        self.data_file_name = data_file_name
        self.source_file_name = source_file_name
        self.dim1 = height
        self.dim2 = width
        self.dim3 = depth

        self.Data_Filtering()
        self.construct_data_set()
        
    # Source_path = current_directory/"Source file Name" if it has
    def Data_Filtering(self):

        source_path = self.path + "/" + self.source_file_name
        data_path = self.path + "/" + self.data_file_name

        source_list = os.listdir(source_path)
        source_list.remove(".DS_Store")

        for data in source_list:

            image_data = Image.open(source_path+"/"+data)
            
            if image_data.size[0] != 1440 or image_data.size[1] != 1080:

                image_data = image_data.resize((1440,1080))

            image_data.save(data_path+"/"+data)

    def label_name(self,name):

        if name[0] == "y":

            return 1

        else:

            return 0
            

    def construct_data_set(self):

        data_path = self.path + "/" + self.data_file_name

        data_list = os.listdir(data_path)
        data_list.remove(".DS_Store")

        #initialize array

        path = data_path + "/" + data_list[0]
        data_image = plt.imread(path).reshape((1,self.dim1,self.dim2,self.dim3))

        self.data_X = np.array(data_image)
        self.data_Y.append(self.label_name(data_list[0]))

        #---------------------------------------------------------------------

        for data in data_list:
            
            path = data_path + "/" + data
            data_image = plt.imread(path).reshape((1,self.dim1,self.dim2,self.dim3))

            self.data_X = np.append(self.data_X,data_image,axis = 0)
            self.data_Y.append(self.label_name(data))

        self.data_X = self.data_X.reshape(self.data_X.shape[0],-1).T/255
        self.data_Y = np.array(self.data_Y).reshape(1,len(self.data_Y))

                


            





    


































