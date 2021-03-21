
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
        self.grad = {}
        self.para = {}

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
            
            if image_data.size[0] != self.dim1 or image_data.size[1] != self.dim2:

                image_data = image_data.resize((self.dim1,self.dim2))

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
        self.data_X = np.zeros((1,self.dim1,self.dim2,self.dim3))

        for data in data_list:
            
            path = data_path + "/" + data
            data_image = plt.imread(path).reshape((1,self.dim1,self.dim2,self.dim3))

            self.data_X = np.append(self.data_X,data_image,axis = 0)
            self.data_Y.append(self.label_name(data))


        self.data_X = np.delete(self.data_X,(0),axis = 0)
        self.data_X = (self.data_X.reshape(self.data_X.shape[0],-1).T)/255
        self.data_Y = np.array(self.data_Y).reshape(1,len(self.data_Y))
        

    def initialize_para_zero(self):

        """
         w : (1,num_of_px_h*num_of_px_w*depth)
         b : scalar

        """

        w = np.zeros((1,self.dim1*self.dim2*self.dim3))
        b = 0

        return w,b


    def sigmoid(self,z):

        sig = 1/(np.exp(-z)+1) #<- overflow [sometime )
        #sig = 1/(scipy.special.expit(-z)+1)
        #sig = 1/(np.tanh(-z*0.000000001)+1)
        


        return sig


    def propagate(self,w,b,X,Y):

        """
        w : (1,num_of_px_h*num_of_px_w*depth)
        b : scalar
        X : (num_of_px_h*num_of_px_w*depth,num_of_samples)
        Y : (1,num_of_samples)
        Z : (1,num_of_samples)
        """

        m = X.shape[1]
        
        #Forward Propagation
        Z = np.dot(w,X) + b
        A = self.sigmoid(Z)
        cost = np.sum((-Y*np.log(A)-(1-Y)*np.log(1-A)))/m

        #Backward Propagation
        dw = np.dot(X,(A-Y).T)/m
        db = np.sum((A-Y))/m

        #Summmarize
        self.grad["dw"] = dw.T
        self.grad["db"] = db
        
        return cost

    

    def Optimization(self,w,b,X,Y,Iteration = 2000,alpha = 0.01,print_cost = False):


        """
        w : (1,num_of_px_h*num_of_px_w*depth)
        b : scalar
        X : (num_of_px_h*num_of_px_w*depth,num_of_samples)
        Y : (1,num_of_samples)
        Z : (1,num_of_samples)
        
        """

        for i in range(Iteration):

            #For and back Propagation
            cost = self.propagate(w,b,X,Y)

            #Update parameter
            w = w - alpha*self.grad["dw"]
            b = b - alpha*self.grad["db"]

            if print_cost and i%100 == 0:

                print("Cost after iteration %i: %f" %(i, cost))
                

        #Summmarize             
        self.para["w"] = w
        self.para["b"] = b

    def Classification(self,w,b,sample):

        """
        w : (1,num_of_px_h*num_of_px_w*depth)
        b : scalar
        sample : (num_of_px_h*num_of_px_w*depth,num_of_samples)
        Z : (1,num_of_samples)
        A : (1,num_of_samples)
        """

        Z = np.dot(w,sample)+b
        A = self.sigmoid(Z)

        result = []
        
        for i in range(A.shape[1]):

            if A[0][i] > 0.5:

                result.append(1)

            else:

                result.append(0)


        return np.array(result).reshape(1,len(result))

    
    def Modelling(self,X,Y,Iteration = 2000,alpha = 0.5,print_cost = False):
        

        """
        w : (1,num_of_px_h*num_of_px_w*depth)
        b : scalar
        X : (num_of_px_h*num_of_px_w*depth,num_of_samples)
        Y : (1,num_of_samples)
        Z : (1,num_of_samples)
        
        """

        #initialization
        w,b = self.initialize_para_zero()

        #Train Model
        self.Optimization(w,b,X,Y,Iteration,alpha,print_cost)

        #Predict Result
        Prediction_Train = self.Classification(self.para["w"],self.para["b"],X)
        Accuracy_Train = 100 - np.mean(np.abs(Prediction_Train - Y))*100

        print("Accuracy(Training Set):{}%".format(Accuracy_Train))























    

    
    





    


































