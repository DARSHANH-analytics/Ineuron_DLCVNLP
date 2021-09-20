import pandas as pd
import numpy as np

class perceptron:
    def __init__(self,lr,epochs):
        self.lr = lr
        self.epochs = epochs
                
    def fit(self,x,y):
        self.x = x
        self.y = y
        print('------inputs---------')
        print(self.x)
        #x_with_bias = np.concatenate([self.x,-np.ones((self.x.shape[0],1))],axis=1)
        #x_with_bias = pd.concat([self.x,pd.Series(-np.ones(self.x.shape[0]))],axis=1)
        x_with_bias = pd.concat([self.x,pd.Series(-np.ones(len(self.x)))],axis=1)
        print('------x_with_bias---------')  
        print(x_with_bias)
        self.weights = np.random.randn(x_with_bias.shape[1])
        print('------weights---------')
        print(self.weights)
        for e in range(self.epochs):
            z = np.dot(x_with_bias,self.weights)
            y_pred = np.where(z>0,1,0)
            print('------y_pred---------')   
            print(y_pred)
            self.y_error = self.y - y_pred
            print('------y_error---------')
            print(self.y_error)
            print('------x_with_bias.T---------Transforming to match the error matrix since column of first matrix to match \
the row of second matrix')
            if min(self.y_error) == 0 and max(self.y_error) == 0 :
                break;
            print(x_with_bias.T)            
            self.weights = self.weights + self.lr * np.dot(x_with_bias.T,self.y_error)
            print(f"updated weights after epoch:\n{e} : \n{self.weights}")            
    
    def predict(self,x):
        x_with_bias = np.concatenate([x,-np.ones((len(x),1))],axis=1)
        z = np.dot(x_with_bias,self.weights)
        return np.where(z>0,1,0)
    
    def total_loss(self):
        total_loss = np.sum(self.y_error)
        print(f"total loss: {total_loss}")
        return total_loss