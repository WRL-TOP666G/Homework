'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import math

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    
    #Number of Red, Green, Blue, Total pics
    self.total_length=3694
    self.R_length=1352
    self.G_length=1199
    self.B_length=1143
    
    #Probability
    self.R_prob=(self.R_length/self.total_length)
    self.G_prob=(self.G_length/self.total_length)
    self.B_prob=(self.B_length/self.total_length)
    
    #Mean
    self.RR_mean=0.7525060911938757
    self.RG_mean=0.34808562478245914
    self.RB_mean=0.34891228680821595
    
    self.GR_mean=0.350609167770528
    self.GG_mean=0.7355148898592014
    self.GB_mean=0.3294935321918604
    
    self.BR_mean=0.34735903110150496
    self.BG_mean=0.3311135127716902
    self.BB_mean=0.7352649546257728
    
    #Variance
    self.sigma_square_RR=0.0370867022105675
    self.sigma_square_RG=0.06201456207728437
    self.sigma_square_RB=0.062068457840488304

    self.sigma_square_GR=0.055781152694426794
    self.sigma_square_GG=0.034814963937182225
    self.sigma_square_GB=0.056068642903266165
  
    self.sigma_square_BR=0.05458537840583435
    self.sigma_square_BG=0.056883076264008056
    self.sigma_square_BB=0.035771903528807894
    
    #Other
    self.pi=3.1415
    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    #Use Naive Bayes Classifier and Gaussian Naive Bayes
    n=len(X)
    y= np.empty([n, 1])
    for i in range(len(X)):
        r=X[i][0]
        g=X[i][1]
        b=X[i][2]
        p_RR=math.exp(-((r-self.RR_mean)**2)/(2*self.sigma_square_RR) )/math.sqrt((2*self.pi*self.sigma_square_RR) )
        p_GR=math.exp(-((g-self.RG_mean)**2)/(2*self.sigma_square_RG) )/math.sqrt((2*self.pi*self.sigma_square_RG) )
        p_BR=math.exp(-((b-self.RB_mean)**2)/(2*self.sigma_square_RB) )/math.sqrt((2*self.pi*self.sigma_square_RB) )
        
        p_RG=math.exp(-((r-self.GR_mean)**2)/(2*self.sigma_square_GR) )/math.sqrt((2*self.pi*self.sigma_square_GR) )
        p_GG=math.exp(-((g-self.GG_mean)**2)/(2*self.sigma_square_GG) )/math.sqrt((2*self.pi*self.sigma_square_GG) )
        p_BG=math.exp(-((b-self.GB_mean)**2)/(2*self.sigma_square_GB) )/math.sqrt((2*self.pi*self.sigma_square_GB) )
        
        p_RB=math.exp(-((r-self.BR_mean)**2)/(2*self.sigma_square_BR) )/math.sqrt((2*self.pi*self.sigma_square_BR) )
        p_GB=math.exp(-((g-self.BG_mean)**2)/(2*self.sigma_square_BG) )/math.sqrt((2*self.pi*self.sigma_square_BG) )
        p_BB=math.exp(-((b-self.BB_mean)**2)/(2*self.sigma_square_BB) )/math.sqrt((2*self.pi*self.sigma_square_BB) )
        

        
        p_R=self.R_prob*p_RR*p_GR*p_BR
        p_G=self.G_prob*p_RG*p_GG*p_BG
        p_B=self.B_prob*p_RB*p_GB*p_BB
        
        print("%f, %f, %f",p_R, p_G, p_B)
        
        #Decide the pixel is Red, Green, or Blue
        if p_R>=p_G and p_R>=p_B:
            y[i]=1
        if p_G>=p_R and p_G>=p_B:
            y[i]=2
        if p_B>=p_R and p_B>=p_G:
            y[i]=3
    
    # YOUR CODE HERE
    # Just a random classifier for now
    # Replace this with your own approach
    
    #y = 1 + np.random.randint(3, size=X.shape[0])
    return y

