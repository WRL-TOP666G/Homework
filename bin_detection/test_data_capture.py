'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
from scipy.stats import multivariate_normal

#from generate_rgb_data import read_pixels
#from pixel_classifier import PixelClassifier

if __name__ == '__main__':
  # test the classifier
  
  folder = 'data_capture/blue_bin'
  
  n=0
  x_r=[]
  x_g=[]
  x_b=[]
  X=[]
  for filename in os.listdir(folder):
    if filename.split('.')[-1]!='npy':
      continue

    tmp=np.load(os.path.join(folder,filename),allow_pickle=True)
    n+=len(tmp)

    #n+=len(tmp)
    for i in tmp:
      X.append(i)
      x_r.append(i[0])
      x_g.append(i[1])
      x_b.append(i[2])
  print(tmp)
  #gaussian = SimpleGaussian(X)
  #print(gaussian.predict(X))

  mean=np.mean(X, axis=0)
  cov=np.cov(np.transpose(X))
  #print(multivariate_normal.pdf(X, mean, cov))
  #R_mean=(sum(X)/len(X))[0]
  #G_mean=(sum(X)/len(X))[1]
  #B_mean=(sum(X)/len(X))[2]



  #print(mean)
  #print(cov)


  '''
  sigma_square_R = 0.0
  sigma_square_G = 0.0
  sigma_square_B = 0.0
  for i in X:
    sigma_square_R += (i[0] - R_mean) ** 2
    sigma_square_G += (i[1] - G_mean) ** 2
    sigma_square_B += (i[2] - B_mean) ** 2

  sigma_square_R /= (len(X) - 1)
  sigma_square_G /= (len(X) - 1)
  sigma_square_B /= (len(X) - 1)


  print("Mean: ")
  print(R_mean)
  print(G_mean)
  print(B_mean)


  print("sigma: ")
  print(sigma_square_R)
  print(sigma_square_G)
  print(sigma_square_B)

  '''
  #print(len(X))
