'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2

def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values
  '''  
  n = len(next(os.walk(folder))[2]) # number of files
  X = np.empty([n, 3])
  i = 0
  
  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))
  
  for filename in os.listdir(folder):  
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1
    
    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()
      

  return X


if __name__ == '__main__':
  folder = 'data/training'
  X1 = read_pixels(folder+'/red', verbose = True)
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))


  # Calculate the mean in R G B in real red, green, blue pics
  RR_mean=(sum(X1)/len(X1))[0]
  RG_mean=(sum(X1)/len(X1))[1]
  RB_mean=(sum(X1)/len(X1))[2]
  
  GR_mean=(sum(X2)/len(X2))[0]
  GG_mean=(sum(X2)/len(X2))[1]
  GB_mean=(sum(X2)/len(X2))[2]
  
  BR_mean=(sum(X3)/len(X3))[0]
  BG_mean=(sum(X3)/len(X3))[1]
  BB_mean=(sum(X3)/len(X3))[2]
  
  # Calculate the variance in R G B in real red, green, blue pics
  sigma_square_RR=0.0
  sigma_square_RG=0.0
  sigma_square_RB=0.0
  for i in X1:
    sigma_square_RR+=(i[0]-RR_mean)**2
    sigma_square_RG+=(i[1]-RG_mean)**2
    sigma_square_RB+=(i[2]-RB_mean)**2

  sigma_square_RR/=(len(X1)-1)
  sigma_square_RG/=(len(X1)-1)
  sigma_square_RB/=(len(X1)-1)

  print(sigma_square_RR)
  print(sigma_square_RG)
  print(sigma_square_RB)
  

  sigma_square_GR=0.0
  sigma_square_GG=0.0
  sigma_square_GB=0.0
  
  for i in X2:
      sigma_square_GR+=(i[0]-GR_mean)**2
      sigma_square_GG+=(i[1]-GG_mean)**2
      sigma_square_GB+=(i[2]-GB_mean)**2

  sigma_square_GR/=(len(X2)-1)
  sigma_square_GG/=(len(X2)-1)
  sigma_square_GB/=(len(X2)-1)

  print(sigma_square_GR)
  print(sigma_square_GG)
  print(sigma_square_GB)


  sigma_square_BR=0.0
  sigma_square_BG=0.0
  sigma_square_BB=0.0
  
  for i in X3:
      sigma_square_BR+=(i[0]-BR_mean)**2
      sigma_square_BG+=(i[1]-BG_mean)**2
      sigma_square_BB+=(i[2]-BB_mean)**2

  sigma_square_BR/=(len(X3)-1)
  sigma_square_BG/=(len(X3)-1)
  sigma_square_BB/=(len(X3)-1)

  print(sigma_square_BR)
  print(sigma_square_BG)
  print(sigma_square_BB)





