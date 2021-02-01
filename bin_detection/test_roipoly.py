'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt

import matplotlib
import numpy as np
matplotlib.use('TkAgg')



if __name__ == '__main__':

  folder = 'data/training'
  for filename in os.listdir(folder):
    #if filename!='0014.jpg' and filename!='0015.jpg' and filename!='0019.jpg' and filename!='0029.jpg' and filename!='0033.jpg':
      #continue
    # read the first training image
    #filename = '0001.jpg'
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # display the image and use roipoly for labeling
    fig, ax = plt.subplots()
    ax.imshow(img)
    my_roi = RoiPoly(fig=fig, ax=ax, color='r')
    # get the image mask
    try:
      mask = my_roi.get_mask(img)
    except:
      continue

    # display the labeled region and the image mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])
    ax1.imshow(img)
    ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))

    ax2.imshow(mask)
    plt.show(block=True)


    n=0
    i=0

    for j in range(len(img)):
      for k in range(len(img[0])):
        if mask[j][k] == True:
          n+=1

    list = np.empty([n, 3])

    for j in range(len(img)):
      for k in range(len(img[0])):
        if mask[j][k]==True:
          #print(img[j][k].astype(np.float64)/255)
          list[i]=img[j][k].astype(np.float64)/255
          i+=1
    np.save(os.path.join('data_capture/blue', '{}'.format(filename)),list)

    np.save(os.path.join('data_capture/blue', '{}'.format('mask_'+filename)), mask)

    for i in list:
      print(i)
    #print(list)
    #a=np.load('data_capture/0059.jpg.npy',allow_pickle=True)
    #print(len(a))
    #for i in a:
      #print(i)
    '''
    print(ax1)
    print(ax2)
    print(my_roi.x)
    c=np.array((my_roi.x, my_roi.y))
    print(c)
    np.save(os.path.join('data_capture/blue_bin_axis', '{}'.format(filename)),c)
    '''
  print("Finish")
