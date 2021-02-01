'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import math
import cv2, os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from skimage.measure import label, regionprops

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		#Mean
		'''
		self.blue_mean
		self.green_mean
		self.grey_mean
		self.red_mean
		self.white_mean
		self.yello_mean

		#sigma
		self.blue_cov
		self.green_cov
		self.grey_cov
		self.red_cov
		self.white_cov
		self.yello_cov
		
		#length
		self.blue_length=4289082
		self.green_length=1301708
		self.grey_length=4892316
		self.red_length= 234180
		self.white_length=333581
		self.yello_length=39341
		self.total_length=self.blue_length+self.green_length+self.grey_length+self.red_length+self.white_length+self.yello_length
		self.nonblue_length=self.total_length-self.blue_length
		'''
		'''
		#Other function
		#Blue
		B=[]
		folder = 'data_capture/blue_tmp'
		for filename in os.listdir(folder):
			#if filename.spilt('_')[0] =='mask':
				#continue
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				B.append(i)
		self.blue_mean = np.mean(B, axis=0)
		self.blue_cov = np.cov(np.transpose(B))
		self.blue_length = len(B)
		#Green
		G=[]
		folder = 'data_capture/green'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				G.append(i)
		self.green_mean = np.mean(G, axis=0)
		self.green_cov = np.cov(np.transpose(G))
		self.green_length = len(G)
		#Grey
		Gr=[]
		folder = 'data_capture/grey'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				Gr.append(i)
		self.grey_mean = np.mean(Gr, axis=0)
		self.grey_cov = np.cov(np.transpose(Gr))
		self.grey_length = len(Gr)
		#Red
		R=[]
		folder = 'data_capture/red'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				R.append(i)
		self.red_mean = np.mean(R, axis=0)
		self.red_cov = np.cov(np.transpose(R))
		self.red_length = len(R)
		#white
		W=[]
		folder = 'data_capture/white'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				W.append(i)
		self.white_mean = np.mean(W, axis=0)
		self.white_cov = np.cov(np.transpose(W))
		self.white_length = len(W)
		#yello
		Y=[]
		folder = 'data_capture/yello'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			for i in tmp:
				Y.append(i)
		self.yello_mean = np.mean(Y, axis=0)
		self.yello_cov = np.cov(np.transpose(Y))
		self.yello_length = len(Y)


		#Non-Blue
		NB=[]
		#folder = 'data_capture/green'
		colors=['green', 'grey', 'red', 'white', 'yello']
		for color in colors:
			folder = 'data_capture/'
			folder+=color
			for filename in os.listdir(folder):
				if filename.split('.')[-1] != 'npy':
					continue
				tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
				for i in tmp:
					NB.append(i)

		self.nonblue_mean = np.mean(NB, axis=0)
		self.nonblue_cov = np.cov(np.transpose(NB))
		self.nonblue_length = len(NB)

		self.total_length=self.blue_length+self.nonblue_length
		'''
		#########

		bl=[]
		nonbl=[]
		folder_orig = "data/training"
		folder = 'data_capture/blue_mask'
		for filename in os.listdir(folder):
			if filename.split('.')[-1] != 'npy':
				continue
			tmp = np.load(os.path.join(folder, filename), allow_pickle=True)
			filename_orig=filename.split('_')[1].split('.')[0]+'.'+filename.split('_')[1].split('.')[1]
			if filename_orig in os.listdir(folder_orig):
				img = cv2.imread(os.path.join(folder_orig, filename_orig))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				for i in range(len(img)):
					for j in range(len(img[0])):
						if tmp[i][j] is np.bool_(True):
							bl.append(img[i][j])
						else:
							nonbl.append(img[i][j])
			else:
				continue
		print(len(bl))
		print(len(nonbl))
		print(len(bl[0]))
		print(len(nonbl[0]))
		#print(bl)
		self.bl_mean = np.mean(bl, axis=0)
		self.bl_cov = np.cov(np.transpose(bl))
		self.bl_length = len(bl)
		#print(bl)

		self.nonbl_mean = np.mean(nonbl, axis=0)
		self.nonbl_cov = np.cov(np.transpose(nonbl))
		self.nonbl_length = len(nonbl)


		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask_img=np.empty(shape=(len(img),len(img[0])))
		#img = img.astype(np.float64) / 255
		'''
		pblue=multivariate_normal.pdf(img, self.blue_mean, self.blue_cov)
		pgreen=multivariate_normal.pdf(img, self.green_mean, self.green_cov)
		pgrey=multivariate_normal.pdf(img, self.grey_mean, self.grey_cov)
		pred=multivariate_normal.pdf(img, self.red_mean, self.red_cov)
		pwhite=multivariate_normal.pdf(img, self.white_mean, self.white_cov)
		pyello=multivariate_normal.pdf(img, self.yello_mean, self.yello_cov)
		pnonblue=multivariate_normal.pdf(img, self.nonblue_mean, self.nonblue_cov)
		'''
		pbl=multivariate_normal.pdf(img, self.bl_mean, self.bl_cov)
		pnonbl = multivariate_normal.pdf(img, self.nonbl_mean, self.nonbl_cov)
		#print(pbl[0][0])
		#print(pnonbl[0][0])
		'''
		pblue*=(self.blue_length/self.total_length)
		pgreen*=(self.green_length/self.total_length)
		pgrey*=(self.grey_length/self.total_length)
		pred*=(self.red_length/self.total_length)
		pwhite*=(self.white_length/self.total_length)
		pyello*=(self.yello_length/self.total_length)
		pnonblue*=(self.nonblue_length/self.total_length)
		'''
		pbl = pbl * (self.bl_length/(self.bl_length+self.nonbl_length))
		pnonbl = pnonbl * (self.nonbl_length/(self.bl_length+self.nonbl_length))
		print(pbl[0][0])
		print(pnonbl[0][0])
		#print(type(pblue))
		pbl[0][0]*=10
		pnonbl[0][0] *= 10
		print(pbl[0][0])
		print(pnonbl[0][0])

		for i in range(len(img)):
			for j in range(len(img[0])):
				#if pblue[i][j]>pgrey[i][j] and pblue[i][j]>pwhite[i][j]:
				#if pblue[i][j]>pgreen[i][j] and pblue[i][j]>pgrey[i][j] and pblue[i][j]>pred[i][j] and pblue[i][j]>pwhite[i][j] and pblue[i][j]>pyello[i][j]:
					#mask_img[i][j]=1
				#else:
					#mask_img[i][j]=0
				if pbl[i][j]>pnonbl[i][j]:
					mask_img[i][j]=1
				else:
					mask_img[i][j]=0



		'''
		for i in range(len(img)):
			for j in range(len(img[0])):
				img[i][j] = img[i][j].astype(np.float64) / 255
				#blue
				blue=[]
				p_blue=1.0
				for k in range(3):
					blue.append(math.exp(-((img[i][j][k] - self.blue_mean[k]) ** 2) / (2 * self.blue_sigma[k])) / math.sqrt((2 * self.pi * self.blue_sigma[k])) )
					p_blue*=blue[k]
				p_blue*=(self.blue_length/self.total_length)

				#green
				green = []
				p_green = 1.0
				for k in range(3):
					green.append(math.exp(-((img[i][j][k] - self.green_mean[k]) ** 2) / (2 * self.green_sigma[k])) / math.sqrt((2 * self.pi * self.green_sigma[k])))
					p_green *= green[k]
				p_green *= (self.green_length / self.total_length)

				# grey
				grey = []
				p_grey = 1.0
				for k in range(3):
					grey.append(math.exp(-((img[i][j][k] - self.grey_mean[k]) ** 2) / (2 * self.grey_sigma[k])) / math.sqrt((2 * self.pi * self.grey_sigma[k])))
					p_grey *= grey[k]
				p_grey *= (self.grey_length / self.total_length)

				#red
				red = []
				p_red = 1.0
				for k in range(3):
					red.append(math.exp(-((img[i][j][k] - self.red_mean[k]) ** 2) / (2 * self.red_sigma[k])) / math.sqrt((2 * self.pi * self.red_sigma[k])))
					p_red *= red[k]
				p_red *= (self.red_length / self.total_length)

				#white
				white = []
				p_white = 1.0
				for k in range(3):
					white.append(math.exp(-((img[i][j][k] - self.white_mean[k]) ** 2) / (2 * self.white_sigma[k])) / math.sqrt((2 * self.pi * self.white_sigma[k])))
					p_white *= white[k]
				p_white *= (self.white_length / self.total_length)

				#yello
				yello = []
				p_yello = 1.0
				for k in range(3):
					yello.append(math.exp(-((img[i][j][k] - self.yello_mean[k]) ** 2) / (2 * self.yello_sigma[k])) / math.sqrt((2 * self.pi * self.yello_sigma[k])))
					p_yello *= yello[k]
				p_yello *= (self.yello_length / self.total_length)


				if p_blue>p_green and p_blue>p_grey and p_blue>p_red and p_blue>p_white and p_blue>p_yello:
					mask_img[i][j]=1
				else:
					mask_img[i][j]=0

		#np.savetxt('img.txt',img)
		'''

		print("Show picture")
		plt.imshow(mask_img)
		plt.show()

		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		# YOUR CODE HERE
		x = np.sort(np.random.randint(img.shape[0],size=2)).tolist()
		y = np.sort(np.random.randint(img.shape[1],size=2)).tolist()
		boxes = [[x[0],y[0],x[1],y[1]]]
		boxes = [[182, 101, 313, 295]]
		return boxes


