# Recycling Bin Detection
#### In this project, I will try to train the given dataset for mean and variance and use Navie Bayes Classifier and Gaussian Naive Bayes to build the model, predict which color it is and calculate the accuracy.


"test_roipoly.py" 
read the images and draw the area to extract the data



"bin_detector.py"
parameters -> different color mean, covariance, number
segmant_image function -> calculate the multivariate gaussian to decide whether is blue or not 
                                             and reture data
get_bounding_boxes function-> calculate the box of region and return x, y axis



"test_data_capture.py"
calculate the mean and covariance


"test_bin_detector.py"
import bin_detector.py to load the data in and find the blue recycling bin 


