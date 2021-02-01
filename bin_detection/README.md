# Recycling Bin Detection
#### In this project, I will try to train the given dataset for mean and variance and use Navie Bayes Classifier and Gaussian Naive Bayes to build the model, predict which color it is and calculate the accuracy.


"generate_rgb_data.py" 
read_pixels function -> reads 3D pixel value from each image and return nx3 matrix containing    
                                      the piexl values.
main function -> read the 3D pixel value and calculate the mean and variance in RGB value in the 
                           real color in red, green, blue

"pixel_classifier.py"
PixelClassifer  parameters ->  the number of Red, Green, Blue, Total pics
                                                mean, variance in the rgb value of red, green, blue color 
                                                pi=3.1415
classify function -> using Navie Bayes Classifier and Gaussian Naive Bayes to predict the data is 
                               red, green, or blue. and return integer in 1:red, 2:green, 3:blue.
            


"test_pixel_classifier.py"
main function-> import the read_pixels and PixelClassifier function in and test the new data. Then 
                          print the precision


