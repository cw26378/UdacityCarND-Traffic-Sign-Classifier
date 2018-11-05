#**Traffic Sign Recognition**

##Writeup Template

###

---

**Build a Traffic Sign Recognition Project using Convolutional Neural Network**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test-1.png "Traffic Sign 1"
[image5]: ./examples/test-2.png "Traffic Sign 2"
[image6]: ./examples/test-3.png "Traffic Sign 3"
[image7]: ./examples/test-4.png "Traffic Sign 4"
[image8]: ./examples/test-5.png "Traffic Sign 5"
[image9]: ./examples/test-6.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Please find the attached python notebook file and html as a result summary.


###Data Set Summary & Exploration

####1. The first step is to load the data and provide a basic summary of the data set. In the code, the analysis is done by using python, numpy and/or pandas methods.

I used the pandas library `pickle` to calculate summary statistics of the traffic signs data set.
import pickle

''' python
import pickle
training_file = "./train.p"
validation_file= "./valid.p"
testing_file = "./test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(X_train.shape, X_valid.shape, X_test.shape)

import numpy as np
n_classes = len(np.unique(y_train))
'''


* The shape of training set is (34799, 32, 32, 3), namely there are 34799 images of size 32*32 with RGB channels.
* The shape of the validation set is (4410, 32, 32, 3), and thus we have 4410 images in validation set.
* The shape of test set is (12630, 32, 32, 3), meaning 12630 images in testing set.
* The shape of a traffic sign image is 32*32*3.
* The number of unique classes/labels in the data set is `n_classes` = 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Histogram bar charts of training/validation/testing sets are showing how many training examples for each traffic sign under the label number from 0 to 42.

'''python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))

plot = fig.add_subplot(441)
plot.hist(y_train, bins = 42, normed = True)
plot.title.set_text('\n training set')
plot = fig.add_subplot(442)
plot.hist(y_valid, bins = 42, normed = True)
plot.title.set_text('\n validation set')
plot = fig.add_subplot(443)
plot.hist(y_test, bins = 42, normed = True)
plot.title.set_text('\n testing set')

plt.tight_layout()
fig.suptitle('Distribution of labels', fontsize=13)
fig = plt.gcf()
'''


![The ditribution of labels in data sets][./examples/histogram.png "Label distribution"]

Furthermore, a randomly picked training image is printed out together with the associated label number and the explanation. This just serves as a sanity check showing that the data is loaded and represented correctly.

'''python
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(2,2))
plt.imshow(image)

with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    read_list = list(reader)
sign_list = read_list[1 : len(read_list)-1]  

print("the random number = ", index)
print("the label number of this sign is:", sign_list[y_train[index]][0],"; and the designated sign is: ", sign_list[y_train[index]][1])
'''

For the specific example shown below, the output reads:

`the random number =  1200`
`the label number of this sign is: 36 ; and the designated sign is:  Go straight or right`

![Selected traffic sign image with label][./examples/random_image_example.png "Go straight or right"]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the most identifiable patterns of traffic signs are shapes and edges, and that grayscale images have 3 times smaller input size compared to RGB images.

Here is an example of a traffic sign image before and after grayscaling.

![An Example of image before and after grayscale][./examples/grayscale_exp-before.png] [./examples/grayscale_exp-after.png]

As a last step, I normalized the image data because the normalization will help to speed up the training of neural network and be better prepared for different input data.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        			             		|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 gray image   						            	|
| Convolution1 5x5*6  	| 1x1 stride, same padding, outputs 28x28x6 	  |
| RELU					        |	28x28x6				                    						|
| Max pooling	       	  | 2x2 stride, outputs 14x14x6             			|
| Convolution2 5x5x16	  | 1x1 stride, same paddling, output 10*10*16	  |
| RELU					        |	10x10x16				                     					|
| Max pooling	       	  | 2x2 stride, outputs 5x5x16              			|
| Fully connected1     	| flatten becomes 400, outputs 120 (dropout 50%)|
| Fully connected2		  | Outputs 84   					                				|
| Softmax				        | output = 43  									                |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters: BATCH_SIZE = 128, rate = 0.001, and I trained the model for 80 epochs. In order to reduce overfitting, I have used L2 regularization with lambda set as reg_lambda = 0.3. Then trained model is otained by calculating and minimizing the cross entropy type of loss function for the softmax.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.95
* test set accuracy of 0.93

An iterative approach was chosen:
The first model is based on LeNet without dropout or other regularization.The problem with this initial model is that well training accuracy converges to 1 very fast, the validation accuracy stopped improving at 0.7~0.8 after a few epochs. This is showing a high variance issue (overfitting), and I decided to add regularization and dropouts. My experience on this particular problem is that L2 regularization after tuning the parameter lambda worked well. And with the dropout added, the training accuracy increased more slowly, but the validation did not saturate after accuracy comes to 0.8, and eventually validation accuracy gets to 0.95.

Another issue is that the training can be pretty slow. So I tried to play with the convolutional filter size(from 3x3 to 5x5) and found that the result with slightly larger conv filter size speed up the training but did not hurt performance.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. The images (after resize and converted to gray scale) are shown in the jupyter notebook.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.
Here are the results of the prediction:

| Image			            |     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| Left turn      		    |    Left turn									                |
| Children crossing    	| Children crossing 							              |
| Speed limit (30km/h)	| Speed limit (30km/h)							            |
| No entry	       		  | No entry			                 		     				|
| Construction			    | Construction                    							|
| STOP Sign  			      | Priority road                   							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. Given that test set has also 93% accuracy this is reasonable. I also included the 6th images because I actually found that Stop sign image cannot be correctly identified by the model when selecting 5 images from download. This is to illustrate that I did not cherry picking the good 5 images from the web. Actually, I even tried two stop signs images but neither of the two stop signs are correctly labled. I checked the distribution histogram of the training examples. Clearly, at label = 14, the data size is small (dip shown in the histogram figure around 14). This may be related the failure at reading stop sign.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 51st cell of the Ipython notebook.

I would like to discuss three out of the six images tested.
For the first image, the model is moderately sure that this is left turn and the result is correct. The top five soft max probabilities were

| (Relative) Probability|     Prediction	       				     	|
|:---------------------:|:-----------------------------------:|
| 3199         			    | 34   		Turn left aheead			      |
| 1023    				      | 14 		Stop sign					     	      |
| 395					          | 13		Yield								          |
| 383	      		        | 35		Ahead only	 			          	|
| 322				            |  3    Speed limt 60 km/h     		    |

For the second image, the model is marginally gave the correct prediction that this is a children crossing sign. The top five soft max probabilities were
| (Relative) Probability|     Prediction	       				     	|
|:---------------------:|:-----------------------------------:|
| 3917         			    | 28   		Children crossing			      |
| 3905    				      | 29 		Bike Crossing				     	    |
| 3625					        | 30		Be aware of ice						    |
| 1769	      			    | 38		keep right	 			            |
| 733				            | 24    Road narrow on the right      |

Similar analysis can be run on the other images. Typically the 4th image is a moderately confident prediction, while the 3rd and 5th are relatively very confident since the top prediction has a much larger relative probability value than that of the rest of predictions.

For the last image image, we can see that the model really had a hard time telling the correct answer. Other than the problem that label = 14 does not have many training data, the fact that the Stop sign image is too small after resize may also deteriorate the performance of model. From the following prediction ranking we can see that Stop sign is not listed in the top five candidates, meaning that the model indeed failed at this image.

| (Relative) Probability|     Prediction	       				     	|
|:---------------------:|:---------------------------------------------:|
| 3318                	| 12   		Priority road			                    |     
| 1919    				      | 10No passing for vehicles over 3.5 metric tons|
| 1117					        | 11 Right-of-way at the next intersection		  |
| 1059	      			    | 40		Roundabout mandatory 		    	          |
| 888				            | 42End of no passing by vehicles over 3.5 metric tons  	 |
