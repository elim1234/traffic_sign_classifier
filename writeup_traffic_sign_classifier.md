# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_output_figs/training_set_historgram.png "Visualization"
[image4]: ./my_traffic_signs/children_crossing.png "Traffic Sign 1"
[image5]: ./my_traffic_signs/speed_limit_30.png "Traffic Sign 2"
[image6]: ./my_traffic_signs/priority_road_sign.png "Traffic Sign 3"
[image7]: ./my_traffic_signs/right_of_way_at_intersection.png "Traffic Sign 4"
[image8]: ./my_traffic_signs/do_not_enter.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and my project code is on the project workspace: CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the native python len(), and numpy shape and unique to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of instances of different traffic sign types in the training set. The sign types are not evenly represented however, the sample set is sufficient to achieve the required classifier accuracy.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data so that they are consistent with initial zero mean unit variance of the weights. Contrary to the suggested (x-128)/128 which results in a number from 0 to 2, I normalized the data using x/255 - 0.5 which results in a number between -1 and 1.

Converting to greyscale was suggested but this seemed like a waste of color information that would be useful in differentiating signs. So I opted to retain the color information and train the network on 3-color images.

If necessary, I could augment the data to generate an even sampling of sign types. I could create additional samples by changing scale and lightness and by adding random noise to existing images. As it turns out, the original dataset was sufficient to get very high classifier accuracy.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was the Lenet architecture which consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Fully connected		| output 120									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 84										|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 43 classes								|
| Softmax				| 												|

I included dropout between the fully-connected layers to add redundancy and robustness to the classifier and prevent overfitting.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a batch size of 128, a number of epochs of 100, a learning rate of 0.0004 and a dropout keep probability of 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.960
* test set accuracy of 0.942

I started with the Lenet architecture because that was the architecture presented in the lesson and because two layers of 5x5 convolution seemed suitable for capturing bold geometries of street signs, one layer for geometric primitives, a second layer for relative placement. It initially had poor accuracy probably due to overfitting and the unevenly distributed training set. Adding dropout to the fully-connected layers resulted in the final architecture that was able to achieve a high accuracy with some tuning.

I started with a learning rate of 0.001 and then reduced it until the result had a consistent downward trend. Then I increased the number of epochs until tbe validation accuracy tapered off. I set the dropout probability to keep the validation accuracy consistent with the training accuracy.

The consistency between the training accuracy, validation accuracy, and test accuracy shows that the result is not overfitted to the training or validation data. The high accuracy of the test results shows that the model is reliable.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be the most challenging to classify because the sign has the more complex icon. Images 1, 2, and 5 have the most complex backgrounds but they are clearly separable by color, which is why the implemented color classifier is probably more accurate than a grey scale classifier.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing 	| Children crossing								| 
| 30 km/h  				| 30 km/h 										|
| Priority road			| Priority road					 				|
| Right of way			| Right of way									|
| No entry				| No entry		      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.942.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a children crossing sign (probability of 0.87), and the image does contain a children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.870 				| Children crossing								| 
| 0.089  				| Dangerous curve to the right					|
| 0.009					| Speed limit (60km/h)		 					|
| 0.008					| Vehicles over 3.5 metric tons prohibited		|
| 0.005					| End of no passing								|

This was the least confident result among the 5 new images probably due to the relative complexity of the sign icon itself and relative lower count of test images of this class. This can be improved by augmenting the data set as described above.

For the remaining 4 images, the model is almost 100% confident in identifying the sign and is corrent in every instance.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| Speed limit (30km/h)							| 
| 0.000  				| Speed limit (20km/h)							|
| 0.000					| Speed limit (70km/h)		 					|
| 0.000					| Speed limit (50km/h)		 					|
| 0.000					| General caution								|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| Priority road									| 
| 0.000  				| Yield											|
| 0.000					| No passing for vehicles over 3.5 metric tons	|
| 0.000					| Ahead only		 							|
| 0.000					| Stop											|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| Right-of-way at the next intersection			| 
| 0.000  				| Beware of ice/snow							|
| 0.000					| Double curve		 							|
| 0.000					| Pedestrians		 							|
| 0.000					| Vehicles over 3.5 metric tons prohibited		|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| No entry										| 
| 0.000  				| Stop											|
| 0.000					| Go straight or left		 					|
| 0.000					| Speed limit (20km/h)		 					|
| 0.000					| Yield											|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


