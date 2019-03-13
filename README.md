# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./Training.png "Training"
[image2]: ./Validation.png "Validation"
[image3]: ./Testing.png "Testing"
[image4]: ./german_images/001.jpg "Go straight or right"
[image5]: ./german_images/002.jpg "No entry"
[image6]: ./german_images/003.jpg "Road work"
[image7]: ./german_images/004.jpg "No passing"
[image8]: ./german_images/005.jpg "Speed limit (30km/h)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! My code is contained here in the Udacity workspace and here is a link to my code in Github [project code](https://github.com/jodansweenman/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python library uncluding numpy to calculate the summary stats of the data set.

* The size of training set is: 34799 images
* The size of the validation set is: 4410 images
* The size of test set is: 12630 images
* The shape of a traffic sign image is: 32x32 pixels
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

To look at the datasets, I separately graphed the Training, Validation, and Testing to show the amount of images per class. I thought that looking at the data this way would help show my future training might be deficient. Here is an exploratory visualization of the datasets. It is comprised of three bar graphs starting with the training, then validation, and then finally the testing dataset.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

When I began with pre-processing the data, I converted the images to grayscale and then normalize them according to the same method as Sermanet and LeCun for their ConvNet classifier. This was done by taking each image and subracting 128 and then dividing by 128. This method worked well for me.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x16  	|
| Flatten				| outputs 400   								|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Droput				| probability .5								|
| Fully connected		| outputs number of classes						|
 
I adopted the techniques used in the preceding lesson on the LeNet 5, but added in more techniques of weight and biases into my layers after adapting a method that I found on an article on medium.com entitled, the LeNet 5 in 9 lines of code using Keras. I did not adopt the Keras model, but adapted some of the Python to my own LeNet and then adapted to looking at the TensorFlow API.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a very similar approach to the one implemented in the LeNet example. I used the softmax of the cross entropy of a hot encoding of the class labels. I then used the Adam Optimizor in order to minimize the amount of loss based on the average of the cross entropy softmax. I tained for 15 epochs, reducing the learning rate at the number went up. I tried training longer, but the sumber was well saturated before I reached higher epoch levels. The batch size was kept at 128. Utilizing the method that I found, I initialize the weights and boiases with a truncated normal distribution centered at zero.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .928
* validation set accuracy of .992
* test set accuracy of .908

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I originally chose the LeNet method that we were taught in the lesson, as I thought it was a good baseline starting point.
* What were some problems with the initial architecture?
There was simply not enough accuracy, especially with the pictures being in the RGB color space, the initial method was not sufficient.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Based off of the paper provided, I began by converting all of the images to grayscale and my accuracy was much greater. I also added in the weights and biases in order to fine tune the training as each layer is gone through.
* Which parameters were tuned? How were they adjusted and why?
I changed the learning rate to .002 and increased the standard deviation to play around and was able to get the accruacy higher.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I changed the model by taking out the fully connected layers near the end and adding in more depth to the convolution layers, which was based on the structure in the paper. Overall, this added bit more accuracy, but I didn't see much of a change. This drove me to change the structure back to something that more reflects a traditional LeNet, but with only two fully connected layers. I decided to add a dropout with a keep probability of .5. This seemed to work well over all.
Lastly, I tuned in the amount of epochs, I started with 10, but this was not sufficient, and I tried 43, just because that was the number of classes. In doing this, I found that the model became saturated well before it hit 43 epoch iterations, and I kept the number to 15, as it seemed to be a nice mix of resources combined with saturation.
I think that adding in the dropout made my model perform a lot better.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![](/german_images/001.jpg =250x250) ![](https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png =250x250) ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there is a watermark on the picture, and while it is high resolution, I am resizing the image to a 32x32 image. I also believe that the 4th image will be harder to classify as it is fairly low resolution and looks even blockier after resizing the image to 32x32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or right	| Go straight or right 							| 
| No entry     			| No entry 										|
| Road work 			| Road work										|
| No passing      		| No passing    				 				|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 90.8%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is almost completely sure that this is a "Go straight or right" sign (probability of 0.99996), and the image does in fact contain a "Go straight or right" sign. The top five soft max probabilities were 0.99996, 0.000036, 0.000006, 0.000001, and 0.0000002.

For the second image, the model is pretty much 100% sure that his is a "No Entry" sign, with a top prediction that rounds to 1.0. This proves to be true. The other 4 were 1.52e-09, 6.31e-12, 3.40e-13, and 1.17e-13.

For the third image, the model is pretty sure that the image is a "Road Work" sign, with a probability of 0.997, and the image contains a "Road Work" sign. The other 4 softmax probabilities are 0.0026, 0.0000089, 0.00000066, and 0.00000039.

For the fourth image, the model is fairly sure that the image is a "No Passing" sign, with a probability of 0.99266, and the image contains a "No Passing" sign. The other 4 softmax probabilities are 0.00399, 0.00172, 0.00091, 0.00034.

For the fifth and final image, the model is almost completely sure that his is a "Speed limit (30km/h)" road sign, with a probability of 0.99997, and the image recognition is correct, as it is a "Speed limit (30km/h)" sign. The other 4 softmax probabilities are 0.000029, 1.83e-13, 4.18e-14, and 1.95e-19.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99996      			| Go Straight or Right							| 
| 1.00000  				| No Entry 										|
| 0.99738				| Road Work										|
| 0.99266      			| No Passing					 				|
| 0.99997 			    | Speed limit (30km/h) 							|
