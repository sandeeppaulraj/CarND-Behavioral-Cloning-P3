# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**



[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted

#### 1. Submission includes the folowing files

My project includes the following files:

* P3.ipynb where initial prototyping was performed
* P3_generator.ipynb which has the final model using fit generator
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

After carefully following the project guidelines, i started implementing the project in an ipython notebook. I called this "P3.ipynb". I do some data exploration and output some sample images as well.
My intention in this notebook was to keep things simple and start building the model, the way we did in the project videos.

#### 2. Driving car in Autonomous Mode

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Model code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Iy my case, all i had to do to develop this file was to copy over the various code cells from the ipython notebook
```sh
P3_generator.ipynb
```


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).



#### 2. Attempts to reduce overfitting in the model

As can be seen above after normailizing and cropping the images, i implemented the NVIDIA model.
I did not tinker with this model much. All i add was to add a dropout layer.

#### 3. Model parameter tuning

The model used an adam optimizer and i did not make any updates.
When i made updates to the adama optimizer paramaters, my model did not improve much so i stuck witht the default

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architectureconsisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image							|
| Normalization			|												|
| Cropping				|												|
| Convolution 5x5     	| 2x2 stride, outputs 31x158x24 				|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, outputs 14x77x36 					|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, outputs 5x37x48 					|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, outputs 3x35x64 					|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, outputs 1x33x64 					|
| RELU					|												|
| Flatten				| 2112 outputs									|
| Fully connected		| 2112 inputs 100 outputs						|
| Dropout				| 0.4 											|
| Fully connected		| 100  inputs  50 outputs						|
| Fully connected		| 50   inputs  10 outputs						|
| Fully connected		| 10   inputs  1  outputs						|
|						|												|
|						|												|

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
