# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**


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
* wideo.mp4 that shows the car driving around 3 laps of track 1

After carefully following the project guidelines, I started implementing the project in an ipython notebook. I called this "P3.ipynb". I do some data exploration and output some sample images as well.
My intention in this notebook was to keep things simple and start building the model, the way we did in the project videos.

#### 2. Driving car in Autonomous Mode

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Model code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In my case, all i had to do to develop this file was to copy over the various code cells from the ipython notebook
```sh
P3_generator.ipynb
```

The main difference between the 2 notebooks is that one has a generator and the other does not.

### Data Exploration and Initial Setup

#### 1. Data Exploration

I used the the notebook below to do some initial data exploration
```sh
P3.ipynb
```

In the above notebook, i gauged the histogram of the various steering angles.


#### 2. Reading in images

I used cv2.imread to read in my images. In my model, I read in the center, left and right images.
I also flip the center image and thereby increase the number of images in the data set. Thus essentially I end up having images that is equal 4 times the number of data points. 

I also made a mistake in this section that I realized later on in the project. The images are read using BGR format but i need the images to be in RGB format. I converted all the images i read into RGB format. The model behaved in a very weird way without this update; there were occassions when the car just went off course when it should not have done so. I should have read the project guidelines carefully. There was a warning about this.

In the notebook
```sh
P3.ipynb
```

I read in images, flip images and store them all in memory. I did this for initial prototyping work.

In the notebook
```sh
P3_generator.ipynb
```

I actually have a generator.

#### 3. Splitting the Data

I split the data into a training and validation set. This can be clearly seen in both my ipython notebooks. This is handled a little differently based on which notebook is being looked at.

The final model which is based of the notebook "P3 generator" essentially splits the lines of the original csv file into training and validation samples.

#### 4. Generator

As explained in the project videos, i have setup a generator. This is way more efficient even on an Amazon EC2 instance. I used the template provided in the proejct and made modifications to the number of images read. I had to do this since i was reading in center, left and right images. I also flip the center image. I then call this generator for both the training and validation samples.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was to come to a final solution after following the individual steps mentioned in the project guideline videos. I started by reading the center images from the udacity provided data. I slowly built on this by cropping images and then implementing the NVIDIA model. The NVIDIA model is really simple to implement especially with Keras. The activation function I used was "RELU".

After this, to ensure that my environment was setup fine, i built the model on Amazon EC2 and transferred the model to my Windows machine using WINSCP to start gauging how well my model worked in the autonomous mode.

To my surprise, the model worked reasonably well though it did go off the track. I then proceeded to add the left and right images as well. The important thing to keep in mind is that when adding left and right images, corresponding values should also be added for the angle measurements. If we don't do this we will get an error since there will be a mismatch in the total number of images and angle measurements. I also add flipped images to my set of images. The flipped image was essentially flipping the center image. The angle measurement for the corresponding flipped image was the negative of the angle measurement of the original center image.

Using fit generator i generated my model. When i ran the model and checked the simulator on autonomous mode, to my surprise i was amazed to see that the car drove very well. There after, i had to only do minor tweaks to my model. I reduced the number of epochs to two. I also added a dropout layer between the first and second fully connected layer to reduce overfitting.

Perhaps the most important adjustment i made in my model was to adjust the angle measurements for the left and right images. As mentioned previously we have a left, center and right image but one value for steering measurement.  For the left angle measurement I add a factor of 0.0085. For the right angle measurement I subtract a factor of 0.0085. I experimented with various values and decided on this after a lot of trials.


At the end of all these updates, the car was able to driver around track 1 without going off. This can be seen in the video below which shows the car being driven for three laps.
```sh
video.mp4
```


#### 2. Model parameter tuning

The model used an adam optimizer and i did not make any updates.
When i made updates to the adam optimizer parameters, my model did not improve much so i stuck with the default

#### 3. Training data

I used the udacity provided training data. As can be seen from the submission video, I could easily drive around track 1 for three laps without any issues. I did generate my own data; however I did not need them atleast to test on Track 1. I suspect I will definitely need to use my own training data and enhance my model using various augmentation techniques to drive around the more challenging track. I intend to try this soon at a later date.

#### 4. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

As can be seen below after normalizing and cropping the images, I implemented the NVIDIA model. I did not tinker with this model much. All I add was to add a dropout layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image							|
| Normalization			| Divide by 255 and subtract 0.5				|
| Cropping				| Remove 70 pixels from top and 25 from bottom	|
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


#### 5. Final Thoughts

I have been keeping things simple in all my projects as I ramp up on this rather exciting journey. I follow the guidelines and the various explanation videos to setup the various code templates and realize that doing this helps a lot in making a lot of progress. I re-use various code templates provided in the guidelines as well. This can be seen in my ipython notebooks. One thing that gets lost  are the various incremental additions I made to the initial model. In the near future, I have to enhance my model using advanced image augmentation techniques to make my model better. As mentioned previously, I have a hunch that I will need to do this for running the car in the more challenging track.
