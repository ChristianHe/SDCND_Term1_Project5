# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Visualize the training data
* Extract the HOG features and Color Histogram features
* Build a SVC classifier to detect the vechicles
* Apply slide hog windows and predict the result in each windows
* Draw the heap diagram of prediction results
* Apply the threshold to remove the false positives
* Label the vehicles and draw onto the frame

[//]: # (Image References)
[image1]: ./output_images/datavisualization.png
[image2]: ./output_images/hogfeatures.png
[image3]: ./output_images/slidewindows.png
[image4]: ./output_images/boxes1.png
[image5]: ./output_images/boxes2.png
[image6]: ./output_images/boxes3.png
[image7]: ./output_images/boxes4.png
[image8]: ./output_images/boxes5.png
[image9]: ./output_images/boxes6.png
[image10]: ./output_images/heatmap1.png
[image11]: ./output_images/heatmap2.png
[image12]: ./output_images/heatmap3.png
[image13]: ./output_images/heatmap4.png
[image14]: ./output_images/heatmap5.png
[image15]: ./output_images/heatmap6.png
[video1]: ./project_video_output_v3.mp4

[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the forth and fifth code cells of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and It seems that the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` from the classroom gives me a relatively good result. Though the `GRAY` color space runs much faster, the `YCrCb` color space gives me more detections.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained `LinearSVC` with `GridSearchCV`, which searches over specified parameter values for an estimator and cross validate the training process. The code for this step is contained in the 8th ~ 10th code cells. First I use the `StandardScaler` to normalize the training data set and define the labels vertor. Then I grid search the `LinearSVC` model with parameters `C = 1` and `C = 10`. But the result indicates the C parameter does not have a significant impact. **The Best score is 92.4%**.

I have also trained the `svc` model the 'kernel' in 'rbf' and 'linear', the 'rbf' kernel performs better than the 'linear' kernel. But It takes almost 3 hours for the vehicle detection on the video with the 'rbf' model while the 'linear' kernel takes only 15 mins. 

kernel|C|Score
-|-|-
rbf|1|91.6%
rbf|10|91.8%
linear|1|90.2%
linear|10|90.2%

Also, I have found that the `svc` with `linear` kernel is not equivalent with the `LinearSVC`. In this case, the `LinearSVC` performs better. Here is [one of the answer](https://stackoverflow.com/questions/33843981/under-what-parameters-are-svc-and-linearsvc-in-scikit-learn-equivalent) why they produce different results.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the following window positions with 1 cell overlapping over the image.

window | ystart | ystop
-|-|-
64|400|528
80|400|576
196|400|576
128|400|656
192|400|656

And came up with this (no overlap here).

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 3-channel HOG features in the feature vector. Here are some example images. Frome the example images, we can see there's lots of false positive detections, like trees and shadows. They will be eliminated with the heatmap in the video frame.

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output_v3.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap and the bounding boxes of example images.

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my implementation, there are lots of false positive detection with sliding windows. One possible way is to use the Gray `HOG` features. But the Gray HOG sometimes will fail when detecting the white color vehicle. Minimize the number of sliding windows may get down the false positives, but sometimes in the video, the vehicle can not be detected. Maybe I should get into the frames where the vehicle is not detected to check where the problem is. 
Another enhancement may be the speed of frame processing. 
