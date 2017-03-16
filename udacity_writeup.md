**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/carnotcar.jpg
[image2]: ./output/hogFeatures.jpg
[image3]: ./output/detectedWindowsAllScales.jpg
[image5]: ./output/cumheatmap.jpg
[image6]: ./output/labeledRegions.jpg
[image7]: ./output/boundingBoxes.jpg
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. 

The code for the HOG feature extraction is located in the `HogFeatures` property getter of the `Scene` class in scene.py. In order to improve performance and speed up the pipeline, the HOG features are calculated for the entire image the first time they are requested and are then subsampled for the viewport being requested. The `Scene` class is used in the training and prediction phases of this project. 

During the training phase I read all of the `vehicles` and `non-vehicle` images into lists and process each individuall. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car-not-car][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG features][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

In order to find a good combination of hog parameters I removed all other features and trained the classifier with HOG features alone. This allowed me to try several combinations of HOG parameters and compare the resulting classifier accuracy.

In the end I found that the best results came from the following settings:
- (8, 8) pixels per cell
- (2, 2) cells per block
- 9 orientations
- All (3) HOG channels

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code that trains the SVM is located in the `extractFeatures()` function in train.py as well the `Classifier` class in classifier.py. Once the `vehicle` and `not-vehicle` images were read into lists, they we passed to the `extractFeatures()` function which performs the following operation on each:
- Create a Scene instance with the image.
- Extract the spatial features.
- Extract the color histogram features.
- Extract the HOG features.
- Concatenate all of the features and add them to a the array of feature vectors.

Once the features have been extracted from the images, matching labels of 0 (not-car) or 1 (car) are created and both lists are passed to the `Train` function of the `Classifier` instance. The `Classifier` normalizes the features with `sklearn.StandardScaler()` and then separates them into train and test sets with an 80/20 split. 

The Linear SVC instance is then trained on the data and persisted to disk for later use.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The code that implements the sliding window search is in the `SearchWindows()` function of the `Scene` class in scene.py. This function takes arguments that specify the search range and step size in i and j and builds a grid of windows to search. 
In order to determine effective window and step sizes, I extracted several frames from the project video with vehicles at varying positions, sizes and distances and experimented to see what worked best. I found that very small windows (48x48 px) with small overlaps of 50%-75% worked well for detecting vehicles in the distance, however these small windows slow down the search quite a bit so they are limited to a narrow band in the center of the image. For vehicles that are closer, window sizes of 64px - 96px work well. These larger windows are less computationally expensive so they are allowed to span larger search ranges.

In the end I used 3 window sizes with the following configurations:

| Size (px) | Range Y (percent of image) | Overlap X/Y (percent of overlap) |
--- | --- | ---
| 48 x 48 | 0.55 - 0.65 | 75 / 75 |
| 80 x 80 | 0.6 - 0.8   | 75 / 75 |
| 96 x 96| 0.5 - 0.95 | 75 / 75 |

![3 scales of sliding windows][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The image above shows examples of the pipeline working. Ultimately I searched on 3 scales (images on left) using YCrCb color space, 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. This provided very good positive detection with minimal false detection.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=3cPSfGndScE&feature=youtu.be)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As each frame is processed in the `ProcessFrame` method of the `VehicleDetector` class in detector.py, the positions of positive detections are used to create a heatmap that is accumulated over a span of several frames. This multi-frame accumulation creates a filter that only passes objects with lifespans longer than the number of frames considered. The cumulative heatmap is then thresholded and passed to `scipy.ndimage.measurements.label()` to group and identify the 'hot' regions. I make the assumption that each labeled region corresponds to a vehicle and create a bounding box for each. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![Cumulative heatmap][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![Detected regions][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Bounding boxes][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue that I had during the implementation this project was finding the right amount of filtering to minimize the false detections while still retaining the true bounding boxes. One method that I found to help with this issue was to use a relatively low threshold but apply it over a longer video span. For example, I found that using a threshold of 6 over 4 frames provided better results than a threshold of 4 over 2 frames. The first case has a lower average threshold per frame which increases the signal from the vehicles while decreasing the signal from the transient detections with short lifetimes. 
 
Another issue is the speed of the overall pipeline. The final configuration averaged about 2 seconds per frame in the processing step. This is very slow and is nowhere near where it would need to be for a realtime production solution. Improving this would most likely start by finding a way to use fewer windows and/or scales during the search process and to do this the classifier would need improvement. I plan to continue work on this project and train the classifier with more data to improve the performance. 


