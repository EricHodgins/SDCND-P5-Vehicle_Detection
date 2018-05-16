## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The method for extracting HOG features can be found in the file `Feature_Extractor.py`.  THe method `get_hog_features` specifically.  I used the hog function from python's skimage.

I started by reading in all the `vehicle` and `non-vehicle` images.  You can find samples in the notebook.  I choose 1/5th of the images.  Originally I had tried them all, but thought maybe taking some out would reduce overfitting.  

The `Feature_Extractor` extracts features for binned color features, color histogram features and HOG features.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

You can see examples of this in the notebook.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled on these:

-color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
-orient = 15  # HOG orientations
-pix_per_cell = 8 # HOG pixels per cell
-cell_per_block = 2 # HOG cells per block
-hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
-spatial_size = (32, 32) # Spatial binning dimensions
-hist_bins = 32    # Number of histogram bins
-spatial_feat = True # Spatial features on or off
-hist_feat = True # Histogram features on or off
-hog_feat = True # HOG features on or off
-y_start_stop = [400, None] # Min and max in y to search in slide_window()

I tried RGB, YCrCb, HLS, YUV color spaces.  orient from 3 to 16.  pix_per_cell from 2 to 8. hog_channel 0, 'ALL'.  What I have above seemed to work pretty good.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the features extracted from the `Feature_Extractor.py` class.  The classifier can be found in the `Vehicle_Classfier.py` file.  The features used were binned color features, color histogram features and HOG features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The Sliding Window Search can be found in the file `Searcher.py`.  Initially I just choose a scale of 1.5 that's fed into the `find_cars` method in the `Searcher` class.  That worked okay.

For the video I used different scales.  The closer to the car you are the bigger the box should be.  And closer to the horizon the smaller the box should be.  So, I played with some numbers to try and mimick that.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on YUV, All HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Some images can be found in the notebook.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_output/bbox_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

An example can be found in the notebook.


THe video implementation takes into account 10 frames. Averages the heatmaps generated for all frames and then uses a threshold of 10 to try and filter out the false positives.



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

