
import cv2
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label

class Scene:
    """The scene class is used to provide feature subsampling on an image. 
    
    Rarams:
        image: The image that features will be extracted from
        window_size: The size of the window in units of 8 pixel by 8 pixel cells.
        color_space: An optional color space to transform the image to before extracting features.
        hog_color_channel: An array of channel indices that should be used when extracting hog features.
    """
    
    def __init__(self, image, window_size=[8,8], color_space='RGB', hog_color_channel=[0,1,2]):
        self.gridSize = 8
        self.feature_size = [8,8]
        self.windowSize = np.int32(window_size)
        self.colorSpace = color_space
        self.originalImage = image
        self._scale_factor = np.divide(self.windowSize, self.feature_size)
        self.image = self.__processImage(image, color_space)
        self.gridShape = (int(self.image.shape[0]/self.gridSize), int(self.image.shape[1]/self.gridSize))
        self.viewPort = tuple(slice(0, int(dim/self.gridSize)+1, None) for dim in self.image.shape[0:2])

        # hog setup
        self._hog = None
        self._orientation_bins = 9
        self._cells_per_block=2
        self._channel=hog_color_channel
        
        # color hist setup
        self._colorHist = None
        self._color_bins = 32
        self.ss0 = None
        self.ss1 = None
        
    def __processImage(self, image, color_space):
        """Scales the input image to maintain 8x8 cell feature boxes and tranforms the color space if needed. 
        The scaling is required as the size of the feature vector must remain constant between different 
        window sizes.
        
        Params:
            image: The image to transform.
            color_space: The color space to use.
        Returns:
            The modified image.
        """
        
        new_size = tuple(np.int32(np.divide(image.shape[0:2], self._scale_factor)))
        feature_image = cv2.resize(image, new_size[::-1])
        
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2YCrCb)
        
        return feature_image
        
    def __getitem__(self, arg):
        """Overloaded subscripting operator used to return a tuple of slice objects. This is
        usefull when used with the viewPort property 'scene.viewPort = scene[1:2,1:2,1]'
        
        """
        
        return arg
        
    def view(self):
        """The current view of the image.
        
        returns: An image that represents the current view of the overall image.
        """
        
        self.ss0 = slice(self.viewPort[0].start*self.gridSize, self.viewPort[0].stop*self.gridSize, None)
        self.ss1 = slice(self.viewPort[1].start*self.gridSize, self.viewPort[1].stop*self.gridSize, None)
        return self.image[self.ss0, self.ss1]
    
    @property
    def viewPort(self):
        """Setter for viewPort property.
        """
        return self._viewPort
    
    @viewPort.setter
    def viewPort(self, arg):
        """Getter for viewPort property. Raises an exception if the argument is not a 2-tuple of slice
        objects.
        """
        
        if type(arg) is not tuple or len(arg) != 2:
            raise ValueError("Argument must be a 2-tuple of slice objects.")
        self._viewPort = arg
    
    @property
    def HogFeatures(self):
        """The hog features are created lazily for the entire input image. Once the Hog features
        are created, the features for the current viewPort will be returned.
        """
        
        # create the hog features only if they have not yet been created
        if self._hog is None:
            self._hog = []
            
            # create hog features for all of the requested hog channels
            for channel in self._channel:
                hog_channel = self.image[:,:,channel]
                self._hog.append(hog(hog_channel, orientations=self._orientation_bins, pixels_per_cell=(self.gridSize, self.gridSize),
                           cells_per_block=(self._cells_per_block, self._cells_per_block), transform_sqrt=True, 
                           visualise=False, feature_vector=False))
        
        # the Hog features have a different shape than the grid so we need to fix them up here.
        maxHogDim0 = int(self.feature_size[0] - self._cells_per_block + 1)
        maxHogDim1 = int(self.feature_size[1] - self._cells_per_block + 1)
        maxStop0 = np.minimum(self.viewPort[0].stop, self.viewPort[0].start + maxHogDim0)
        maxStop1 = np.minimum(self.viewPort[1].stop, self.viewPort[1].start + maxHogDim1)
        hs0 = slice(self.viewPort[0].start, maxStop0 , self.viewPort[0].step)
        hs1 = slice(self.viewPort[1].start, maxStop1 , self.viewPort[1].step)
        
        # ravel the current view of the hog features in a 1d vector
        return np.hstack([hog[hs0, hs1,...] for hog in self._hog]).ravel()
    
    @property
    def ColorHistFeatures(self):
        """The color histogram features are created lazily. The first time they are requested, a map of
        the histogram in each grid cell is calculated and saved. After this point, calculating the histograms
        for a given viewPort is simply a matter of suming up the histograms from the contibuting cells.
        """
        
        view = self.view()
        hist0 = np.histogram(view[:,:,0], bins=self._color_bins, range=(0,256))
        hist1 = np.histogram(view[:,:,1], bins=self._color_bins, range=(0,256))
        hist2 = np.histogram(view[:,:,2], bins=self._color_bins, range=(0,256))
        return np.vstack((hist0[0], hist1[0], hist2[0])).ravel()
    
    @property
    def SpatialFeatures(self):
        """ Creates spatial features by downsampling the current view of the image to a 32x32 image and then
        raveling this to a 1d vector.

        """
        
        self.ss0 = slice(self.viewPort[0].start*self.gridSize, self.viewPort[0].stop*self.gridSize, None)
        self.ss1 = slice(self.viewPort[1].start*self.gridSize, self.viewPort[1].stop*self.gridSize, None)
        return cv2.resize(self.image[self.ss0, self.ss1], (32, 32)).ravel()
    
    def SearchWindows(self, i_start_stop, j_start_stop, ij_step_size, classifier):
        """Creates a sliding window between the provided i/j start/stop with the provided overlap
        and searches the image for vehicles using the provided classifer. All distances/lengths should be
        provided in units of gridSize and all coordinates are in ij order.
        
        Args:
            i_start_stop: A 2-tuple that specifies the i search range as a fraction of the image height.
            j_start_stop: A 2-tuple that specifies the j search range as a fraction of the image width.
            ij_step_size: A 2-tuple that specifies the step sizes as fractions of the windows size.
            classifier: The classifier to use when testing if a given cell for is a vehicle or not.
            
        """
        
        # Compute the span of the region to be searched in i/j
        h,w = self.image.shape[0:2]
        i_start_stop = (np.float32(i_start_stop)*h/self.gridSize).astype(np.int32)
        j_start_stop = (np.float32(j_start_stop)*w/self.gridSize).astype(np.int32)
        ij_span = np.int32([i_start_stop[1] - i_start_stop[0], j_start_stop[1] - j_start_stop[0]])

        # Compute the number of pixels per step in x/y
        ij_step = np.multiply(self.feature_size, ij_step_size)
        
        # compute the number of windows in x/y
        ij_num_windows = np.int32((ij_span - self.feature_size) / ij_step + 1)
        
        # build the windows
        all_windows = []
        hot_windows = []
        for i in range(ij_num_windows[0]):
            for j in range(ij_num_windows[1]):
                
                # the top-left and bottom-right points are constructed in ij order and added to the list
                tl = np.int32([i*ij_step[0] + i_start_stop[0], j*ij_step[1] + j_start_stop[0]])
                br = np.int32(tl + self.feature_size)
                window = (tuple(tl), tuple(br))
                all_windows.append(window)
                
                # set the viewport on the scene (self) and then calculate the features
                self.viewPort = self[tl[0]:br[0], tl[1]:br[1]]        
                spatial = self.SpatialFeatures
                colorHist = self.ColorHistFeatures
                hogFeatures = self.HogFeatures
                features = np.concatenate((spatial, colorHist, hogFeatures))  
                test_features = classifier.X_scaler.transform(features.reshape(1, -1))
                prediction = classifier.Predict(test_features)
        
                # if a car is predicted, the window goes in the hot_windows list
                if prediction == 1:
                    hot_windows.append(window)
         
        # return hot windows and all windows with associated scale factors
        return (self._scale_factor, hot_windows), (self._scale_factor, all_windows)
    
    def DrawWindows(self, window_sets):
        """Draws the provided window sets on the image.
        
        Params:
            window_sets: The windows sets to draw. Each set is a 2-tuple composed of a scale and a list of windows
            to draw at that scale. (scale, [windows])
        Returns:
            A copy of the original image with the windows sets drawn on it.
        """
        
        draw_img = np.copy(self.originalImage)
        for scale, windows in window_sets:
            scalei, scalej = scale[0]*self.gridSize, scale[1]*self.gridSize
            for window in windows:
                points = ((int(window[0][1]*scalej), int(window[0][0]*scalej)),(int(window[1][1]*scalei), int(window[1][0]*scalei)))
                cv2.rectangle(draw_img, points[0], points[1], (0,0,1.0), 6)
        return draw_img
    
    def ContributeHeat(self, heatmap, window_sets, threshold=0):
        """Uses the provided window sets to add heat to the provided heat map.
        
        Params:
            heatmap: The heatmap to add to.
            window_sets: The windows sets to add to. Each set is a 2-tuple composed of a scale and a list of windows
            to draw at that scale. (scale, [windows])
            threshold: An optional threshold to apply to the heatmap after adding contributions. 
        Returns:
            The heatmap with contributions from the provided window sets.
        """
        
        for scale, windows in window_sets:
            scalei, scalej = scale[0]*self.gridSize, scale[1]*self.gridSize
            for window in windows:
                points = ((int(window[0][0]*scalej), int(window[0][1]*scalej)),(int(window[1][0]*scalei), int(window[1][1]*scalei)))
                heatmap[points[0][0]:points[1][0], points[0][1]:points[1][1]] += 1
                
        np.clip(heatmap, 0, 255)
        heatmap[heatmap < threshold] = 0
        return heatmap
    
    def DrawHeatMap(self, window_sets, threshold=0):
        """Creates a heatmap with the provided window sets.
        
        Params:
            window_sets: The windows sets to add to. Each set is a 2-tuple composed of a scale and a list of windows
            to draw at that scale. (scale, [windows])
            threshold: An optional threshold to apply to the heatmap. 
        Returns:
            The heatmap with contributions from the provided window sets.
        """
        
        heatmap = np.zeros_like(self.originalImage[:,:,0]).astype(np.float32)
        return self.ContributeHeat(heatmap, window_sets, threshold)
    
    def GetLabeledRegions(self, window_sets=None, heatmap=None, threshold=0, visualize=False):
        """Gets labeled regions from the provided window sets or heatmap.
        
        Params:
            window_sets: The windows sets used to create a heatmap to convert to labeled regions. Each set is 
            a 2-tuple composed of a scale and a list of windows to draw at that scale. (scale, [windows]). This 
            parameter may be None if a heatmap is provided.
            heatmap: A heatmap to create the labeled regions from. This parameter may be None if windows_sets are
            provided. If both are set, this argument takes precedence.
            threshold: An optional threshold to apply to the heatmap before the labeled regions are calculated.
        Returns:
            A 2-tuple with the scale and detected regions.
        """
        
        # if the caller did not provide a heatmap then we create one from the window_sets 
        if heatmap is None:
            if window_sets is None:
                raise ValueError('A heatmap or window_sets must be provided.')
            heatmap = self.DrawHeatMap(window_sets, threshold)
            
        heatmap[heatmap < threshold] = 0
            
        # get the labeled image
        label_img, num_labels = label(heatmap)
        
        # find the top left (tl) and bottom right (br) points that contain the labeled regions
        detected_regions = []
        for region_label in range(1, num_labels+1):
            region_i, region_j = (label_img == region_label).nonzero()
            tl = (np.amin(region_i), np.amin(region_j))
            br = (np.amax(region_i), np.amax(region_j))
            detected_regions.append((tl, br))
        
        # the heat map is already at a scale that matches the original image
        retVal = ([1.0/self.gridSize, 1.0/self.gridSize], detected_regions)
        if visualize is True:
            retVal = (retVal, label_img)
            
        return retVal