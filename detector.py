
from scene import Scene
from classifier import Classifier
from collections import deque
import numpy as np

class VehicleDetector:
    """A class to help maintain state between video frames.
    
    Args:
        classifier: The classifier to use to process video frames.
        heatmap_size: The size of the heatmap to be used.
        memory: The number of previous frames to take into account for thresholding.
        color_space: The color_space to use when classifying frames.
        hog_color_channel: The color channels to use for hog features.
    """
    
    def __init__(self, classifier, heatmap_size=(720, 1280), memory=3, threshold=2, color_space='YCrCb', hog_color_channel=[0,1,2]):
        self.classifier = classifier
        self.heatmapSize = heatmap_size
        self.memory = memory
        self.threshold = threshold
        self.colorSpace = color_space
        self.hogColorChannel = hog_color_channel
        self.heatmapQueue = self._initializeQueue(self.memory)
        self.previousRegions = None
        self.hotWindows = None
        
    def _initializeQueue(self, memory):
        """Initializes the deque used to hold previous heatmaps.
        
        Args:
            memory: The number of previous frames to hold in the queue.
        """
        
        queue = deque([])
        for _ in range(memory):
            queue.appendleft(np.zeros(shape=self.heatmapSize, dtype=np.float32))
        return queue
    
    def ProcessFrame(self, frame):
        """Searches for vehicles in the provided video frame and draws their bounding boxes.
        
        Args:
            frame: The video frame to process.
        """
        
        # the frame needs to be converted to the format that was used during training: float [0.0, 1.0]
        frame = frame.astype(np.float32) / 255.0
        self.hotWindows = []
        
        window_sizes = ((6,6), (10,10), (12,12))
        overlaps = ((0.25,0.25),(0.125,0.125), (0.125,0.125))
        y_ranges = ((0.55, 0.7), (0.55, 0.8), (0.5, 0.95)) 
        x_ranges = ((0.5, 1.0), (0.5, 1.0), (0.4, 1.0))

        # loop through all window sizes and search the image
        for window, overlap, y_range, x_range in zip(window_sizes, overlaps, y_ranges, x_ranges):
            scene = Scene(frame, window_size=window, color_space=self.colorSpace, hog_color_channel=self.hogColorChannel)
            h, a = scene.SearchWindows(y_range, x_range, overlap, self.classifier)
            self.hotWindows.append(h)
        
        # get the heatmap for the current frame and add it to the historical frames
        heatmap = scene.DrawHeatMap(self.hotWindows)
        self.heatmapQueue.appendleft(np.zeros(shape=self.heatmapSize, dtype=np.float32))
        for hmap in self.heatmapQueue:
            hmap += heatmap
        
        # finally, get the detected regions from the multi-frame heatmap
        detected_regions = scene.GetLabeledRegions(heatmap=self.heatmapQueue.pop(), threshold=self.threshold)
        detected_regions = self._sanitizeRegions(detected_regions)
        
        # extract the windows from the detected_regions and low-pass filter them WRT time
        if self.previousRegions is not None and len(self.previousRegions) == len(detected_regions[1]):
            try:
                filtered_regions = self._filterRegions(detected_regions)
                out_frame = scene.DrawWindows([filtered_regions])
                self.previousRegions = filtered_regions[1]
            except ValueError:
                out_frame = scene.DrawWindows([detected_regions])
                self.previousRegions = detected_regions[1]
        else:
            # there is a mis-match in number of detected regions so just ignore the previous regions
            out_frame = scene.DrawWindows([detected_regions])
            self.previousRegions = detected_regions[1]
          
        # format the return image for video stream
        return np.uint8(out_frame * 255)
    
    def GetHeatMap(self):
        return self.heatmapQueue[-1]
    
    def DrawPositiveDetections(self, frame):
        scene = Scene(frame)
        return scene.DrawWindows(self.hotWindows)
    
    def GetLabelsMap(self):
        scene = Scene(frame)
    
    def _sanitizeRegions(self, regions):
        sanitary = []
        scale, windows = regions
        for window in windows:
            w,h = window[1][1] - window[0][1], window[1][0] - window[0][0] 
            if w > 64 and h > 64:
                sanitary.append(window)
        return (scale, sanitary)
    
    def _filterRegions(self, detected_regions):
        filtered_regions = []
        scale, regions = detected_regions

        # iterate through the detected regions and find the matching region from the previous frame.
        for i, region in enumerate(regions):
            previous_region = self._matchRegion(region)
            filtered_regions.append(self._filterRegion(region, previous_region, 0.3))

        return (scale, filtered_regions)
                   
    def _filterRegion(self, new, old, new_factor):
        
        out = []
        new_factor, old_factor = new_factor, 1 - new_factor
        for n, o in zip(new, old):
            ret = (new_factor*n[0] + old_factor*o[0], new_factor*n[1] + old_factor*o[1])
            out.append(ret)
        return tuple(out)
    
    def _matchRegion(self, region):
        
        # find the center point of the new region
        cpi, cpj = (region[1][0] - region[0][0]) / 2, (region[1][1] - region[0][1]) / 2 
        
        # iterate through the previous regions and find the first one that contains the new cp
        for pr in self.previousRegions:
            tl, br = pr
            if tl[0] < cpi and tl[1] < cpj and br[0] > cpi and br[j] > cpj:
                return pr
        
        raise ValueError('Could not find a matching region is the previous regions')