
from moviepy.editor import VideoFileClip

classifier = Classifier()
classifier.Load('./basic_classifier.pkl')
detector = VehicleDetector(classifier, memory=20, threshold=60, color_space='YCrCb', hog_color_channel=[0,1,2])

projectVideo = './udacity/project_video.mp4'
projectVideoOut = './udacity/project_video_out9.mp4'
clip1 = VideoFileClip(projectVideo)
out_clip = clip1.fl_image(detector.ProcessFrame)
%time out_clip.write_videofile(projectVideoOut, audio=False)