
import time
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# A helper function to extract the training features from an image
def extractFeatures(images):
    
    features = []
    oneP = len(images)/100
    for i, image in enumerate(images):
        
        # report progress every 1000 images processed
        if i % 1000 == 0:
            print(i)
        
        # create the scene object and then extract the spatial, colorHist and hog features
        scene = Scene(mpimg.imread(image), color_space='YCrCb', hog_color_channel=[0,1,2])
        spatial = scene.SpatialFeatures
        colorHist = scene.ColorHistFeatures
        hogFeatures = scene.HogFeatures
        feature = np.concatenate((spatial, colorHist, hogFeatures))
        features.append(feature)
    
    return features

def trainClassifier():
    
    # Read in car and non-car images
    cars = glob.glob('./data/vehicles/*/*.png')
    notcars = glob.glob('./data/non-vehicles/*/*.png')

    # extract all of the features from the images
    print('extracting features...')
    car_features = extractFeatures(cars)
    notcar_features = extractFeatures(notcars)

    # Create feature vectors array stacks for X and y
    print('building X and y...')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # train a classifier and save it
    classifier = Classifier()
    classifier.Train(X, y, save_to='./YCrCb_AllHog_Classifier.pkl')