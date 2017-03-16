
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.datasets import load_digits

class Classifier:
    """A class to help train and persist a classifier."""
    
    def __init__(self):
        self.accuracy = None
        self.classifier = None
        self.X_scaler = None
    
    def Train(self, X, y, test_split=0.2, save_to=None):
        """Trains the labeled data provided and checks accuracy against the test data.
        
        Args:
            X: The training data.
            y: The training labels.
            test_split: The fraction of test data to split off from the training data.
            save_to: If provides, the classifier will be saved to the provided location.
        """
        
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=test_split, random_state=rand_state)
        
        # use a linear SVC classifier
        self.classifier = LinearSVC()
        
        # measure how long it takes to train the classifier
        t=time.time()
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
            
        self.accuracy = round(self.classifier.score(X_test, y_test), 4)
        print('Test accuracy = ', self.accuracy)
        
        if save_to is not None:
            joblib.dump((self.classifier, self.X_scaler, self.accuracy), save_to, compress=9)
            print('Saved trianed classifier to ', save_to)
        
    def Predict(self, X):
        """Predicts if the given samples are vehicles or not.
        
        Args:
            X: The samples.
        """
        
        return self.classifier.predict(X)
    
    def Load(self, load_from):
        """Load the classifier from the provided file.
        
        Args:
            load_from: The file to load the classifier from.
        """
        
        self.classifier, self.X_scaler, self.accuracy = joblib.load(load_from)
        
        