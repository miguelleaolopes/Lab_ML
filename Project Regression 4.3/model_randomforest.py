from initialization import *
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

class RansomForestModel:

    def __init__(self):
        self.model = model_selection.RandomForestClassifier()

    def fit(self):
        self.model.fit(x_train,y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(x_test)

    def calculate_score(self):
        serfc_cv_score = cross_val_score(self.model, x_test, y_test, cv=10, scoring='f1')



    




