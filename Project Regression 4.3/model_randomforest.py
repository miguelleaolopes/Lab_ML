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
        self.score = cross_val_score(self.model, x_test, y_test, cv=10, scoring='f1')

    def print_scores(self):
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        print('\n')
        print("=== All AUC Scores ===")
        print(self.score)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", self.score.mean())



mod = RansomForestModel
mod.fit()
mod.predict()
mod.calculate_score()
mod.print_scores()




