import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.model_trainer import ModelTrainer


class GradientBoostingTrainer(ModelTrainer):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        return model

    def train(self, X_train, y_train):
        main_dir = 'C://Users\kanan\Desktop\ML-Map\eta_prediction//artifact/'
        self.model.fit(X_train, y_train)
        with open(main_dir + 'gradient_boosting_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)  # Convert probabilities to class predictions

        # Print precision, recall, F1-score
        print("\nClassification Report:\n",
              classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))

        # Generate and display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                    yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
