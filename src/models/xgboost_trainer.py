from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from src.models.model_trainer import ModelTrainer


class XGBoostTrainer(ModelTrainer):
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost Test Accuracy: {accuracy:.4f}")
        return accuracy
