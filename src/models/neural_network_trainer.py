import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.model_trainer import ModelTrainer
from src.models.save_training_weights import SaveWeights


class NeuralNetworkTrainer(ModelTrainer):
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.mae)
        return model

    def train(self, X_train, y_train):
        main_dir = 'C://Users\kanan\Desktop\ML-Map\eta_prediction\logs\model_weights/'
        self.history = self.model.fit(X_train, y_train, epochs=10000, batch_size=512, validation_split=0.2,
                                 callbacks=[SaveWeights(main_dir + 'model1')])
        self.model.save(main_dir + 'model1.h5')
        self.plot_training_history()


    def evaluate(self, X_test, y_test):
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

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


    def plot_training_history(self):
        # Extract the accuracy and loss from the history object
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        # Create subplots with 2 plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot accuracy
        ax1.plot(acc, label='Training Accuracy')
        ax1.plot(val_acc, label='Validation Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Plot loss
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
