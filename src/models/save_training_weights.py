from keras.callbacks import Callback
import os
import keyboard

class SaveWeights(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.start_save = True
        os.makedirs(log_dir, exist_ok=True)
        self.delete_files_in_dir()

    def on_epoch_end(self, epoch, logs=None):
        if keyboard.is_pressed('q'):
            print("Stopping training")
            self.model.stop_training = True
        if keyboard.is_pressed('s'):
            print("\nSaving weights is started.\n")
            self.start_save = True
            self.delete_files_in_dir()
        if keyboard.is_pressed('t'):
            self.start_save = False
            print("\nSaving weights is stopped.\n")
        if self.start_save:
            # Save the weights at the end of each epoch
            weight_path = os.path.join(self.log_dir, f'weights_epoch{epoch}.h5')
            self.model.save_weights(weight_path)
            # Save the logs (loss and validation loss)
            with open(os.path.join(self.log_dir, 'saved_weights_logs.txt'), 'a') as f:
                f.write(f'Epoch: {epoch}, Loss: {logs["loss"]}, Val Loss: {logs["val_loss"]}\n')

    def delete_files_in_dir(self):
        # Check if the directory is not empty
        if os.listdir(self.log_dir):
            # Delete all files in the directory
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                os.remove(file_path)