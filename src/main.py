from src.data.data_processor import DataProcessor
from src.models.neural_network_trainer import NeuralNetworkTrainer
from src.pipeline.train_evaluate_pipeline import ModelPipeline

# Define the columns
feature_cols = ['city_id', 'accept_hour', 'accept_day_of_week', 'is_weekend',
                'origin_lat', 'origin_lon',
                'destination_lat', 'destination_lon',
                'edd', 'provider_A', 'provider_B',
                'provider_C', 'provider_D']
target_col = 'ata'

# Initialize the data processor
data_processor = DataProcessor(file_path='C://Users\kanan\Desktop\ML-Map\eta_prediction\data//raw//rides_data.pq')

# Neural Network Model Pipeline
nn_trainer = NeuralNetworkTrainer(input_shape=len(feature_cols))
nn_pipeline = ModelPipeline(data_processor, nn_trainer)
nn_accuracy = nn_pipeline.run(feature_cols, target_col)
