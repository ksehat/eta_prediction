from src.data.data_processor import DataProcessor
from src.models.neural_network_trainer import NeuralNetworkTrainer
from src.models.random_forest_trainer import RandomForestTrainer
from src.models.gradient_boosting_trainer import GradientBoostingTrainer
from src.models.knn_trainer import KNNTrainer
from src.pipeline.train_pipeline import ModelPipeline

# Define the columns
feature_cols = ['accept_hour', 'accept_day_of_week', 'accept_month', 'is_weekend',
                'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon',
                'edd', 'provider_A', 'provider_B', 'provider_C', 'provider_D']
target_col = 'accurate_provider'

# Initialize the data processor
data_processor = DataProcessor(file_path='C://Users\kanan\Desktop\ML-Map\eta_prediction\data//raw//rides_data.pq')

# Neural Network Model Pipeline
nn_trainer = NeuralNetworkTrainer(input_shape=len(feature_cols))
nn_pipeline = ModelPipeline(data_processor, nn_trainer)
nn_accuracy = nn_pipeline.run(feature_cols, target_col)

# Random Forest Model Pipeline
rf_trainer = RandomForestTrainer()
rf_pipeline = ModelPipeline(data_processor, rf_trainer)
rf_accuracy = rf_pipeline.run(feature_cols, target_col)

# knn Model Pipeline
knn_trainer = KNNTrainer()
knn_pipeline = ModelPipeline(data_processor, knn_trainer)
knn_accuracy = knn_pipeline.run(feature_cols, target_col)
