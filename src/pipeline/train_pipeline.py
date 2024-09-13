class ModelPipeline:
    def __init__(self, data_processor, model_trainer):
        self.data_processor = data_processor
        self.model_trainer = model_trainer

    def run(self, feature_cols, target_col):
        # Data Preprocessing & feature engineering
        self.data_processor.load_data()
        self.data_processor.feature_engineering()
        self.data_processor.split_data(feature_cols, target_col)
        self.data_processor.scale_data(feature_cols)
        self.data_processor.cluster_data()
        # self.data_processor.knn_regressor()

        # Model Training
        self.model_trainer.train(self.data_processor.X_train, self.data_processor.y_train)

        # Model Evaluation
        self.model_trainer.evaluate(self.data_processor.X_test, self.data_processor.y_test)