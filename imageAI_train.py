from imageai.Prediction.Custom import ModelTraining
import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("Fruits")
model_trainer.trainModel(num_objects=95, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True)
