
from tensorflow.keras.models import Model

# number of epochs: how many times do you want
# to pass the same batch size to train
# total batch size = total train data size
class ModelTrainer:
    def __init__(self,
                 train_data_gen,
                 steps_per_epoch,
                 epochs,
                 validation_gen,
                 validation_steps,
                 model):
        self.train_data_gen = train_data_gen
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.validation_gen = validation_gen
        self.validation_steps = validation_steps
        self.model = model

    def train_model(self):
        history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            validation_data=self.validation_gen,
            validation_steps=self.validation_steps
        )
        return history



