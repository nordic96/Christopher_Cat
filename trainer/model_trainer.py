import os
import tensorflow as tf
import glob
# number of epochs: how many times do you want
# to pass the same batch size to train
# total batch size = total train data size
CHECKPOINT_PATH = 'training/cp.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
MODEL_WEIGHT_FILENAME = 'christopher_model.h5'


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

    def load_model_weights(self):
        self.model = tf.keras.model.load_model(MODEL_WEIGHT_FILENAME)
        print('model successfully loaded!')
        self.model.summary()

    def train_model(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                         save_weights_only=True,
                                                         verbose=1)
        if len(glob.glob(CHECKPOINT_DIR)) != 0:
            latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            self.model.load_weights(latest)

        history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            validation_data=self.validation_gen,
            validation_steps=self.validation_steps,
            callbacks=[cp_callback]
        )
        self.model.save(MODEL_WEIGHT_FILENAME)
        print('model weights saved!')
        return history



