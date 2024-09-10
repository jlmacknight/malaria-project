import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from model import CNNHyperModel
from utils import early_stopping_callback, preprocess

# Load data
(train_data, val_data, test_data), info = tfds.load(
    "malaria",
    split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
    as_supervised=True,
    with_info=True,
)
train_data = (
    train_data.map(preprocess)
    .cache()
    .shuffle(info.splits["train"].num_examples)
    .batch(16)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
val_data = val_data.map(preprocess).cache().batch(16).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preprocess).cache().batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Define the Hyperband tuner
tuner = kt.Hyperband(
    hypermodel=CNNHyperModel(),
    objective="val_accuracy",
    max_epochs=50,  
    factor=3,
    directory="malaria_cnn",
    project_name="malaria_cnn",
)

# Perform hyperparameter search
tuner.search(
    train_data,
    epochs=50,  
    validation_data=val_data,
    callbacks=[early_stopping_callback()], 
)

# Get the best model and evaluate it
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_accuracy = best_model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy:.4f}")
