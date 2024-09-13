import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from model import CNNHyperModel
from utils import early_stopping_callback, preprocess

(train_data, val_data, test_data), info = tfds.load(
    "malaria",
    split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
    as_supervised=True,
    with_info=True,
)
train_data = (
    train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .cache()
    .shuffle(info.splits["train"].num_examples)
    .batch(64, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
val_data = (
    val_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .cache()
    .batch(64, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
test_data = (
    test_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .cache()
    .batch(64, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


tuner = kt.Hyperband(
    hypermodel=CNNHyperModel(),
    objective="val_accuracy",
    max_epochs=50,
    factor=3,
    directory="malaria_cnn",
    project_name="malaria_cnn",
)


tuner.search(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stopping_callback()],
)

best_model = tuner.get_best_models(num_models=1)[0]


best_model.compile(
    optimizer=best_model.optimizer,
    loss=best_model.loss,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ],
)

history = best_model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stopping_callback()],
)

test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(test_data)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)


print(f"Accuracy: {test_accuracy:.4f}")
print(f"Loss: {test_loss:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
