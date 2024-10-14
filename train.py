import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from model import CNNHyperModel
from utils import early_stopping_callback, preprocess


def load_and_preprocess_data() -> (
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]
):
    """
    Load and preprocess data from the tfds Malaria dataset.

    Parameters:
        None

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
        A tuple containing:
            train_data (tf.data.Dataset): Training data.
            val_data (tf.data.Dataset): Validation data.
            test_data (tf.data.Dataset): Test data.
            info (tfds.core.DatasetInfo): Dataset information.
    """
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

    return train_data, val_data, test_data, info


def build_tuner() -> kt.Hyperband:
    """
    Initializes the Keras HyberBand tuner.

    Parameters:
        None

    Returns:
        keras_tuner.Hyperband: Keras HyberBand tuner.
    """
    tuner = kt.Hyperband(
        hypermodel=CNNHyperModel(),
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="malaria_cnn",
        project_name="malaria_cnn",
    )
    return tuner


def tune_model(
    tuner: kt.Hyperband, train_data: tf.data.Dataset, val_data: tf.data.Dataset
) -> tf.keras.Model:
    """
    Carries out hyperparameter optimization to find the best model for a give configuration.

    Parameters:
        tuner (keras_tuner.Hyperband): The Keras Tuner Hyperband instance.
        train_data (tf.data.Dataset): Training data.
        val_data (tf.data.Dataset): Validation data.

    Returns:
        tf.keras.Model: The best model obtained from the tuning process.
    """
    tuner.search(
        train_data,
        epochs=50,
        validation_data=val_data,
        callbacks=[early_stopping_callback()],
    )

    return tuner.get_best_models(num_models=1)[0]


def compile_and_train_model(
    model: tf.keras.Model, train_data: tf.data.Dataset, val_data: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """
    Compile and train the model using the training and validation data.

    Parameters:
        model (tf.keras.Model): The model to compile and train.
        train_data (tf.data.Dataset): Preprocessed training data.
        val_data (tf.data.Dataset): Preprocessed validation data.

    Returns:
        tf.keras.callbacks.History: Training history of the model.
    """

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    history = model.fit(
        train_data,
        epochs=50,
        validation_data=val_data,
        callbacks=[early_stopping_callback()],
    )
    return history


def evaluate_model(
    model: tf.keras.Model, test_data: tf.data.Dataset
) -> tuple[float, float, float, float, float]:
    """
    Evaluate the trained model on the test data. Calcualtes and prints perfomance metrics.

    Parameters:
        model (tf.keras.Models): The trained model to evaluate.
        test_data (tf.data.Dataset): Preprocessed test data.

    Returns:
        tuple [float, float, float, float, float]: A tuple containing:
            test_loss (float): The loss on the test dataset.
            test_accuracy (float): The accuracy on the test dataset.
            test_precision (float): The precision on the test dataset.
            test_recall (float): The recall on the test dataset.
            test_f1 (float): The F1 score on the test dataset.
    """

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data)

    test_f1 = (
        2
        * (test_precision * test_recall)
        / (test_precision + test_recall + tf.keras.backend.epsilon())
    )

    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    return test_loss, test_accuracy, test_precision, test_recall, test_f1


def main():
    train_data, val_data, test_data, info = load_and_preprocess_data()

    tuner = build_tuner()
    best_model = tune_model(tuner, train_data, val_data)

    compile_and_train_model(best_model, train_data, val_data)

    evaluate_model(best_model, test_data)


if __name__ == "__main__":
    main()
