import tensorflow as tf


def early_stopping_callback() -> tf.keras.callbacks.EarlyStopping:
    """
    Returns an EarlyStopping callback that stops training when the validation loss
    does not improve and restores the previous best weights to the model.

    Parameters:
        None

    Returns:
        tf.keras.callbacks.EarlyStopping: Early stopping callback with custom settings.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )


def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess the image by resizing and normalization.

    Parameters:
        image (tf.Tensor): Input image tensor.
        label (tf.Tensor): Corresponding label tensor.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            image (tf.Tensor): Resized and nomralized image.
            label (tf.Tensor): Corresponding label.
    """
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
