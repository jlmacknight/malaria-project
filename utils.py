import tensorflow as tf

def early_stopping_callback():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,  
        restore_best_weights=True,
    )

def preprocess(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label