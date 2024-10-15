import tensorflow as tf
import keras_tuner as kt


class CNNHyperModel(kt.HyperModel):
    """
    HyperModel subclass defining the structure of the CNN and hyperparameters for tuning.

    Attributes:
        None

    Method:
        build(hp: keras_tuner.HyperParameters) -> tf.keras.Model:
            Build and return a CNN model with the hyperparameter configurations defined.
    """

    def build(self, hp) -> tf.keras.Model:
        """
        Constructs a CNN based on the hyperparameter configuration.

        Parameters:
            hp (keras_tuner.HyperParameters): Object containing hyperparamter space and current
            values.

        Returns:
            tf.keras.Model: A compiled model instance.
        """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.RandomFlip())
        model.add(tf.keras.layers.RandomRotation(0.2))
        model.add(tf.keras.layers.RandomZoom((-0.2, 0.2)))
        model.add(tf.keras.layers.RandomTranslation(0.1, 0.1))
        model.add(tf.keras.layers.RandomContrast(0.2))

        for i in range(hp.Int("num_conv_layers", 1, 6)):
            model.add(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        f"filters_{i+1}",
                        min_value=32,
                        max_value=128,
                        step=2,
                        sampling="log",
                    ),
                    kernel_size=hp.Choice(f"kernel_size_{i+1}", values=[3, 5]),
                    activation="relu",
                    padding="same",
                )
            )
            model.add(tf.keras.layers.MaxPool2D())
            model.add(tf.keras.layers.BatchNormalization())
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(
                        f"dropout_rate_conv_{i+1}",
                        min_value=0.0,
                        max_value=0.5,
                        step=0.1,
                    )
                )
            )

        model.add(tf.keras.layers.Flatten())

        for j in range(hp.Int("num_dense_layers", 1, 3)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        f"dense_units_{j+1}", min_value=32, max_value=128, step=32
                    ),
                    activation="relu",
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(
                        f"dropout_rate_dense_{j+1}",
                        min_value=0.0,
                        max_value=0.5,
                        step=0.1,
                    )
                )
            )

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float(
                    "learning_rate",
                    min_value=1e-5,
                    max_value=1e-2,
                    step=10,
                    sampling="log",
                )
            ),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
            ],
        )

        return model
