import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB3(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
    )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = tf.keras.layers.BatchNormalization()(x)

        top_dropout_rate = 0.5
        num_classes=128
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        x = tf.keras.layers.Dense(num_classes, activation="ReLU")(x)

        num_classes1=64
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout1")(x)
        x = tf.keras.layers.Dense(num_classes1, activation="ReLU")(x)

        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(x)

        full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
        )

        full_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
