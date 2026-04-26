#!/usr/bin/env python3
"""
Transfer Knowledge: CIFAR-10 classification using EfficientNetB0.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess_data(X, Y):
    """
    Pre-process CIFAR-10 data for EfficientNetB0.
    """
    # Normalise pixels to [0, 1]
    X_p = X.astype("float32") / 255.0

    # One-hot encode labels
    Y_p = keras.utils.to_categorical(Y, 10)

    return X_p, Y_p


def build_model():
    """
    Build a transfer-learning model:
      Input (32×32×3)
        → Lambda: resize to 224×224   (required by EfficientNetB0)
        → EfficientNetB0 base (frozen initially)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu) + Dropout(0.4)
        → Dense(128, relu) + Dropout(0.3)
        → Dense(10, softmax)
    """
    inputs = keras.Input(shape=(32, 32, 3), name="input_32x32")

    # Scale up to the size EfficientNet was trained on
    x = keras.layers.Lambda(
        lambda img: tf.image.resize(img, (224, 224)),
        name="upscale_224"
    )(inputs)

    # EfficientNetB0 base — include_top=False removes the ImageNet classifier
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None          # we add our own pooling below
    )
    base.trainable = False    # freeze all base layers initially

    x = base.output
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.BatchNormalization(name="bn_head")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = keras.layers.Dropout(0.4, name="drop_256")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = keras.layers.Dropout(0.3, name="drop_128")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_efficientnet")
    return model, base


def main():
    # ------------------------------------------------------------------ #
    # 1. Load & preprocess data                                           #
    # ------------------------------------------------------------------ #
    print("Loading CIFAR-10 …")
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    Y_train = Y_train.flatten()
    Y_test  = Y_test.flatten()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test,  Y_test  = preprocess_data(X_test,  Y_test)

    # ------------------------------------------------------------------ #
    # 2. Build model                                                      #
    # ------------------------------------------------------------------ #
    model, base = build_model()
    model.summary()

    # ------------------------------------------------------------------ #
    # 3. Phase 1 — train only the Dense head                             #
    #    To speed things up we cache the frozen-base output ONCE,        #
    #    then train a tiny sub-model that takes those features as input. #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: caching frozen-base features ===")

    # Sub-model that stops just before our custom head
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.get_layer("gap").output,
        name="feature_extractor"
    )

    # Predict in batches — this runs the full frozen base once
    batch = 256
    print("  Computing training features …")
    train_features = feature_extractor.predict(X_train, batch_size=batch, verbose=1)
    print("  Computing validation features …")
    val_features   = feature_extractor.predict(X_test,  batch_size=batch, verbose=1)

    # Build a lightweight head-only model that trains on cached features
    feat_input = keras.Input(shape=(train_features.shape[1],), name="cached_features")
    h = keras.layers.BatchNormalization()(feat_input)
    h = keras.layers.Dense(256, activation="relu")(h)
    h = keras.layers.Dropout(0.4)(h)
    h = keras.layers.Dense(128, activation="relu")(h)
    h = keras.layers.Dropout(0.3)(h)
    h = keras.layers.Dense(10, activation="softmax")(h)
    head_model = keras.Model(feat_input, h, name="head_only")

    head_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=3, verbose=1
        ),
    ]

    print("  Training head on cached features …")
    head_model.fit(
        train_features, Y_train,
        validation_data=(val_features, Y_test),
        epochs=30,
        batch_size=512,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # Copy learned head weights into the full model
    head_layers = ["bn_head", "dense_256", "drop_256", "dense_128", "drop_128", "predictions"]
    head_model_layers = head_model.layers[1:]   # skip the Input layer

    for full_name, head_layer in zip(head_layers, head_model_layers):
        model.get_layer(full_name).set_weights(head_layer.get_weights())

    # ------------------------------------------------------------------ #
    # 4. Phase 2 — fine-tune top ~40 % of base layers                   #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 2: fine-tuning top layers ===")

    # Unfreeze the top portion of the base
    base.trainable = True
    fine_tune_from = int(len(base.layers) * 0.6)   # unfreeze top 40 %
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Base layers fine-tuned: {trainable_count} / {len(base.layers)}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),   # much lower LR for fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "cifar10.h5", monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # Data augmentation
    data_aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    # Build augmented dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        .shuffle(10000)
        .batch(64)
        .map(lambda x, y: (data_aug(x, training=True), y),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        .batch(64)
        .prefetch(AUTOTUNE)
    )

    print("  Fine-tuning full model …")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks_phase2,
        verbose=1
    )

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {acc * 100:.2f}%")

    if acc < 0.87:
        print("WARNING: accuracy below 87 % — consider more fine-tune epochs.")

    # ModelCheckpoint already saved the best weights; ensure final compile
    model.save("cifar10.h5")
    print("Model saved → cifar10.h5")
