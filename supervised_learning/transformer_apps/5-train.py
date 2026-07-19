#!/usr/bin/env python3
"""
Train module for Transformer model
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule for Transformer training"""

    def __init__(self, d_model, warmup_steps=4000):
        """Initializer"""
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Calculates learning rate based on current step"""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden,
        input_vocab, target_vocab,
        max_len, max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    def loss_function(real, pred):
        """Calculates the masked loss value"""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Calculates the masked accuracy value"""
        accuracies = tf.equal(real, tf.argmax(pred, axis=-1))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        """Performs a single training step"""
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, comb_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp, tar_inp, True,
                enc_mask, comb_mask, dec_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss.update_state(loss)
        train_accuracy.update_state(accuracy_function(tar_real, predictions))

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inp, tar) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    "Epoch {}, Batch {}: Loss {}, Accuracy {}".format(
                        epoch + 1,
                        batch,
                        train_loss.result().numpy(),
                        train_accuracy.result().numpy()
                    )
                )

        print(
            "Epoch {}: Loss {}, Accuracy {}".format(
                epoch + 1,
                train_loss.result().numpy(),
                train_accuracy.result().numpy()
            )
        )

    return transformer
