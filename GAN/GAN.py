import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from IPython import display


def make_generator_model():
    """The Generator"""
    inputs = tf.keras.layers.Input(shape=(100,))
    x = tf.keras.layers.Dense(256 * 13 * 13)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Reshape((13, 13, 256))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2DTranspose(1, 3, 2, 'same')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    generator = tf.keras.Model(inputs, x)
    return generator


def make_discriminator_model():
    """The discriminator is a CNN-based image classifier."""
    inputs = tf.keras.layers.Input(shape=(52, 52, 1))
    x = tf.keras.layers.Conv2D(32, 3, 2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    discriminator = tf.keras.Model(inputs, x, name='discriminator')
    return discriminator


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, batch_size, generator, discriminator, g_optimizer, d_optimizer, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, batch_size, generator, discriminator, g_optimizer, d_optimizer, noise_dim):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, batch_size, generator, discriminator, g_optimizer, d_optimizer, noise_dim)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                          epoch + 1,
        #                          seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generate_and_save_images(generator,
                                     epochs,
                                     seed)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def display_image(epoch_no):
    """Display a single image using the epoch number"""
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


WM38_data = np.load('./MixedWM38.npz')
print(WM38_data['arr_0'].shape)  # (38015, 52, 52)
print(WM38_data['arr_1'].shape)  # (38015, 8)

# Center_indices = np.where((WM38_data['arr_1'] == np.array([1, 0, 0, 0, 0, 0, 0, 0])).all(axis=1))[0]
# Donut_indices = np.where((WM38_data['arr_1'] == np.array([0, 1, 0, 0, 0, 0, 0, 0])).all(axis=1))[0]
# edgeLoc_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 1, 0, 0, 0, 0, 0])).all(axis=1))[0]
# edgeRing_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 0, 1, 0, 0, 0, 0])).all(axis=1))[0]
# Loc_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 0, 0, 1, 0, 0, 0])).all(axis=1))[0]
# Nearful_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 0, 0, 0, 1, 0, 0])).all(axis=1))[0]
Scratch_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 0, 0, 0, 0, 1, 0])).all(axis=1))[0]
# Random_indices = np.where((WM38_data['arr_1'] == np.array([0, 0, 0, 0, 0, 0, 0, 1])).all(axis=1))[0]
# print(Center_indices.shape,
#       Donut_indices.shape,
#       edgeLoc_indices.shape,
#       edgeRing_indices.shape,
#       Loc_indices.shape,
#       Nearful_indices.shape,  # 149
#       Scratch_indices.shape,
#       Random_indices.shape)  # 866

# Center_image, Center_label = WM38_data['arr_0'][Center_indices], WM38_data['arr_1'][Center_indices]
# Donut_image, Donut_label = WM38_data['arr_0'][Donut_indices], WM38_data['arr_1'][Donut_indices]
# edgeLoc_image, edgeLoc_label = WM38_data['arr_0'][edgeLoc_indices], WM38_data['arr_1'][edgeLoc_indices]
# edgeRing_image, edgeRing_label = WM38_data['arr_0'][edgeRing_indices], WM38_data['arr_1'][edgeRing_indices]
# Loc_image, Loc_label = WM38_data['arr_0'][Loc_indices], WM38_data['arr_1'][Loc_indices]
# Nearful_image, Nearful_label = WM38_data['arr_0'][Random_indices], WM38_data['arr_1'][Random_indices]
Scratch_image, Scratch_label = WM38_data['arr_0'][Scratch_indices], WM38_data['arr_1'][Scratch_indices]  # shape:(52,52)
Scratch_image = np.expand_dims(Scratch_image, axis=-1).astype(np.float32)
# print(Scratch_image.shape)  # (1000, 52, 52, 1)

plt.imshow(Scratch_image[0], cmap='gray')
plt.axis('off')
plt.show()


generator = make_generator_model()
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
batch_size = 128
num_examples_to_generate = 16  # num must be an integer with 1 <= num <= 16

train_dataset = tf.data.Dataset.from_tensor_slices(Scratch_image).batch(batch_size).shuffle(batch_size)

noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
decision = discriminator(generated_image)

# 使用同一組seed,生成結果才能比較
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Train the model
train(train_dataset, EPOCHS, batch_size=batch_size, generator=generator, discriminator=discriminator,
      g_optimizer=generator_optimizer, d_optimizer=discriminator_optimizer, noise_dim=noise_dim)

# Restore the latest checkpoint.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
display_image(EPOCHS)

