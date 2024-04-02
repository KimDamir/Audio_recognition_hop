import numpy as np
import tensorflow as tf
from hopfield import extended_storkey_update
import time
BATCH_SIZE = 10

def squeeze(audio, labels):
    audio=tf.squeeze(audio, axis=-1)
    return audio, labels

def get_spectogram(waveform):
    spectogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectogram = tf.abs(spectogram)
    spectogram = spectogram[..., tf.newaxis]
    return spectogram

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
    
def plot_spectrogram(spectrogram, ax):
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def train(sess, network, dataset):
    """
    Train the Hopfield network.
    """
    image_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(124*129))
    label_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=(5))
    joined = tf.compat.v1.concat((tf.greater_equal(image_ph, 0.5), label_ph), axis=-1)
    update = extended_storkey_update(joined, network.weights)
    sess.run(tf.compat.v1.global_variables_initializer())
    i = 0
    start_time = time.time()
    images = np.concatenate([images for images, labels in dataset], axis=0)
    labels = np.concatenate([labels for images, labels in dataset], axis=0)
    for label, image in zip(labels, images):
        sess.run(update, feed_dict={image_ph: image, label_ph: label})
        i += 1
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            frac_done = i / len(dataset.images)
            remaining = elapsed * (1-frac_done)/frac_done
            print('Done %.1f%% (eta %.1f minutes)' % (100 * frac_done, remaining/60))
 
def accuracy(sess, network, dataset):
    """
    Compute the test-set accuracy of the Hopfield network.
    """
    images_ph = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, 124*129))
    preds = classify(network, images_ph)
    num_right = 0
    images = np.concatenate([images for images, labels in dataset], axis=0)
    labels = np.concatenate([labels for images, labels in dataset], axis=0)
    for i in range(0, len(images), BATCH_SIZE):
        ims = images[i : i+BATCH_SIZE]
        labs = labels[i : i+BATCH_SIZE]
        preds_out = sess.run(preds, feed_dict={images_ph: ims})
        num_right += np.dot(preds_out.flatten(), labs.flatten())
    return num_right / len(images)

def classify(network, images):
    """
    Classify the images using the Hopfield network.

    Returns:
      A batch of one-hot vectors.
    """
    numeric_vec = tf.cast(tf.greater_equal(images, 0.5), tf.float32)*2 - 1
    thresholds = network.thresholds[-5:]
    logits = tf.matmul(numeric_vec, network.weights[:124*129, -5:]) - thresholds
    return tf.one_hot(tf.argmax(logits, axis=-1), 5)  