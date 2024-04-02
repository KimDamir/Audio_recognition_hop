import pathlib
import functions
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from hopfield import Network, extended_storkey_update
from sklearn.metrics import classification_report
from keras import Sequential
from keras import layers
from keras.optimizers import SGD


tf.compat.v1.disable_v2_behavior()
DATASET_PATH = 'data/'
data_dir = pathlib.Path(DATASET_PATH)

train_ds, val_ds = keras.utils.audio_dataset_from_directory(directory=data_dir, labels='inferred', batch_size=1, validation_split=0.2, seed=0 , subset='both', output_sequence_length=16000)

train_ds = train_ds.map(functions.squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(functions.squeeze, tf.data.AUTOTUNE)
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)
  
train_spectrogram_ds = functions.make_spec_ds(train_ds)
val_spectrogram_ds = functions.make_spec_ds(val_ds)
test_spectrogram_ds = functions.make_spec_ds(test_ds)

train_spectrogram_ds = train_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

network = Network(124*129 + 5)
with tf.compat.v1.Session() as sess:
    print('Training...')
    functions.train(sess, network, train_spectrogram_ds)
    print('Evaluating...')
    print('Validation accuracy: ' + str(functions.accuracy(sess, network, val_spectrogram_ds)))
    print('Testing accuracy: ' + str(functions.accuracy(sess, network, test_spectrogram_ds)))
    print('Training accuracy: ' + str(functions.accuracy(sess, network, train_spectrogram_ds)))