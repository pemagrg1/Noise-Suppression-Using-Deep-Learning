import glob

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv1D, Conv1DTranspose, Input
from tqdm.notebook import tqdm

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(
    config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data_path = "dataset"
clean_sounds = glob.glob(data_path+'/CleanData/*')
noisy_sounds = glob.glob(data_path+'/NoisyData/*')

# Splittting the dataset into train and test
test_clean_sounds = clean_sounds[8000:]  # 3571 data
test_noisy_sounds = noisy_sounds[8000:]
clean_sounds = clean_sounds[:8000]
noisy_sounds = noisy_sounds[:8000]

batching_size = 12000

# PREPARE TRAIN SET
clean_sounds_list, _ = tf.audio.decode_wav(
    tf.io.read_file(clean_sounds[0]), desired_channels=1)
for i in tqdm(clean_sounds[1:]):
  so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  clean_sounds_list = tf.concat((clean_sounds_list, so), 0)

noisy_sounds_list, _ = tf.audio.decode_wav(
    tf.io.read_file(noisy_sounds[0]), desired_channels=1)
for i in tqdm(noisy_sounds[1:]):
  so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  noisy_sounds_list = tf.concat((noisy_sounds_list, so), 0)

clean_sounds_list.shape, noisy_sounds_list.shape

clean_train, noisy_train = [], []

for i in tqdm(range(0, clean_sounds_list.shape[0]-batching_size, batching_size)):
  clean_train.append(clean_sounds_list[i:i+batching_size])
  noisy_train.append(noisy_sounds_list[i:i+batching_size])

clean_train = tf.stack(clean_train)
clean_train.shape

noisy_train = tf.stack(noisy_train)
noisy_train.shape

# PREPARE TEST SET
test_clean_sounds_list, _ = tf.audio.decode_wav(
    tf.io.read_file(test_clean_sounds[0]), desired_channels=1)
for i in tqdm(test_clean_sounds[1:]):
  so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  test_clean_sounds_list = tf.concat((test_clean_sounds_list, so), 0)

test_noisy_sounds_list, _ = tf.audio.decode_wav(
    tf.io.read_file(test_noisy_sounds[0]), desired_channels=1)
for i in tqdm(test_noisy_sounds[1:]):
  so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  test_noisy_sounds_list = tf.concat((test_noisy_sounds_list, so), 0)

test_clean_sounds_list.shape, test_noisy_sounds_list.shape

clean_test, noisy_test = [], []

for i in tqdm(range(0, test_clean_sounds_list.shape[0]-batching_size, batching_size)):
  clean_test.append(test_clean_sounds_list[i:i+batching_size])
  noisy_test.append(test_noisy_sounds_list[i:i+batching_size])

clean_test = tf.stack(clean_test)
clean_test.shape
noisy_test = tf.stack(noisy_test)
noisy_test.shape

# Creating tf.data.Dataset


def get_dataset(x_train, y_train):
  dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.shuffle(100).batch(64, drop_remainder=True)
  return dataset


train_dataset = get_dataset(noisy_train, clean_train)
test_dataset = get_dataset(noisy_test, clean_test)

# CREATING THE MODEL
# series_input = Input(shape = (series_input_train.shape[1],1,))
# inp = None,series_input
inp = Input(shape=(batching_size, 1))
c1 = Conv1D(2, 32, 2, 'same', activation='relu')(inp)
c2 = Conv1D(4, 32, 2, 'same', activation='relu')(c1)
c3 = Conv1D(8, 32, 2, 'same', activation='relu')(c2)
c4 = Conv1D(16, 32, 2, 'same', activation='relu')(c3)
c5 = Conv1D(32, 32, 2, 'same', activation='relu')(c4)

dc1 = Conv1DTranspose(32, 32, 1, padding='same')(c5)
conc = Concatenate()([c5, dc1])
dc2 = Conv1DTranspose(16, 32, 2, padding='same')(conc)
conc = Concatenate()([c4, dc2])
dc3 = Conv1DTranspose(8, 32, 2, padding='same')(conc)
conc = Concatenate()([c3, dc3])
dc4 = Conv1DTranspose(4, 32, 2, padding='same')(conc)
conc = Concatenate()([c2, dc4])
dc5 = Conv1DTranspose(2, 32, 2, padding='same')(conc)
conc = Concatenate()([c1, dc5])
dc6 = Conv1DTranspose(1, 32, 2, padding='same')(conc)
conc = Concatenate()([inp, dc6])
dc7 = Conv1DTranspose(1, 32, 1, padding='same', activation='linear')(conc)
model = tf.keras.models.Model(inp, dc7)
model.summary()

# TRAINING THE MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
              loss=tf.keras.losses.MeanAbsoluteError())
history = model.fit(train_dataset, epochs=20)

# EVALUATE THE MODEL
model.evaluate(test_dataset)

# SAVE THE MODEL
model.save('NoiseSuppressionModel_Model.h5')
