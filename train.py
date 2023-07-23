import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Conv1DTranspose,Concatenate,Input
import numpy as np
import IPython.display
import glob
from tqdm.notebook import tqdm
import librosa.display
import matplotlib.pyplot as plt

clean_sounds = glob.glob('/dataset/CleanData/*')
noisy_sounds = glob.glob('/dataset/NoisyData/*')
print(len(clean_sounds))

# print(type(clean_sounds))
# clean_sounds = clean_sounds[:1000]
# noisy_sounds = noisy_sounds[:1000]
# print(len(clean_sounds))

clean_sounds_list,_ = tf.audio.decode_wav(tf.io.read_file(clean_sounds[0]),desired_channels=1)
for i in tqdm(clean_sounds[1:]):
  so,_ = tf.audio.decode_wav(tf.io.read_file(i),desired_channels=1)
  clean_sounds_list = tf.concat((clean_sounds_list,so),0)

noisy_sounds_list,_ = tf.audio.decode_wav(tf.io.read_file(noisy_sounds[0]),desired_channels=1)
for i in tqdm(noisy_sounds[1:]):
  so,_ = tf.audio.decode_wav(tf.io.read_file(i),desired_channels=1)
  noisy_sounds_list = tf.concat((noisy_sounds_list,so),0)

clean_sounds_list.shape,noisy_sounds_list.shape

batching_size = 12000

clean_train,noisy_train = [],[]

for i in tqdm(range(0,clean_sounds_list.shape[0]-batching_size,batching_size)):
  clean_train.append(clean_sounds_list[i:i+batching_size])
  noisy_train.append(noisy_sounds_list[i:i+batching_size])

clean_train = tf.stack(clean_train)
noisy_train = tf.stack(noisy_train)

clean_train.shape,noisy_train.shape

def get_dataset(x_train,y_train):
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    dataset = dataset.shuffle(100).batch(64,drop_remainder=True)
    return dataset

# train_dataset = get_dataset(noisy_train[:40000],clean_train[:40000])
# test_dataset = get_dataset(noisy_train[40000:],clean_train[40000:])

train_dataset = get_dataset(noisy_train[:50],clean_train[:50])
test_dataset = get_dataset(noisy_train[50:],clean_train[50:])

# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


# librosa.display.waveshow(np.squeeze(clean_train[5].numpy(),axis=-1))
# librosa.display.waveshow(data, sr=sampling_rate)

# plt.show()

# librosa.display.waveshow(np.squeeze(noisy_train[5].numpy(),axis=-1))
# plt.show()

# Model
inp = Input(shape=(batching_size,1))
c1 = Conv1D(2,32,2,'same',activation='relu')(inp)
c2 = Conv1D(4,32,2,'same',activation='relu')(c1)
c3 = Conv1D(8,32,2,'same',activation='relu')(c2)
c4 = Conv1D(16,32,2,'same',activation='relu')(c3)
c5 = Conv1D(32,32,2,'same',activation='relu')(c4)

dc1 = Conv1DTranspose(32,32,1,padding='same')(c5)
conc = Concatenate()([c5,dc1])
dc2 = Conv1DTranspose(16,32,2,padding='same')(conc)
conc = Concatenate()([c4,dc2])
dc3 = Conv1DTranspose(8,32,2,padding='same')(conc)
conc = Concatenate()([c3,dc3])
dc4 = Conv1DTranspose(4,32,2,padding='same')(conc)
conc = Concatenate()([c2,dc4])
dc5 = Conv1DTranspose(2,32,2,padding='same')(conc)
conc = Concatenate()([c1,dc5])
dc6 = Conv1DTranspose(1,32,2,padding='same')(conc)
conc = Concatenate()([inp,dc6])
dc7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(conc)
model = tf.keras.models.Model(inp,dc7)
model.summary()

tf.keras.utils.plot_model(model,show_shapes=True,show_layer_names=False)

# Training
print("===TRAINING=====")
model.compile(optimizer=tf.keras.optimizers.Adam(0.002),loss=tf.keras.losses.MeanAbsoluteError(), run_eagerly=True)
history = model.fit(train_dataset,epochs=1)

from IPython.display import Audio
# Audio(np.squeeze(noisy_train[22].numpy()),rate=16000)

Audio(tf.squeeze(model.predict(tf.expand_dims(tf.expand_dims(noisy_train[22],-1),0))),rate=16000)
print(model.evaluate(test_dataset))

model.save('NoiseSuppressionModel_Model_server.h5')
