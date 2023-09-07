import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Conv1DTranspose,Concatenate,Input
import numpy as np
from tqdm.notebook import tqdm
import numpy as np
from scipy.io.wavfile import write

def create_model():
    batching_size = 12000
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
    return model

def get_audio(path):
    audio,_ = tf.audio.decode_wav(tf.io.read_file(path),1)
    return audio

def inference_preprocess(path):
    audio = get_audio(path)
    audio_len = audio.shape[0]
    batches = []
    for i in range(0,audio_len-batching_size,batching_size):
        batches.append(audio[i:i+batching_size])

    batches.append(audio[-batching_size:])
    diff = audio_len - (i + batching_size)
    return tf.stack(batches), diff

def predict(path):
    test_data,diff = inference_preprocess(path)
    predictions = model.predict(test_data)
    final_op = tf.reshape(predictions[:-1],((predictions.shape[0]-1)*predictions.shape[1],1))
    final_op = tf.concat((final_op,predictions[-1][-diff:]),axis=0)
    return final_op

# LOAD THE SAVED MODEL
model = create_model()
model.load_weights('NoiseSuppressionModel_Model.h5')

batching_size = 12000
# rate = 44100
rate = 16000

audio_path = "NoisyData/p226_006.wav"
data = tf.squeeze(predict(audio_path))
scaled = np.int16(data / np.max(np.abs(data)) * 32767)
write('suppressed_audio_p226_006.wav', rate, scaled)