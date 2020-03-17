import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# LOAD TEXT FILES #############################################################
epIV_text = open('StarWars_EpisodeIV_script.txt', 'r').read()
epV_text = open('StarWars_EpisodeV_script.txt', 'r').read()
epVI_text = open('StarWars_EpisodeVI_script.txt', 'r').read()

# [Debugging] Test Read
# print(epIV_text[:300])
# print(epV_text[:300])
# print(epVI_text[:300])

# VECTORIZE TEXT ##############################################################
#   Allow text to be understood by neural network
vocab = sorted(set(epIV_text))

# [1] dictionary of unique characters
char_to_index = {unique:index for index, unique in enumerate(vocab)}
# [2] numpy array of unique characters
index_to_char = np.array(vocab)
# [3] recompose script as array of integers
text_as_int = np.array([char_to_index[char] for char in epIV_text])

# SEQUENCE LENGTH #############################################################
seq_length = 100    # arbitrary but manageable
examples_per_epoch = len(epIV_text) // (seq_length + 1) # // for rounded division (no fractional text gen)

# Tensorflow Dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# for i in char_dataset:
#     print(index_to_char[i.numpy()])

# BATCHING ####################################################################
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
# for item in sequences:
#     print(repr(''.join(index_to_char[item.numpy()])))

# Split Input and Target
#   Identify target given the first character in a sequence
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# for input_example, target_example in dataset.take(5):
#     print('input data', repr(''.join(index_to_char[input_example.numpy()])))
#     print('target data', repr(''.join(index_to_char[target_example.numpy()])))
#
# for i, (input_index, target_index) in enumerate(zip(input_example[:10], target_example[:10])):
#     print('step {:4d}'.format(i))
#     print('     input {} ({:s})'.format(input_index, repr(index_to_char[input_index])))
#     print('     expected output {} ({:s})'.format(target_index, repr(index_to_char[target_index])))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)

# BUILD MODEL #################################################################
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
                                tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
                                tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                                tf.keras.layers.Dense(vocab_size)
                                ])
    return model

# Build and Compile Model
#model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim,rnn_units=rnn_units, batch_size=BATCH_SIZE)

# Observe Parameters and Summary
# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, '# (batch_size, seq_length, vocab_size)')

#model.summary()

# TRAIN MODEL #################################################################
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#model.compile(optimizer='adam', loss=loss)

# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor('val_acc', patience=15))


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
EPOCHS = 192

#history = model.fit(dataset, epochs = EPOCHS, callbacks=[checkpoint_callback])

model = build_model(vocab_size, embedding_dim, rnn_units,batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1,None]))

model.summary()

def generate_text(model, start_string):
    num_generate = 1500
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval,0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(index_to_char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string='INT.'))
