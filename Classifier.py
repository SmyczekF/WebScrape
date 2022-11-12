#C:/Users/Rafał/AppData/Local/Programs/Python/Python310/python.exe "c:/Users/Rafał/Desktop/Text Classifier/Classifier.py"

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

################################VECTORIZATION LAYER##################################

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  no_special_characters = tf.strings.regex_replace(stripped_html,'[ąćęłńóśźż–\xc2\xa0\xbd\xbc(0-9)]','' )
  #no_special_characters = tf.strings.unicode_decode(stripped_html,'UTF-8')
  return tf.strings.regex_replace(no_special_characters,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 2000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

################################VECTORIZATION LAYER##################################

data_dir = os.path.join(os.getcwd(), 'data')

print(os.listdir(data_dir))

batch_size = 32

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    data_dir, 
    batch_size=batch_size)

#for text_batch, label_batch in raw_train_ds.take(1):
# for i in range(5):
#    print("Review", text_batch.numpy()[i])
#    print("Label", label_batch.numpy()[i])

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", custom_standardization(first_review))
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

vocab_dir = "data\\vocab.txt"

f = open(vocab_dir, "w")

#for i in range(len(vectorize_layer.get_vocabulary())):
#  tekst = str(i) + " ---> " + str(vectorize_layer.get_vocabulary()[i]) + "\n"
#  f.write(tekst)


print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))