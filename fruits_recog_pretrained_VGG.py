import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from keras import layers
from keras.models import Sequential
from keras import optimizers
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, regularizers, Flatten, Dense
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import os, os.path
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
import numpy as np 

CNN_base = VGG16(weights='imagenet',include_top=False, input_shape=(100,100,3))

start_t = timer()


# declare the paths of downloaded folder; Training, Validation and Test folder
training_dir = '/Users/bathuy/Downloads/fruits-360/Training'
validation_dir = '/Users/bathuy/Downloads/fruits-360/Validation'
test_dir = '/Users/bathuy/Downloads/fruits-360/Test'

# define the parameters
saved_path = os.path.join(os.getcwd(), 'saved_models')
try:
    os.mkdir(saved_path)
except:
    pass
    
model_name = 'CNN_pretrained_VGG.h5'
model_weights = 'CNN_scratch_pretrained_VGG_weights.h5'
num_classes = 95


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

# define a function to extract features of the CNN_base outputs
def extract_features(dir_path, samples):
    feature_maps = np.zeros(shape=(samples, 3, 3, 512))
    labels = np.zeros(shape=(samples))
    generator = datagen.flow_from_directory(dir_path, 
                            target_size=(100, 100), batch_size=batch_size)
    i = 0
    for images_batch, labels_batch in generator:
        features_batch = CNN_base.predict(images_batch)
        feature_maps[i*batch_size: (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= samples:
            break
    return feature_maps, labels

# extract the outputs of VGG net for the inputs: training set, validation set and test set
train_features, train_labels = extract_features(training_dir, 48905)
validation_features, validation_labels = extract_features(validation_dir, 8205)
test_features, test_labels = extract_features(test_dir, 8216)

# reshape the features before feeding into the Dense layers
train_features = np.reshape(train_features, (48905, 3, 3, 512))
validation_features = np.reshape(validation_features, (8205, 3, 3, 512))
test_features = np.reshape(test_features, (8216, 3, 3, 512))
#--------------------------------------------------------------------------
# define the classifier
model = Sequential()
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(95))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt ,metrics=['accuracy'])

model.summary()
#---------------------------------------------------------------------------------

# train the model
num_epochs = 1


history = model.fit(train_features,train_labels, epochs=num_epochs,
                            validation_data=(validation_features, validation_labels),
                            batch_size=batch_size)


model.save(os.path.join(saved_path, model_name))
model.save_weights(os.path.join(saved_path, model_weights))

# --------display history--------
# list all data in history
print(history.history.keys())

test_loss, test_acc = model.evaluate(test_features, test_labels, steps=8205//batch_size, verbose=1)
print('Test accuracy = ', test_acc)

# summarize history for accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(saved_path, 'train_test_accuracy_VGG.png'))
plt.clf()  # clear figure

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title(('Training and validation loss'))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(saved_path, 'train_test_loss_VGG.png'))
plt.clf()

stop_t = timer()
print('Elapsed time: {}'.format(stop_t - start_t))
