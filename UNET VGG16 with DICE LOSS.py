import os
import pandas as pd
import numpy as np
import keras
import scipy
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications import VGG16

import numpy as np
import os
import random
from PIL import Image

import tensorflow as tf
import tensorflow.python.keras.backend as K

import matplotlib.pyplot as plt

# Seed
seed = 2019
np.random.seed(seed)
random.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1,inter_op_parallelism_threads = 1)
tf.set_random_seed(seed)
sess = tf.Session(graph = tf.get_default_graph(), config = session_conf)
K.set_session(sess)

#%%

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=200,
                                   horizontal_flip=True,
                                   width_shift_range=30,
                                   zoom_range=0.08
                                   )
val_datagen = ImageDataGenerator(rescale=1./255
                                 )
test_datagen = ImageDataGenerator(rescale=1./255
                                  )

os.chdir('D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/')

data_path = 'D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/'
frames_path_t = data_path + 'frames/train_frames/'
frames_path_v = data_path + 'frames/val_frames/'
frames_path_te = data_path + 'frames/test_frames/'
masks_path_t = data_path + 'masks/train_masks/'
masks_path_v = data_path + 'masks/val_masks/'
masks_path_te = data_path + 'masks/test_masks/'


input_size = 256, 256
batch_size_conv = 4
batch_train = 49

color_mode='rgb'
#%%
seeds = 1
train_image_generator = train_datagen.flow_from_directory(frames_path_t,
                                                          color_mode = color_mode,
                                                          class_mode = None,
                                                          batch_size = batch_train,
                                                          shuffle=False,
                                                          seed = seeds)

train_mask_generator = train_datagen.flow_from_directory(masks_path_t,
                                                         color_mode='grayscale',
                                                         class_mode = None,
                                                         batch_size = batch_train,
                                                         shuffle=False,
                                                          seed = seeds)


#%%
##augmentation images
train_images = next(train_image_generator)
train_masks = next(train_mask_generator)
for i in range(2):
    train_image = next(train_image_generator)
    train_mask= next(train_mask_generator)
    train_images = np.concatenate((train_images,train_image),0)
    train_masks = np.concatenate((train_masks,train_mask),0)

#%%
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next()) 
        
#train
data_x_train = train_images
np.save('D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/data_x_train',data_x_train)

data_y_train = train_masks
np.save('D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/data_y_train',data_y_train)
#%%
val_image_generator = val_datagen.flow_from_directory(frames_path_v,
                                                       color_mode = color_mode,
                                                       class_mode = None,
                                                       batch_size = batch_size_conv,
                                                       shuffle=False,
                                                       seed = seeds)

val_mask_generator = val_datagen.flow_from_directory(masks_path_v,
                                                      color_mode = 'grayscale',
                                                      class_mode = None,
                                                      batch_size = batch_size_conv,
                                                      shuffle=False,
                                                          seed = seeds)

test_image_generator = test_datagen.flow_from_directory(frames_path_te,
                                                         color_mode = color_mode,
                                                         class_mode = None,
                                                         batch_size = batch_size_conv,
                                                         shuffle=False,
                                                         seed = seeds)

test_mask_generator = test_datagen.flow_from_directory(masks_path_te,
                                                        color_mode = 'grayscale',
                                                        class_mode = None,
                                                        batch_size = batch_size_conv,
                                                        shuffle=False,
                                                        seed=seeds)

#load augmented data from looping
data_x_train=np.load('D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/data_x_train.npy')
data_y_train=np.load('D:/S2/NCTU/biomedical image segmentation/coba run 3 grayscale only image with mask/data_y_train.npy')
data_y_train=np.round(data_y_train)

# Validation
val_image_generator.batch_size  = val_image_generator.samples
data_x_val = next(val_image_generator)
data_x_val.shape

val_mask_generator.batch_size  = val_mask_generator.samples
data_y_val = next(val_mask_generator)
data_y_val.shape

# Test
test_image_generator.batch_size  = test_image_generator.samples
data_x_test = next(test_image_generator)
data_x_test.shape

test_mask_generator.batch_size  = test_mask_generator.samples
data_y_test = next(test_mask_generator)
data_y_test.shape

data_x_full = np.concatenate((np.concatenate((data_x_train, data_x_val), axis = 0), data_x_test), axis = 0)
data_y_full = np.round(np.concatenate((np.concatenate((data_y_train, data_y_val), axis = 0), data_y_test), axis = 0))
#%%
input_shape = (256, 256, 3)

def VGG_16(weights_path=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())

    if weights_path:
        model.load_weights(weights_path)

    return model

#con_vgg_model = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5') #with imagenet (download the h5 file via github)

con_vgg_model = VGG16(include_top=False, weights= None, input_shape=input_shape) #with no imagenet

'add dropout'
# Store the fully connected layers
fc1 = con_vgg_model.layers[-3]
fc2 = con_vgg_model.layers[-2]
predictions = con_vgg_model.layers[-1]

# Create the dropout layers
dropout1 = Dropout(0.85)
dropout2 = Dropout(0.85)

# Reconnect the layers
x = dropout1(fc1.output)
x = fc2(x)
x = dropout2(x)
predictors = predictions(x)
'=========='
con_model = Model(inputs = con_vgg_model.input, outputs=predictors)

#con_model = Model(inputs = con_vgg_model.input, outputs=con_vgg_model.layers[-2].output)

from collections import defaultdict, OrderedDict
from keras.models import Model

layer_size_dict = defaultdict(list)
inputs = []
for layer_index, c_layer in enumerate(con_model.layers):
    if not c_layer.__class__.__name__ == 'InputLayer':
        layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
    else:
        inputs += [c_layer]

# Freeze dict
layer_size_dict = OrderedDict(layer_size_dict.items())
for k, v in layer_size_dict.items():
    print(k, [w.__class__.__name__ for w in v])

# Take the last layer of each shape and make it into an output
pretrained_encoder = Model(inputs = con_model.get_input_at(0), 
                           outputs = [v[-1].get_output_at(0) for k, v in layer_size_dict.items()])
pretrained_encoder.trainable = False

from keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D

x_wid, y_wid = data_x_train.shape[1:3]
in_t0 = Input(data_x_train.shape[1:], name = 'T0_Image')
wrap_encoder = lambda i_layer: {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}

t0_outputs = wrap_encoder(in_t0)
lay_dims = sorted(t0_outputs.keys(), key = lambda x: x[0])
skip_layers = 2
last_layer = None
for k in lay_dims[skip_layers:]:
    cur_layer = t0_outputs[k]
    channel_count = cur_layer._keras_shape[-1]
    cur_layer = Conv2D(channel_count//2, kernel_size=(3,3), padding = 'same', activation = 'linear')(cur_layer)
    cur_layer = BatchNormalization()(cur_layer) # gotta keep an eye on that internal covariant shift
    cur_layer = Activation('relu')(cur_layer)
    
    if last_layer is None:
        x = cur_layer
    else:
        last_channel_count = last_layer._keras_shape[-1]
        x = Conv2D(last_channel_count//2, kernel_size=(3,3), padding = 'same')(last_layer)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([cur_layer, x])
    last_layer = x
    
final_output = Conv2D(data_y_train.shape[-1], kernel_size=(1,1), padding = 'same', activation = 'sigmoid')(last_layer)
crop_size = 20
final_output = Cropping2D((crop_size, crop_size))(final_output)
final_output = ZeroPadding2D((crop_size, crop_size))(final_output)

unet_model = Model(inputs = [in_t0],outputs = [final_output])

from keras.callbacks import ModelCheckpoint

' DICE LOSS '
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + 
                                           K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
' ============ '


opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)

unet_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc',tf.keras.metrics.AUC(),dice_coef])

save_weight_path = 'weights binary try dropout/saved_UNet_model-{epoch:03d}-{val_acc}-{auc:.5f}-{dice_coef}.h5'

checkpoint_save = ModelCheckpoint(save_weight_path, monitor=['val_acc','auc','dice_coef'], verbose=1,
                             save_best_only=False, save_weights_only=True, mode='max',
                             period=1)
callbacks_list = [checkpoint_save]    
    
results = unet_model.fit(data_x_train,data_y_train, epochs = 50, 
                                   validation_data = (data_x_val,data_y_val) ,
                                   callbacks = callbacks_list,
                                   batch_size = 4
                                   )    
    
unet_model.save_weights('UNet_dropout.h5')    
    
unet_model_json = unet_model.to_json()
with open("UNet_dropout.json", "w") as json_file:
    json_file.write(unet_model_json)
    
#%%
train_score = unet_model.evaluate(data_x_train, data_y_train, verbose=1,batch_size=2)
val_score = unet_model.evaluate(data_x_val, data_y_val, verbose=1,batch_size=2)
test_score = unet_model.evaluate(data_x_test, data_y_test, verbose=1,batch_size=2)

predict_train = unet_model.predict(data_x_train)
predict_val = unet_model.predict(data_x_val)
predict_test = unet_model.predict(data_x_test)

predict_full = unet_model.predict(data_x_full)

predict_train.shape
np.max(predict_train)

predict_full.shape
np.max(predict_full)

import imageio
for i in range(0, predict_train.shape[0]):
    imageio.imwrite('results try dice loss/predict train set ' + str(i) + ' result_grayscale.jpg', predict_train[i])

for i in range(0, predict_val.shape[0]):
    imageio.imwrite('results try dice loss/predict val set ' + str(i) + ' result_grayscale.jpg', predict_val[i])

for i in range(0, predict_test.shape[0]):
    imageio.imwrite('results try dice loss/predict test set ' + str(i) + ' result_grayscale.jpg', predict_test[i])

#%%
'''
Load new saved model
'''

json_file = open('UNet_dropout.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('UNet_dropout.h5')

loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
loaded_model.evaluate(data_x_val, data_y_val, verbose=1)


train_score_loaded = loaded_model.evaluate(data_x_train, data_y_train, verbose=1)
val_score_loaded = loaded_model.evaluate(data_x_val, data_y_val, verbose=1)
test_score_loaded = loaded_model.evaluate(data_x_test, data_y_test, verbose=1)


predict_train_loaded = loaded_model.predict(data_x_train)
predict_val_loaded = loaded_model.predict(data_x_val)
predict_test_loaded = loaded_model.predict(data_x_test)

#%%
'''
Load model for every epoch
'''

open_path_model = 'weights binary try dropout'

data_model = []
file_name_model = []
for root, dirs, files in os.walk(open_path_model):
    for filename in files:
         if filename.endswith('.h5'):
            data_model.append(os.path.join(root, filename))
            file_name_model.append(filename)
data_model
len(data_model)

file_name_model
len(file_name_model)


train_loaded_model = loaded_model
val_loaded_model   = loaded_model
test_loaded_model  = loaded_model


#Load val model

val_loaded_model_evaluation = []

for name in data_model:
    print(name)
    val_loaded_model.load_weights(name)
    val_loaded_model.compile(loss=dice_coef_loss, optimizer=opt, metrics=['acc',tf.keras.metrics.AUC(),dice_coef])
    val_loaded_model_evaluation.append(val_loaded_model.evaluate(data_x_val, data_y_val, verbose=1,batch_size=2))

val_loaded_model_evaluation

len(val_loaded_model_evaluation)


np.savetxt("val_dropout_binary.csv", val_loaded_model_evaluation)


#Load train model

train_loaded_model_evaluation = []

for name in data_model:
    print(name)
    train_loaded_model.load_weights(name)
    train_loaded_model.compile(loss=dice_coef_loss, optimizer=opt, metrics=['acc',tf.keras.metrics.AUC(),dice_coef])
    train_loaded_model_evaluation.append(train_loaded_model.evaluate(data_x_train, data_y_train, verbose=1,batch_size=2))

train_loaded_model_evaluation

len(train_loaded_model_evaluation)


np.savetxt("train_dropout_binary.csv", train_loaded_model_evaluation)



test_loaded_model_evaluation = []

for name in data_model:
    print(name)
    test_loaded_model.load_weights(name)
    test_loaded_model.compile(loss=dice_coef_loss, optimizer=opt, metrics=['acc',tf.keras.metrics.AUC(),dice_coef])
    test_loaded_model_evaluation.append(test_loaded_model.evaluate(data_x_test, data_y_test, verbose=1,batch_size=2))

test_loaded_model_evaluation

len(test_loaded_model_evaluation)


np.savetxt("test_dropout_binary.csv", test_loaded_model_evaluation)
#%%

'''
Plotting result
'''

train_metrics = pd.read_csv('D:/S2/NCTU/biomedical image segmentation/coba run 4 (all image)/contrast and no contrast image/train_loaded_model_binaryloss_imagenet.csv', header=None, sep=";")
val_metrics = pd.read_csv('D:/S2/NCTU/biomedical image segmentation/coba run 4 (all image)/contrast and no contrast image/val_loaded_model_binaryloss_imagenet.csv', header=None, sep=";")
test_metrics = pd.read_csv('D:/S2/NCTU/biomedical image segmentation/coba run 4 (all image)/contrast and no contrast image/test_loaded_model_binaryloss_imagenet.csv', header=None, sep=";")

train_loss = train_metrics[0]
train_acc = train_metrics[1]
train_auc = train_metrics[2]
train_dice_coef = train_metrics[3]

val_loss = val_metrics[0]
val_acc = val_metrics[1]
val_auc = val_metrics[2]
val_dice_coef = val_metrics[3]

test_loss= test_metrics[0]
test_acc= test_metrics[1]
test_auc = test_metrics[2]
test_dice_coef = test_metrics[3]

#%%
plt.style.use('ggplot') 

plt.plot(train_loss[0:51], linewidth=3, color = 'dodgerblue')
plt.plot(val_loss[0:51], linewidth=3, color = 'orange')
#plt.plot(test_loss[0:11], linewidth=3, color = 'red')
plt.legend(['train_loss','val_loss','test_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.axis([-1, 50, 0, 0.25])
#plt.savefig("D:/S2/NCTU/biomedical image segmentation/presentation progress 10 output gambar/dice 100 epoch/loss_dropout_binary.png")

#%%
plt.plot(train_acc[0:51], linewidth=3, color = 'dodgerblue')
plt.plot(val_acc[0:51], linewidth=3, color = 'orange')
#plt.plot(test_acc[0:11], linewidth=3, color = 'red')
plt.legend(['train_acc','val_acc','test_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.axis([-1,50, 0.85, 1])
#plt.savefig("D:/S2/NCTU/biomedical image segmentation/presentation progress 10 output gambar/dice 100 epoch/acc_dropout_binary.png")

#%%
plt.plot(train_auc[0:51], linewidth=3, color = 'dodgerblue')
plt.plot(val_auc[0:51], linewidth=3, color = 'orange')
#plt.plot(test_auc[0:11], linewidth=3, color = 'red')
plt.legend(['train_auc','val_auc','test_auc'])
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.axis([-1, 50, 0.3, 1])
#plt.savefig("D:/S2/NCTU/biomedical image segmentation/presentation progress 10 output gambar/dice 100 epoch/auc_dropout_binary.png")

#%%
plt.plot(train_dice_coef[0:51], linewidth=3, color = 'dodgerblue')
plt.plot(val_dice_coef[0:51], linewidth=3, color = 'orange')
#plt.plot(test_dice_coef[0:11], linewidth=3, color = 'red')
plt.legend(['train_dice_loss','val_dice_loss','test_dice_loss'])
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.axis([-1, 50, 0.95, 1])
#plt.savefig("D:/S2/NCTU/biomedical image segmentation/presentation progress 10 output gambar/dice 100 epoch/dice_coef_dropout_binary.png")