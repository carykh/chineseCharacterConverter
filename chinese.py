import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
learning_rate = 0.001
inputs_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 100x100x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=(2,2), padding='same')
# Now 50x50x16
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 50x50x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=(2,2), padding='same')
# Now 25x25x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=48, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 25x25x48
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=(2,2), padding='same')
# Now 13x13x48
conv4 = tf.layers.conv2d(inputs=maxpool3, filters=64, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 13x13x64
maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=(2,2), padding='same')
# Now 7x7x64 (flatten to 1764)

maxpool4_flat = tf.reshape(maxpool4, [-1,7*7*64])

W_fc1 = weight_variable([7*7*64, 500])
b_fc1 = bias_variable([500])
encoded = tf.nn.relu(tf.matmul(maxpool4_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([500, 7*7*64])
b_fc2 = bias_variable([7*7*64])
predecoded_flat = tf.nn.relu(tf.matmul(encoded, W_fc2) + b_fc2)
predecoded = tf.reshape(predecoded_flat, [-1,7,7,64])

### Decoder
upsample1 = tf.image.resize_images(predecoded, size=(13,13), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 13x13x64
conv5 = tf.layers.conv2d(inputs=upsample1, filters=64, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 13x13x48
upsample2 = tf.image.resize_images(conv5, size=(25,25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 25x25x48
conv6 = tf.layers.conv2d(inputs=upsample2, filters=48, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 25x25x32
upsample3 = tf.image.resize_images(conv6, size=(50,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 50x50x32
conv7 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 50x50x16
upsample4 = tf.image.resize_images(conv7, size=(100,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 100x100x16
conv8 = tf.layers.conv2d(inputs=upsample4, filters=16, kernel_size=(5,5), padding='same', activation=tf.nn.relu)
# Now 100x100x16

logits = tf.layers.conv2d(inputs=conv8, filters=1, kernel_size=(5,5), padding='same', activation=None)
#Now 100x100x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)



print("made it here! :D")
sess = tf.Session()
epochs = 2000000
batch_size = 200
IMAGE_SAVE_EVERY = 2
MODEL_SAVE_EVERY = 50
SAVE_FILE_START_POINT = 200

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

if SAVE_FILE_START_POINT >= 1:
    saver.restore(sess,  "/media/rob/Seagate Backup Plus Drive/characters/models/model"+str(SAVE_FILE_START_POINT)+".ckpt")

print("about to start...")
for e in range(SAVE_FILE_START_POINT, epochs):

    batch_data_in = np.empty([0,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    batch_data_out = np.empty([0,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    while batch_data_in.shape[0] < batch_size:
        imageIndex = int(math.floor(random.randrange(14127)))
        strIndex = str(imageIndex)

        imagio_in = misc.imread('/media/rob/Seagate Backup Plus Drive/characters/traditional/'+strIndex+'.png')
        elementToAdd_in = imagio_in[:,:,0]/255.0
        fixedElementToAdd_in = np.asarray(elementToAdd_in).reshape(1,IMAGE_HEIGHT,IMAGE_WIDTH,1)
        

        imagio_out = misc.imread('/media/rob/Seagate Backup Plus Drive/characters/simplified/'+strIndex+'.png')
        elementToAdd_out = imagio_out[:,:,0]/255.0
        fixedElementToAdd_out = np.asarray(elementToAdd_out).reshape(1,IMAGE_HEIGHT,IMAGE_WIDTH,1)
        
        if np.sum(1-fixedElementToAdd_in) >= 1 and np.sum(1-fixedElementToAdd_out) >= 1 and np.sum(np.absolute(np.subtract(fixedElementToAdd_in,fixedElementToAdd_out))) >= 0.1:
            batch_data_in = np.vstack((batch_data_in,fixedElementToAdd_in))
            batch_data_out = np.vstack((batch_data_out,fixedElementToAdd_out))

    print("HOLLOW. WALLCOME TO EPIC "+str(e)+".")
    print(batch_data_in.shape)
    batch_cost, _, _logits, output = sess.run([cost, opt, logits, decoded], feed_dict={inputs_: batch_data_in,
                                                         targets_: batch_data_out})

    print("Epoch: {}/{}...".format(e+1, epochs), "Training loss: {:.4f}".format(batch_cost))
    
    if (e+1)%IMAGE_SAVE_EVERY == 0:
        exampleImage = np.empty([IMAGE_HEIGHT,IMAGE_WIDTH*3,3])
        exampleImage[0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:3] = batch_data_in[0,0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:1]
        exampleImage[0:IMAGE_HEIGHT,IMAGE_WIDTH:IMAGE_WIDTH*2,0:3] = output[0,0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:1]
        exampleImage[0:IMAGE_HEIGHT,IMAGE_WIDTH*2:IMAGE_WIDTH*3,0:3] = batch_data_out[0,0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:1]
        exampleImage = np.clip(exampleImage, 0, 1)
        misc.imsave('/media/rob/Seagate Backup Plus Drive/characters/outputTest/output'+str(e+1)+'.png',exampleImage)

    if (e+1)%MODEL_SAVE_EVERY == 0:
        save_path = saver.save(sess, "/media/rob/Seagate Backup Plus Drive/characters/models/model"+str(e+1)+".ckpt")
        print("MODEL SAVED, BRO: "+str(save_path))
