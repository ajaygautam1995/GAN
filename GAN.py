import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, \
  BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import torch 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using{} device'.format(device))
# Loading the dataset using keras 
data = tf.keras.datasets.mnist
# touple data 
(x_train, y_train), (x_test, y_test) = data.load_data()

# map inputs to (-1, +1) for better training
x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 - 1
#print("x_train.shape:", x_train.shape)
# Flattening the data
N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)
# Dimensionality of the latent space
lat_dim = 100
# Defining the Generator 
def gen(lat_dim):
    i_p = Input(shape=(lat_dim,))
    h_l = Dense(128, activation=LeakyReLU(alpha=0.2))(i_p)
    h_l = BatchNormalization(momentum=0.7)( h_l)
    h_l = Dense(256, activation=LeakyReLU(alpha=0.2))( h_l)
    h_l = BatchNormalization(momentum=0.7)( h_l)
    h_l = Dense(512, activation=LeakyReLU(alpha=0.2))( h_l)
    h_l = BatchNormalization(momentum=0.7)( h_l)
    h_l = Dense(1024, activation=LeakyReLU(alpha=0.2))( h_l)
    h_l = BatchNormalization(momentum=0.7)( h_l)
    h_l = Dense(D, activation='tanh')( h_l)

    model = Model(i_p, h_l)
    return model
# Defining the Discriminator
def disc(img_size):
    i_p = Input(shape=(img_size,))
    h_l = Dense(1024, activation=LeakyReLU(alpha=0.2))(i_p)
    h_l = Dense(512, activation=LeakyReLU(alpha=0.2))( h_l)
    h_l = Dense(256, activation=LeakyReLU(alpha=0.2))( h_l)
    h_l = Dense(1, activation='sigmoid')( h_l)
    model = Model(i_p,  h_l)
    return model
# Combinig the Discriminator and Generator for training
'''for building the discriminator and compile'''
discriminator = disc(D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy'])

'''for building the generator and compile'''
generator = gen(lat_dim)

# input to represent noise sample from latent space
p = Input(shape=(lat_dim,))

# To get the image passing the noise
im = generator(p)

discriminator.trainable = False # only generator will be trained

false_pred = discriminator(im)  #fake Pridiction

# Create the combined model object and compile the combined model
cmb_model = Model(p, false_pred)
cmb_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
'''Training the GAN'''
# Configuration 
batch_size = 256
epochs =10000    # here we training the discriminator and generator only for 10000epochs
sample_period = 200 # for every `sample period` steps generate and save some data


# Creating the  batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

#  losses
disc_losses = [] # Discriminator Loss
gen_losses = []  # Generator loss

# folder to store the generate the image 
if not os.path.exists('GAN_IMAGE'):
    
    os.makedirs('GAN_IMAGE')
# Defining a  function to generate a grid of random samples from the generator
def sam_img(epoch):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, lat_dim)
    imgs = generator.predict(noise)

  # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        
        for j in range(cols):
            axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
            axs[i,j].axis('off')
            idx += 1
    fig.savefig("GAN_IMAGE/%d.png" % epoch)     # sample image from the generator and save into a file
    plt.close()
"""TRAINING"""
for epoch in range(epochs):
  # Selecting a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_img = x_train[idx]
  
  # Generation of  fake/false images
    noise = np.random.randn(batch_size, lat_dim)
    false_img = generator.predict(noise)
  
  # Training of discriminator
    disc_loss_real, disc_acc_real = discriminator.train_on_batch(real_img, ones)
    disc_loss_false, disc_acc_false = discriminator.train_on_batch(false_img, zeros)
    disc_loss = 0.5 * (disc_loss_real + disc_loss_false)
    disc_acc  = 0.5 * (disc_acc_real + disc_acc_false)
  
  # Training of Generator
    noise = np.random.randn(batch_size, lat_dim)
    gen_loss = cmb_model.train_on_batch(noise, ones)
  
    noise = np.random.randn(batch_size, lat_dim)
    gen_loss = cmb_model.train_on_batch(noise, ones)
  
  # losses
    disc_losses.append(disc_loss)
    gen_losses.append(gen_loss)
  
    if epoch % 100 == 0:
        
        print(f"epoch: {epoch+1}/{epochs}, d_loss: {disc_loss:.2f}, \
          d_acc: {disc_acc:.2f}, g_loss: {gen_loss:.2f}")
  
    if epoch % sample_period == 0:
        sam_img(epoch)
plt.plot(gen_loss, label='g_losses')
plt.plot(disc_loss, label='d_losses')
plt.show()
from skimage.io import imread
#visualizing some of the generated image 
img = imread('gan_images/0.png')
plt.imshow(img)
plt.show()
img = imread('GAN_IMAGE/1000.png')
plt.imshow(img)
plt.show()
img = imread('GAN_IMAGE/8000.png')
plt.imshow(img)
plt.show()
        