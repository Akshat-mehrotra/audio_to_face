from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization, Dropout
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img

from os import listdir

from numpy import asarray, ones, zeros
from numpy.random import randint

from PIL import Image

from matplotlib import pyplot as plt


INPUT_SHAPE = (256, 256, 3)
DATA_DIR = 'train'

data = asarray(listdir(DATA_DIR))
n_data = len(data) - 1
print(n_data)
def discriminator(img_shape):
	src_img = Input(img_shape)
	target_img = Input(img_shape)
	merged = Concatenate()([src_img, target_img])

	s = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(merged) # OUT -> (128, 128, 32)
	s = LeakyReLU(0.2)(s)

	s = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(s) # OUT -> (64, 64, 128)
	s = BatchNormalization()(s)
	s = LeakyReLU(0.2)(s)

	s = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(s) # OUT -> (32, 32, 256)
	s = BatchNormalization()(s)
	s = LeakyReLU(0.2)(s)

	s = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(s) # OUT -> (16, 16, 512)
	s = BatchNormalization()(s)
	s = LeakyReLU(0.2)(s)

	s = Conv2D(512, (4,4), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(s) # OUT -> (16, 16, 512)
	s = BatchNormalization()(s)
	s = LeakyReLU(0.2)(s)

	s = Conv2D(1, (4,4), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(s) # OUT -> (16, 16, 512)
	s = Activation('sigmoid')(s)

	model = Model([src_img, target_img], s)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model


def encoder(prev_layer, filters, batchnorm=True):
	s = Conv2D(filters, (4,4), padding='same', strides=(2,2), kernel_initializer=RandomNormal(stddev=0.02))(prev_layer)
	if batchnorm:
		s = BatchNormalization()(s, training=True)
	s = LeakyReLU(.2)(s)
	return s


def decoder(prev_layer, encoder_skip, filters, dropout=True):
	s = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(prev_layer)
	s = BatchNormalization()(s, training=True)
	if dropout:
		s = Dropout(.2)(s, training=True)
	s = LeakyReLU(.2)(s)
	s = Concatenate()([s, encoder_skip])
	return s


def generator(img_shape):
	in_img = Input(shape=img_shape)
	e1 = encoder(in_img, 64, batchnorm=False)
	e2 = encoder(e1, 128)
	e3 = encoder(e2, 256)
	e4 = encoder(e3, 512)
	e5 = encoder(e4, 512)
	e6 = encoder(e5, 512)
	e7 = encoder(e6, 512)

	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(e7)
	b = LeakyReLU()(b)

	d1 = decoder(b, e7, 512)
	d2 = decoder(d1, e6, 512)
	d3 = decoder(d2, e5, 512)
	d4 = decoder(d3, e4, 512, dropout=False)
	d5 = decoder(d4, e3, 256, dropout=False)
	d6 = decoder(d5, e2, 128, dropout=False)
	d7 = decoder(d6, e1, 64, dropout=False)

	out = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d7)
	out = Activation('tanh')(out)

	model = Model(in_img, out)
	return model


def gan(gen, dis, img_shape):
	opt = Adam(lr=0.0002, beta_1=0.5)

	dis.trainable = False
	in_img = Input(shape=img_shape)

	gx = gen(in_img)
	dx = dis([in_img, gx])
	model = Model(in_img, [dx, gx])

	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model


def real_nextsamples(n_samples):
	i = randint(0, n_data+1, n_samples)
	X_A, X_I = list(), list()

	for j in i:
		X_audio = Image.open(f'{DATA_DIR}/{j}/audio.jpg')
		X_img = Image.open(f'{DATA_DIR}/{j}/image.jpg')

		X_audio = X_audio.resize(INPUT_SHAPE[:-1])
		X_img = X_img.resize(INPUT_SHAPE[:-1])

		X_A.append(asarray(X_audio))
		X_I.append(asarray(X_img))

	X_A, X_I = asarray(X_A), asarray(X_I)

	X_A = (X_A - 127.5) / 127.5
	X_I = (X_I - 127.5) / 127.5
	return X_A, X_I


def summarise(epoch, gen, loss, n_samples=3):
	X_realaudio, X_realimgs = real_nextsamples(n_samples)
	X_fakeimgs = gen.predict(X_realaudio)

	# scale to [0,1]
	X_realaudio = (X_realaudio + 1) / 2.0
	X_realimgs = (X_realimgs + 1) / 2.0
	X_fakeimgs = (X_fakeimgs + 1) / 2.0

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realaudio[i])

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeimgs[i])

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realimgs[i])

	pyplot.savefig(f'epoch {epoch+1}')
	pyplot.close()

	gen.save(f'model_{epoch+1}_{loss}.h5')
	print(f">Saved model_{epoch+1}_{loss}.h5")



def train(dis, gen, gan, n_imgs, epochs=100, n_batches=55):
	imgs_per_batch = int(n_imgs//n_batches)
	y_real = ones((imgs_per_batch, dis.output_shape[1], dis.output_shape[2], 1))
	y_fake = zeros((imgs_per_batch, dis.output_shape[1], dis.output_shape[2], 1))
	for epoch in range(epochs):
		for batch in range(n_batches):
			X_realaudio, X_realimgs = real_nextsamples(imgs_per_batch)
			X_fakeimgs = gen.predict(X_realaudio)

			dis_fake_loss = dis.train_on_batch([X_realaudio, X_fakeimgs], y_fake)
			dis_real_loss = dis.train_on_batch([X_realaudio, X_realimgs], y_real)

			gan_loss = gan.train_on_batch(X_realaudio, [y_real, X_realimgs])

			print(f"\tbatch no->{batch+1}")
		print(f"{epoch+1}: gan->{gan_loss}, dis_fake->{dis_fake_loss}, dis_real->{dis_real_loss}")

		if (epoch+1) % 5 == 0:
			summarise(epoch, gen, gan_loss)

dis = discriminator(INPUT_SHAPE)
gen = generator(INPUT_SHAPE)
gan = gan(gen, dis, INPUT_SHAPE)
train(dis, gen, gan, n_imgs=n_data, n_batches=int(n_data/5))
