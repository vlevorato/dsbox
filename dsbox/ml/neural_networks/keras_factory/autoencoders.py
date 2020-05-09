import numpy as np
from dsbox.ml.neural_networks import KerasFactory
from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from sklearn.base import BaseEstimator, ClassifierMixin

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"

__source__ = "https://blog.keras.io/building-autoencoders-in-keras.html"


class SimpleAutoEncoderFactory(KerasFactory):
    def create_model(self, encoding_dim=32, input_size=784):
        """
        Build a simple keras model autoencoder with 3 dense layers.
        
        Mostly used for data denoising or outliers detection.
        
        Parameters
        ----------
        encoding_dim: int, optional (default=32)
            intermediary layer size
            
        input_size: int, optional (default=784)
            original input data size

        Returns
        -------
        3 Keras models (autoencoder, encoder, decoder)
        """
        input_layer = Input(shape=(input_size,))
        encoded_layer = Dense(encoding_dim, activation='relu')(input_layer)
        decoded_layer = Dense(input_size, activation='sigmoid')(encoded_layer)

        autoencoder = Model(input_layer, decoded_layer)
        encoder = Model(input_layer, encoded_layer)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder, decoder


class SparseAutoEncoderFactory(KerasFactory):
    def create_model(self, encoding_dim=32, input_size=784):
        """
        Build a sparse keras model autoencoder with 3 dense layers.
        It adds a regularization constraint in the intermediary layer.
        
        Mostly used for data denoising or outliers detection.
        
        Parameters
        ----------
        encoding_dim: int, optional (default=32)
           intermediary layer size
        
        input_size: int, optional (default=784)
           original input data size
        
        Returns
        -------
        3 Keras models (autoencoder, encoder, decoder)
        """
        input_layer = Input(shape=(input_size,))
        encoded_layer = Dense(encoding_dim, activation='relu',
                              activity_regularizer=regularizers.l1(10e-5))(input_layer)
        decoded_layer = Dense(input_size, activation='sigmoid')(encoded_layer)

        autoencoder = Model(input_layer, decoded_layer)
        encoder = Model(input_layer, encoded_layer)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder, decoder


class DeepAutoEncoderFactory(KerasFactory):
    def create_model(self, input_size=784, start_layer_size=128, sub_layers_level=3):
        """
        Build a deep keras model autoencoder with x dense layers. For each sub-layer,
        the start_layer size is divided by 2.
        With default parameters (start_layer_size=128, sub_layers_level=3), it builds an 
        architecture like this:
        input_size-128-64-32-64-128-input_size
        
        Mostly used for data denoising or outliers detection.
        
        Parameters
        ---------- 
        input_size: int, optional (default=784)
           original input data size
           
        start_layer_size: int, optional (default=128)
            size of the first layer after the input one
            
        sub_layers_level: int, optional (default=3)
            amount of sub-levels to build
        
        Returns
        -------
        2 Keras models (autoencoder, encoder)
        """
        input_layer = Input(shape=(input_size,))
        encoded_layer = Dense(start_layer_size, activation='relu')(input_layer)
        next_layer_size = int(start_layer_size)
        for i in range(1, sub_layers_level):
            next_layer_size = int(next_layer_size / 2)
            encoded_layer = Dense(next_layer_size, activation='relu')(encoded_layer)

        next_layer_size = next_layer_size * 2
        decoded_layer = Dense(next_layer_size, activation='relu')(encoded_layer)

        for i in range(2, sub_layers_level):
            next_layer_size = next_layer_size * 2
            decoded_layer = Dense(next_layer_size, activation='relu')(decoded_layer)

        decoded_layer = Dense(input_size, activation='sigmoid')(decoded_layer)

        autoencoder = Model(input_layer, decoded_layer)
        encoder = Model(input_layer, encoded_layer)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder


class CNNAutoEncoder2DFactory(KerasFactory):
    def create_model(self, shape=(28, 28, 1)):
        """
        Build a 2D convolutional keras model autoencoder.

        Mostly used for image denoising or image outliers detection.

        Parameters
        ----------           
        shape: int, optional (default=(28,28,1))
           original input data size

        Returns
        -------
        2 Keras models (autoencoder, encoder)
        """
        layer_size = np.int(np.round((np.sqrt(shape[0] * shape[1]) * 0.1)))
        pool_size = np.int(np.round(layer_size * 2 / 3.0))

        input_img = Input(shape=shape)

        x = Conv2D(16, (layer_size, layer_size), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
        x = Conv2D(8, (layer_size, layer_size), activation='relu', padding='same')(x)
        x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
        x = Conv2D(8, (layer_size, layer_size), activation='relu', padding='same')(x)
        encoded_layer = MaxPooling2D((pool_size, pool_size), padding='same')(x)

        x = Conv2D(8, (layer_size, layer_size), activation='relu', padding='same')(encoded_layer)
        x = UpSampling2D((pool_size, pool_size))(x)
        x = Conv2D(8, (layer_size, layer_size), activation='relu', padding='same')(x)
        x = UpSampling2D((pool_size, pool_size))(x)
        x = Conv2D(16, (layer_size, layer_size), activation='relu')(x)
        x = UpSampling2D((pool_size, pool_size))(x)
        decoded_layer = Conv2D(1, (layer_size, layer_size), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded_layer)
        encoder = Model(input_img, encoded_layer)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder


class CNNDenoisingAutoEncoder2DFactory(KerasFactory):
    def create_model(self, shape=(28, 28, 1)):
        """
        Build a 2D convolutional keras model autoencoder specialized in image denoising.

        Parameters
        ----------           
        shape: int, optional (default=(28,28,1))
           original input data size

        Returns
        -------
        2 Keras models (autoencoder, encoder)
        """
        layer_size = np.int(np.round((np.sqrt(shape[0] * shape[1]) * 0.1)))
        pool_size = np.int(np.round(layer_size * 2 / 3.0))

        input_img = Input(shape=shape)

        x = Conv2D(32, (layer_size, layer_size), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((pool_size, pool_size), padding='same')(x)
        x = Conv2D(32, (layer_size, layer_size), activation='relu', padding='same')(x)
        encoded_layer = MaxPooling2D((pool_size, pool_size), padding='same')(x)

        x = Conv2D(32, (layer_size, layer_size), activation='relu', padding='same')(encoded_layer)
        x = UpSampling2D((pool_size, pool_size))(x)
        x = Conv2D(32, (layer_size, layer_size), activation='relu', padding='same')(x)
        x = UpSampling2D((pool_size, pool_size))(x)
        decoded_layer = Conv2D(1, (layer_size, layer_size), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded_layer)
        encoder = Model(input_img, encoded_layer)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder


class VariationalAutoEncoderFactory(KerasFactory):
    def create_model(self, input_size=784, intermediate_dim=256, latent_dim=16, epsilon_std=1.0, vae_loss='xent'):
        """
        Build a keras model variational autoencoder.
        
        First, an encoder network turns the input samples x into two parameters in a latent space. Then, we randomly 
        sample similar points z from the latent normal distribution that is assumed to generate the data, 
        via z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor. Finally, a decoder 
        network maps these latent space points back to the original input data.

        The parameters of the model are trained via two loss functions: a reconstruction loss forcing the decoded 
        samples to match the initial inputs (just like in our previous autoencoders), and the KL divergence between the
        learned latent distribution and the prior distribution, acting as a regularization term. You could actually get 
        rid of this latter term entirely, although it does help in learning well-formed latent spaces and reducing 
        overfitting to the training data.
        
        Parameters
        ----------  
        input_size: int, optional (default=784)
           original input data size
           
        intermediate_dim: int, optional (default=16)
           size of the dense layers, placed after the input layer and before the output layer
           
        latent_dim: int, optional (default=16)
           size of the intermediary layer
           
        epsilon_std, float, optional (default=1.0)
            std value used for the normal distribution used to generate data into the latent space
            
        vae_loss: str, {'mse', 'xent'} optional, default 'xent'
            loss function used for reconstruction, MSE or binary Cross Entropy
        
        Returns
        -------
        3 Keras models (autoencoder, encoder, decoder)
        """
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

        input_shape = (input_size,)
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_size, activation='sigmoid')(x)

        decoder = Model(latent_inputs, outputs, name='decoder')

        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = None
        if vae_loss == 'mse':
            reconstruction_loss = mse(inputs, outputs)
        if vae_loss == 'xent':
            reconstruction_loss = binary_crossentropy(inputs, outputs)

        if reconstruction_loss is None:
            raise (ValueError("vae_loss parameter should take a valid value, 'mse' or 'xent'."))

        reconstruction_loss *= input_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        return vae, encoder, decoder

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


class AutoEncoderClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper used to be scikit compliant, as our autoencoder factories returns more than one model.
    """

    def __init__(self, keras_autoencoder, keras_encoder):
        self.autoencoder = keras_autoencoder
        self.encoder = keras_encoder

    def fit(self, X, y, **kwargs):
        return self.autoencoder.fit(X, y, **kwargs)

    def predict(self, X):
        self.encoded_data_ = self.encoder.predict(X)
        return self.autoencoder.predict(X)
