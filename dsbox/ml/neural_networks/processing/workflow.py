import importlib

from dsbox.ml.neural_networks import KerasFactory
from dsbox.ml.neural_networks.keras_factory.image_models import KerasApplicationFactory
from dsbox.ml.neural_networks.processing import Text2Sequence
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class TextNeuralNetPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline used to fit/predict raw text in a classification problem context. It transforms the text data
    via Text2Sequence transformer and passes the result to a keras model.
    
    Parameters
    ----------
    text2seq : object (default=Text2Sequence())
        transformer used to transform text into integer sequences.
        
    factory_class : KerasFactory (default=KerasFactory)
        set the factory used to build the keras model. By default, it takes an abstract class, which has to 
        be replaced by a concrete one (LSTMFactory or CNN_LSTMFactory for instance).
        
    num_labels : int, optional (default=2)
        amount of class labels
        
    Attributes
    ----------
    model_ : KerasClassifier
        keras scikit wrapper containing original keras architecture model.
    
            
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42) # for making doctests stable
    >>> from dsbox.ml.neural_networks.keras_factory.text_models import LSTMFactory
    >>> from dsbox.ml.neural_networks.processing.workflow import TextNeuralNetPipeline
    
    >>> x_train = np.array([u'this is really really awesome !', \
                            u'it is so awesome !', \
                            u'that sucks'] \
                           )
    >>> y_train = np.array([1, 1, 0])

    >>> model = TextNeuralNetPipeline(factory_class=LSTMFactory, num_labels=2, random_state=42)
    >>> _ = model.fit(x_train, y_train, verbose=0) # to avoid doctests error

    >>> x_test = np.array([u'it is really awesome !'])
    >>> model.predict(x_test)
    array([1])
    
    """

    def __init__(self, text2seq=Text2Sequence(), factory_class=KerasFactory, num_labels=2, random_state=42):
        self.text2seq = text2seq
        self.factory = factory_class()
        self.num_labels = num_labels
        self.random_state = random_state

        self.attr_to_check = ["model_"]

    def fit(self, X, y, **kwargs):
        """
        Fit the workflow by building the word corpus, and fitting the keras model.
    
        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str
        y : array-like, shape (n_samples,)
            Class targets.
        **kwargs : 
            parameters passed to inner keras model
    
        Returns
        -------
        self : object
            Returns an instance of self.
        """

        x = self.text2seq.fit_transform(X)
        y_enc = to_categorical(y, self.num_labels)

        self.model_ = KerasClassifier(build_fn=self.factory.create_model,
                                      dictionary_size=self.text2seq.dictionary_size_,
                                      num_labels=self.num_labels)

        self.model_.fit(x, y_enc, **kwargs)

        return self

    def predict(self, X):
        """
        Predict a list of texts to belong to a known class.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        numpy array of class ids.

        """
        check_is_fitted(self, self.attr_to_check)

        x = self.text2seq.transform(X)
        return self.model_.predict(x)

    def predict_proba(self, X):
        """
        Predict a list of texts to belong to a known class.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        probability of belonging to each class id.
        """

        check_is_fitted(self, self.attr_to_check)

        x = self.text2seq.transform(X)
        return self.model_.predict_proba(x)


class ImageNeuralNetPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline used to predict an image class. It uses pre-trained models proposed by Keras applications package.

    Parameters
        ----------
        module_name: str, optional (default='tensorflow.keras.applications.xception')
            specify keras application module name to import model
        model_name: str, optional (default='Xception')
            specify keras application model name to use
        top_pred: int, optional (default=1)
            parameter used to select amount of image categories returned by the predict method
        **kwargs :
            parameters passed to inner keras model

    .. Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/
    """
    def __init__(self,
                 module_name='tensorflow.keras.applications.xception', model_name='Xception',
                 top_pred=1,
                 **kwargs
                 ):
        self.factory = KerasApplicationFactory()
        self.module_name = module_name
        self.model_name = model_name
        self.top_pred = top_pred

        self.preprocess_input = getattr(importlib.import_module(module_name), 'preprocess_input')
        self.decode_predictions = getattr(importlib.import_module(module_name), 'decode_predictions')

        self.last_conv_layer_name_ = None
        self.classifier_layer_names_ = None
        self.img_size_ = None

        self.attr_to_check = ["model_",
                              "last_conv_layer_name_",
                              "classifier_layer_names_"]
        self.kwargs = kwargs

    def _get_img_array(self, img_path, size):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def _make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, classifier_layer_names):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    def get_heatmap(self, x):
        """
        Return a matrix representing the heatmap of the image to understand model prediction assumptions.
        Parameters
        ----------
        x: str
            path to image file

        Returns
        -------
        NumPy heatmap matrix

        """
        check_is_fitted(self, self.attr_to_check)

        img_array = self.preprocess_input(self._get_img_array(x, size=self.img_size_))
        return self._make_gradcam_heatmap(img_array, self.model_, self.last_conv_layer_name_,
                                                   self.classifier_layer_names_)

    def fit(self, X=None, y=None, **kwargs):
        """
        Build pre-trained model by extracting layers information.

        Parameters
        ----------
        X : not used, present for API consistence purpose
        y : not used, present for API consistence purpose
        **kwargs :
            parameters passed to inner keras model

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        self.model_ = self.factory.create_model(module_name=self.module_name,
                                                model_name=self.model_name,
                                                **self.kwargs)

        input_size = self.model_.get_config()['layers'][0]['config']['batch_input_shape']
        self.img_size_ = (input_size[1], input_size[2])

        layers = self.model_.get_config()['layers']
        last_conv_layer_name = None
        last_conv_layer_index = -1
        i = len(layers) - 1
        while last_conv_layer_name is None and i >= 0:
            if 'conv' in layers[i]['name']:
                last_conv_layer_name = layers[i]['name']
                last_conv_layer_index = i
            i -= 1

        self.last_conv_layer_name_ = last_conv_layer_name

        classifier_layer_names = []
        for i in range(last_conv_layer_index + 1, len(layers)):
            classifier_layer_names.append(layers[i]['name'])

        self.classifier_layer_names_ = classifier_layer_names

        return self

    def predict(self, X):
        """
        Predict image tags.

        Parameters
        ----------
        X : array-like, iterable
            Collection of image paths or an iterable which yields image path

        Returns
        -------
        numpy array of predicted image label with probability

        """
        check_is_fitted(self, self.attr_to_check)
        predictions = []
        for x in X:
            img_array = self.preprocess_input(self._get_img_array(x, size=self.img_size_))
            preds = self.model_.predict(img_array)
            predictions.append(self.decode_predictions(preds, top=self.top_pred)[0])

        return predictions
