# Keras and Tensorflow
import keras

from tensorflow import keras, nn
from tensorflow.keras.applications import VGG16, EfficientNetB3, ResNet50

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LeakyReLU, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model, load_model

'''VGG model'''
class VGG(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the VGG model
        vgg_conv = VGG16(weights='imagenet', input_shape=(224,224,3))
        self.VGG_model = Sequential()
        for layer in vgg_conv.layers[:-1]: # excluding last layer from copying
            self.VGG_model.add(layer)
                
    def freeze(self):
        self.VGG_model.trainable = False

    def predict(self, images):
        features = self.VGG_model.predict(images, verbose=1)

        return features



'''EfficientNet B3 model'''
class EffNetB3(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.EfficientNetB3_model = EfficientNetB3(weights='imagenet',
                                include_top = False, 
                                input_shape=(300, 300, 3),
                                pooling = 'avg')
                
    def freeze(self):
        self.EfficientNetB3_model.trainable = False

    def predict(self, images):
        features = self.EfficientNetB3_model.predict(images, verbose=1)

        return features



'''ResNet50 Model'''
class ResNet(keras.Model):
    def __init__(self):
        super().__init__()

        # creating the Inception V3 model
        self.ResNet_model = ResNet50(weights='imagenet', 
                                    input_shape=(512,512,3), 
                                    include_top = False, 
                                    pooling = 'avg')
                
    def freeze(self):
        self.ResNet_model.trainable = False

    def predict(self, images):
        features = self.ResNet_model.predict(images, verbose=1)

        return features



'''Keras model based Encoder'''
class encoder:
    def __init__(self,mode):
        if mode=="VGG":
            self.k=8
            self.yoname="VGG_256.h5"
            
        elif mode=="Resnet":
            self.k=4
            self.yoname="Resnet50_256.h5"
        else :
            self.k=3
            self.yoname="EfficientB3_256.h5"

    def encode(self,X):
        t=MinMaxScaler()
        t.fit(X)
        self.X=t.transform(X)
        n_inputs = self.X.shape[1]
        self.input_data_shape= Input(shape=(n_inputs,))
        self.encoder= Dense(n_inputs)(self.input_data_shape)
        self.encoder= BatchNormalization()(self.encoder)
        self.encoder= LeakyReLU()(self.encoder)
        self.encoder= Dense(n_inputs/self.k)(self.encoder)
        self.encoder= BatchNormalization()(self.encoder)
        self.encoder= LeakyReLU()(self.encoder)
        n_bottleneck = round(float(n_inputs) / (2*self.k))
        self.bottleneck = Dense(n_bottleneck)(self.encoder)
        output = Dense(n_inputs, activation='linear')(self.encoder)
        self.enmodel = Model(inputs=self.input_data_shape,outputs=self.bottleneck)
        self.enmodel.compile(optimizer='adam', loss='mse')

    def encodeTrainer(self):
        self.encoder = Model(inputs=self.input_data_shape, outputs=self.bottleneck)
        Encoded_X=self.encoder.predict(self.X)

        return Encoded_X
        
    def savemodel(self):
        self.encoder.save(self.yoname)
