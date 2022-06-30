import keras.models
from keras.layers import Dense, Dropout, Input, Conv1D, Concatenate, Flatten
from keras.utils.vis_utils import plot_model
import logging

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
#import h5py

from lib.weighted_loss import reweighted_binary_crossentropy


class Model:
    """ Class that encapsulates the machine learning model and related functions. """

    """ Sept. 2021: only keep LogR and MLP for survial predictions """

    def __init__(self, model_type, task, input_dim, encodings, mlp_layers, mlp_width, mlp_dropout, mlp_emb_neurons, logr_c):
        self.model_type = model_type
        if model_type == 'MLP':
            self.model = mlp_model(input_dim=input_dim, width=mlp_width, depth=mlp_layers,
                                   dropout=mlp_dropout)
        elif model_type == 'MLPEmb':
            self.model = mlp_emb_model(input_dim=input_dim, width=mlp_width, depth=mlp_layers, dropout=mlp_dropout,
                                       encodings=encodings, emb_neurons=mlp_emb_neurons)
        elif model_type == 'LogR' and task in ['survival12', 'survival60']:
            self.model = LogisticRegression(C=logr_c, max_iter=1500)
        elif model_type == 'NAIVE':
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            logging.error('Invalid model.')
            exit(-1)

    def plot_model(self, output_directory):
        if self.model_type in ['MLP', 'MLPEmb']:
            plot_model(self.model, to_file=output_directory + 'model.png')

def mlp_model(input_dim, width, depth, dropout):
    """ Function to create the MLP model. """
    model = keras.models.Sequential()

    for i in range(0, depth):
        model.add(Dense(units=width, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    """
    loss = weighted_binary_crossentropy(weights)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    """
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def mlp_model_reweight(input_dim, width, depth, dropout, weight_vector):
    """ Function to create the MLP model with reweighting. """
    model = keras.models.Sequential()

    for i in range(0, depth):
        model.add(Dense(units=width, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))


    loss = reweighted_binary_crossentropy(weight_vector)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    return model


def mlp_emb_model(input_dim, width, depth, dropout, encodings, emb_neurons):
    """ Function to create MLP model with embedding layer for encoded inputs. """

    if input_dim != sum(encodings.values()):
        logging.error("Bad encoding: " + str(input_dim) + " vs. " + str(sum(encodings.values())))
        exit(1)

    # Embedding layer per encoding
    embeddings = []
    inputs = []
    for encoding in encodings.values():
        input_segment = Input(shape=(encoding, 1))
        embeddings.append(Dropout(dropout)(Conv1D(emb_neurons, encoding)(input_segment)))
        inputs.append(input_segment)
    tensors = Concatenate(axis=-1)(embeddings)
    tensors = Flatten()(tensors)

    # Additional feedforward layers
    for i in range(0, depth - 1):
        tensors = Dense(width, kernel_initializer='normal', activation='relu')(tensors)
        tensors = Dropout(dropout)(tensors)

    # Output layer

    predictions = Dense(1, kernel_initializer='normal', activation='sigmoid')(tensors)

    model = keras.models.Model(inputs=inputs, outputs=predictions)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
