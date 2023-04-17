from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from Client import *

class Model():

    def __init__(self) -> None:
        self.shape=0

    def build_model(self,shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

        
    def scale_weights(self, weight,client:Client):
        '''function for scaling a models weights'''
        global_count=42000
        scalar=client.size/global_count
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final


    def avg_weights(self,scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad


    def evaluation(self,X_test, Y_test,  model, comm_round):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(X_test)
        loss = cce(Y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
        #print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
        return acc, loss