'''This file defines a RFNN class, which provides functionality to create an RF
Neural Network. Layers of different types can be added, each with their own set of
parameters. The goal is to provide a framework to quickly experiment with different
neural network architectures'''

# Define imports
import tensorflow as tf
import numpy as np
import sklearn.metrics as sk

# Class to define a Neural Network that can work on RF input data.
# This data is expected to have a 2D input - the data itself can be of any specified type/form (e.g. time/freq)
# The output of the data is expected to be an output class label of any specified type
class RFNN:
    # Define constructor
    def __init__(self, inputDim, numOutputClasses):
        # Error checking on inputs
        assert(len(inputDim) == 2)

        # Assign these fundamental attributes
        self.inputDim = inputDim
        self.numOutputClasses = numOutputClasses

        # Compute actual input shape we'll reshape to and define tensor shapes
        self.inputShape = inputDim[0] * inputDim[1]
        self.inputTensorShape = [-1, inputDim[0], inputDim[1], 1]

        # Define top level input and output tensors of the network
        self.x = tf.placeholder(tf.float32, shape=[None, self.inputShape])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.numOutputClasses])
        self.y_out = []

        # Reshape the input to have four dimensions (to process in TF)
        self.x_in = tf.reshape(self.x, self.inputTensorShape)

        # Initialize list of layers in this network
        self.layers = []
        self.numLayers = 0

        # Define probability of keep (for dropout during training and/or testing)
        self.dropout = False
        self.keep_prob = tf.placeholder(tf.float32) # the actual tensor
        self.keep_prob_val = 0.0                    # value assigned to tensor

        # Default initializations
        self.min_fcn = ''
        self.minFunction = []
        self.train_step = []
        self.correct_prediction = []
        self.accuracy = []
        self.predictions = []

        # Initialize containers to hold predicted & truth labels
        self.y_pred = []
        self.y_true = []

        # Initialize confusion matrices
        self.cm = []
        self.norm_cm = []

        # Initialize lists that hold the batch counter and training accuracy
        self.batches = []
        self.train_accs = []

        # Default TensorFlow session
        self.sess = tf.Session()

    # This function sets up our function to be minimized, and sets up tensors
    # for training, testing and evaluation.
    # TODO: add more types of functions that we can minimize as we need to
    def prepare(self, min_fcn='cross_entropy'):
        # Define our local output tensor from the final layer
        self.y_out = self.layers[self.numLayers - 1].outputTensor

        # Assign function we will use for minimization
        self.min_fcn = min_fcn

        # For now, we only support cross entropy
        assert(self.min_fcn == 'cross_entropy')

        # Define the cross entropy function
        self.minFunction = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out),
                                                         reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.minFunction)
        self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1),
                                           tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                               tf.float32))
        self.predictions = tf.argmax(self.y_out, 1)

        # Finally, initialize our TensorFlow session
        self.sess.run(tf.global_variables_initializer())

    # Function to release tensorflow resources and end session
    def release(self):
        self.sess.close()

    # Function to train our neural network once it has been fully set up
    # The data and labels are passed in via the RFDataReader object
    def train(self, RFData, data_type='AmpPhase', batch_size=50,
              dropout='False', keep_prob_val=0.6):
        # Initialize
        batch_size = batch_size
        batch_ctr = 0

        self.dropout = dropout
        self.keep_prob_val = keep_prob_val

        # Select what kind of data we will process
        if data_type == 'IandQ':
            data = RFData.trainIandQ
            num_training_samples = RFData.trainIandQ.shape[0]
        elif data_type == 'AmpPhase':
            data = RFData.trainAmpPhase
            num_training_samples = RFData.trainAmpPhase.shape[0]
        labels = RFData.trainOutLabels

        # Go through the data in batches
        for idx in range(0, num_training_samples, batch_size):
            # Compute start and end indices
            start_idx = idx
            end_idx = min(idx + batch_size - 1, data.shape[0] - 1)
            local_batch_size = end_idx - start_idx

            # Get data for this training batch
            dataBatch = data[start_idx:end_idx]
            dataBatch = np.reshape(dataBatch, [local_batch_size, self.inputShape])

            # Get labels for this training batch
            labelBatch = np.zeros((local_batch_size, self.numOutputClasses))
            for samp in range(start_idx, end_idx):
                labelBatch[samp - start_idx, int(labels[samp])] = 1

            train_acc = self.accuracy.eval(session=self.sess, feed_dict={self.x: dataBatch,
                                                                         self.y_: labelBatch,
                                                                         self.keep_prob: 1.0})
            
            print('start_idx = ' + str(start_idx) + ':  ' + 'end_idx = ' + str(end_idx))  
            print('Batch number: ' + str(batch_ctr) + ' || Train Acc: ' + str(train_acc))

            # Append to the training vectors
            self.batches.append(batch_ctr)
            self.train_accs.append(train_acc)

            # Execute training step
            if dropout:
                self.train_step.run(session=self.sess, feed_dict={self.x: dataBatch,
                                                                  self.y_: labelBatch,
                                                                  self.keep_prob: self.keep_prob_val})
            else:
                self.train_step.run(session=self.sess, feed_dict={self.x: dataBatch,
                                                                  self.y_: labelBatch})

            # Update batch counter
            batch_ctr += 1

        # Return the final training accuracy
        return train_acc

    # Function to test our neural network on a test set once we have trained
    # The data and labels are passed in via the RFDataReader object
    def test(self, RFData, data_type='AmpPhase'):
        # Select what kind of data we will process
        if data_type == 'IandQ':
            data = RFData.testIandQ
            num_testing_samples = RFData.testIandQ.shape[0]
        elif data_type == 'AmpPhase':
            data = RFData.testAmpPhase
            num_testing_samples = RFData.testAmpPhase.shape[0]

        labels = RFData.testOutLabels

        # Reshape our inputs so we can evaluate
        testData = np.reshape(data, [num_testing_samples, self.inputShape])
        testLabels = np.zeros((num_testing_samples, self.numOutputClasses))
        for samp in range(num_testing_samples):
            testLabels[samp, int(labels[samp])] = 1

        # Evaluate on test data set
        test_acc, y_pred = self.sess.run([self.accuracy, self.predictions],
                                         feed_dict={self.x: testData, self.y_: testLabels,
                                                    self.keep_prob: 1.0})

        # Get actual classification labels
        y_true = np.argmax(testLabels, 1)

        # Store the truth and predictions
        self.y_pred = y_pred
        self.y_true = y_true

        # Create confusion matrices
        self.__create_conf_mtxs()

        # Return the testing accuracy
        return test_acc

    # Function to add a convolutional layer to our Neural Network
    def create_conv_layer(self, num_feature_maps, patch_size,
                          conv_strides=[], conv_padding='', max_pool=[], ksize=[],
                          pool_strides=[], pool_padding=''):

        # Make sure all required fields are provided
        assert(len(patch_size) == 2)

        # Check how many layers are currently in the network, and look up the proper input tensor
        # If we don't have any layers added yet, the input is the original input defined in the constructor
        # If we have layers, our current input will be the output of the last layer
        if not self.numLayers:
            localInputTensor = self.x_in # the first layer we are adding
        else:
            localInputTensor = self.layers[self.numLayers - 1].outputTensor

        # Create a convolutional layer
        layer = ConvLayer(localInputTensor, num_feature_maps, patch_size,
                 conv_strides, conv_padding, max_pool, ksize, pool_strides, pool_padding)

        self.layers.append(layer)
        self.numLayers += 1

    # Function to add a fully connected layer to our Neural Network
    def create_fc_layer(self, num_nodes, dropout='False'):
        # Check how many layers are currently in the network, and look up the proper input tensor
        # If we don't have any layers added yet, the input is the original input defined in the constructor
        # If we have layers, our current input will be the output of the last layer
        if not self.numLayers:
            localInputTensor = self.x_in # the first layer we are adding
        else:
            localInputTensor = self.layers[self.numLayers - 1].outputTensor

        # Create a fully connected layer
        layer = FCLayer(localInputTensor, num_nodes, dropout, self.keep_prob)

        self.layers.append(layer)
        self.numLayers += 1

    # Function to add a readout (softmax) layer to our Neural Network
    def create_readout_layer(self):
        # Check how many layers are currently in the network, and look up the proper input tensor
        # If we don't have any layers added yet, the input is the original input defined in the constructor
        # If we have layers, our current input will be the output of the last layer
        if not self.numLayers:
            localInputTensor = self.x_in # the first layer we are adding
        else:
            localInputTensor = self.layers[self.numLayers - 1].outputTensor

        # Create a readout (softmax) layer
        layer = ReadoutLayer(localInputTensor, self.numOutputClasses)

        self.layers.append(layer)
        self.numLayers += 1

    # Getters for outside access
    def get_train_results(self):
        return self.batches, self.train_accs

    def get_conf_mtxs(self):
        return self.cm, self.norm_cm

    def get_truth_and_predicions(self):
        return self.y_true, self.y_pred

    def __create_conf_mtxs(self):
        # Compute raw confusion matrix (output classes vs. output classes)
        self.cm = sk.confusion_matrix(self.y_true, self.y_pred)

        # Normalize it
        norm_cm = self.__normalize_conf_mtx()
        self.norm_cm = norm_cm

    # Normalize a confusion matrix
    def __normalize_conf_mtx(self):
        norm_cm = []
        for i in self.cm:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                if a != 0:
                    tmp_arr.append(float(j) / float(a))
                else:
                    tmp_arr.append(0)

            norm_cm.append(tmp_arr)

        return np.array(norm_cm)

# define a generic Layer class, with some base attributes
class Layer:
    def __init__(self, layerType, inputTensor):
        assert(layerType == 'Conv' or layerType == 'FC' or layerType == 'Readout')
        self.inputTensor = inputTensor

    # PRIVATE FUNCTION DEFINITIONS BELOW FOR INTERNAL LAYER OPERATIONS #
    def weight_variable(self):
        initial = tf.truncated_normal(self.weight_shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self):
        initial = tf.constant(0.1, shape=self.bias_shape)
        return tf.Variable(initial)

class ConvLayer(Layer):
    # Constructor for ConvLayer class
    def __init__(self, inputTensor, num_feature_maps, patch_size,
                 conv_strides=[], conv_padding='', max_pool=False, ksize=[], pool_strides=[],
                 pool_padding=''):
        # Call base class constructor
        Layer.__init__(self, 'Conv', inputTensor)

        # Initialize Conv layer parameters
        self.num_feature_maps = num_feature_maps
        self.patch_size = patch_size

        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.max_pool = max_pool
        self.ksize = ksize
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding

        # Set default parameters if necessary
        if not self.conv_strides:
            self.conv_strides = [1, 1, 1, 1] # default step across convolutional feature maps

        if not self.max_pool: # don't do any pooling at all
            self.ksize = []
            self.pool_strides = []
        elif self.max_pool: # want to pool, didn't specify stride parameters
            if not self.ksize:
                self.ksize = [1, 2, 2, 1] # default 2x2 max pooling
            if not self.pool_strides:
                self.pool_strides = [1, 2, 2, 1] # default 2x2 max pooling

        if not self.conv_padding:
            self.conv_padding = 'SAME'
        if not self.pool_padding:
            self.pool_padding = 'SAME'

        # Initialize all other parameters
        self.weight_shape = []
        self.bias_shape = []
        self.weights = []
        self.biases = []

        # Initialize output tensor
        self.outputTensor = []

        # Add this layer
        self.__add_layer()

    # Function that adds this layer into our overall neural network tensor flow graph
    def __add_layer(self):
        # Compute shapes for weights and biases
        w0 = self.patch_size[0]
        w1 = self.patch_size[1]
        w2 = int(self.inputTensor.get_shape()[3])
        w3 = self.num_feature_maps

        self.weight_shape = [w0, w1, w2, w3]
        self.bias_shape = [self.num_feature_maps]

        # Initialize weights and biases for this layer with the correct shape
        self.weights = Layer.weight_variable(self)
        self.biases = Layer.bias_variable(self)

        # Initialize output tensor for this layer
        self.outputTensor = tf.nn.relu(self.__conv2d() + self.biases)

        # Max pool if specified, and write to output tensor
        if self.max_pool:
            self.outputTensor = self.__max_pool()

    # Convolution and max pooling functions (treat as private)
    def __conv2d(self):
        return tf.nn.conv2d(self.inputTensor, self.weights, strides=self.conv_strides,
                            padding=self.conv_padding)

    def __max_pool(self):
        return tf.nn.max_pool(self.outputTensor, ksize=self.ksize, strides=self.pool_strides,
                              padding=self.pool_padding)

class FCLayer(Layer):
    # Constructor for FC layer class
    def __init__(self, inputTensor, num_nodes, dropout='False', keep_prob=[]):
        # Call base class constructor
        Layer.__init__(self, 'FC', inputTensor)

        # Initialize FC layer parameters
        self.num_nodes = num_nodes
        self.keep_prob = keep_prob
        self.dropout = dropout

        # Initialize all other parameters
        self.weight_shape = []
        self.bias_shape = []
        self.weights = []
        self.biases = []

        # Initialize output tensor
        self.outputTensor = []

        # Add this layer
        self.__add_layer()

    # Function that adds this layer into our overall neural network tensor flow graph
    def __add_layer(self):
        # Get the shape of the input layer
        inputShape = list(self.inputTensor.get_shape())

        if len(inputShape) == 2:
            inputSize = int(inputShape[1])
            inputNodes = 1
        else:
            inputSize = int(inputShape[1]) * int(inputShape[2])
            inputNodes = int(inputShape[3])

        # Compute shapes for weights and biases
        self.weight_shape = [inputSize * inputNodes, self.num_nodes]
        self.bias_shape = [self.num_nodes]

        # Initialize weights and biases
        self.weights = Layer.weight_variable(self)
        self.biases = Layer.bias_variable(self)

        # Initialize output tensor for this layer
        flatOut = tf.reshape(self.inputTensor, [-1, inputSize * inputNodes])
        self.outputTensor = tf.nn.relu(tf.matmul(flatOut, self.weights) + self.biases)

        # Check if we're doing dropout on this layer
        if self.dropout:
            self.outputTensor = tf.nn.dropout(self.outputTensor, self.keep_prob)

class ReadoutLayer(Layer):
    # Constructor for Readout layer class
    def __init__(self, inputTensor, numOutputClasses):
        # Call base class constructor
        Layer.__init__(self, 'Readout', inputTensor)

        # Set number of output classes
        self.numOutputClasses = numOutputClasses

        # Initialize all other parameters
        self.weight_shape = []
        self.bias_shape = []
        self.weights = []
        self.biases = []

        # Initialize output tensor
        self.outputTensor = []

        # Add this layer
        self.__add_layer()

    # Function that adds this layer into our overall neural network tensor flow graph
    def __add_layer(self):
        # Get number of input nodes from previous layer
        inputNodes = int(self.inputTensor.get_shape()[-1])

        # Set shape for weights and biases of this final layer
        self.weight_shape = [inputNodes, self.numOutputClasses]
        self.bias_shape = [self.numOutputClasses]

        # Initialize weights and biases
        self.weights = Layer.weight_variable(self)
        self.biases = Layer.bias_variable(self)

        # Initialize output tensor for this layer (apply Softmax)
        self.outputTensor = tf.nn.softmax(tf.matmul(self.inputTensor, self.weights) + self.biases)

