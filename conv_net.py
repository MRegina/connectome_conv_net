    # -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:41:17 2017

Code for connectome based classification with convolutional networks (CCNN)

partially based on code from Deep learning course by Udacity 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb

@author: mregina
"""

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

#load sample data
pickle_file = 'tensors_medium_noise.pickle' #or tensors_high_noise.pickle
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  data_tensor1 = save['data_tensor1']
  data_tensor2 = save['data_tensor2']
  labels = save['label']
  del save  # hint to help gc free up memory

# we have 150 "independent" subjects, so subjectIDs are unique, but in case of real data, where one subject has multiple measurements, the same subjectID should be repeated for those meaurements  
subjectIDs=range(150) 

#%%
#define functions for cross-validation, tensor randomization and normalization and performance calculation

#create_train_and_test_folds randomly divides subjectIDs stored in subjects to num_folds sets
#INPUT: num_folds: number of folds in cross-validation (integer)
#       subjects: list of unique subject IDs
#OUTPUT: IDs: array storing unique subject IDs with num_folds columns: each column contains IDs of test subjects of the given fold
def create_train_and_test_folds(num_folds,subjects):
    n=np.ceil(len(subjects)/num_folds).astype(np.int)
    np.random.shuffle(subjects)
    if len(subjects)!=n*num_folds:
        s=np.zeros(n*num_folds)
        s[:len(subjects)]=subjects
        subjects=s
    IDs=subjects.reshape((n,num_folds))
    return IDs

#normalize_tensor standardizes an n dimesional np.array to have zero mean and standard deviation of 1
def normalize_tensor(data_tensor):
    data_tensor-=np.mean(data_tensor)
    data_tensor/=np.max(np.abs(data_tensor))
    return data_tensor

#randomize_tensor generates a random permutation of instances and the corresponding labels before training
#INPUT: dataset: 4D tensor (np.array), instances are concatenated along the first (0.) dimension
#       labels: 2D tensor (np.array), storing labels of instances in dataset,instances are concatenated along the first (0.)
#               dimension, number of columns corresponds to the number of classes, i.e. labels are stored in one-hot encoding 
#OUTPUT: shuffled_dataset: 4D tensor (np.array), instances are permuted along the first (0.) dimension
#        shuffled_labels: 2D tensor (np.array), storing labels of instances in shuffled_dataset
def randomize_tensor(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

#create_train_and_test_data creates and prepares training and test datasets and labels for a given fold of cross-validation
#INPUT: fold: number of the given fold (starting from 0)
#       IDs: array storing unique subject IDs with num_folds columns: each column contains IDs of test subjects of the given fold 
#           (output of reate_train_and_test_folds)
#       subjectIDs: list of subject IDs corresponding to the order of instances stored in the dataset (ID of the same subject might appear more than once)
#       labels: 1D vector (np.array) storing instance labels as integers (label encoding)
#       data_tensor: 4D tensor (np.array), instances are concatenated along the first (0.) dimension
#OUTPUT: train_data: 4D tensor (np.array) of normalized and randomized train instances of the given fold
#        train_labels: 2D tensor (np.array), storing labels of instances in train_data in one-hot encoding
#        test_data: 4D tensor (np.array) of normalized (but not randomized) test instances of the given fold
#        test_labels: 2D tensor (np.array), storing labels of instances in test_data in one-hot encoding
def create_train_and_test_data(fold,IDs,subjectIDs,labels,data_tensor):
    #create one-hot encoding of labels
    num_labels=len(np.unique(labels))
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    
    #identify the IDs of test subjects
    testIDs=np.in1d(subjectIDs,IDs[:,fold])
        
    test_data=normalize_tensor(data_tensor[testIDs,:,:,:]).astype(np.float32)
    test_labels=labels[testIDs]
    
    train_data=normalize_tensor(data_tensor[~testIDs,:,:,:]).astype(np.float32)
    train_labels=labels[~testIDs]
    train_data,train_labels=randomize_tensor(train_data,train_labels)
    
    return train_data,train_labels,test_data,test_labels

#accuracy calculates classification accuracy from one-hot encoded labels and predictions
#INPUT: predictions: 2D tensor (np.array), storing predicted labels (calculated with soft-max in our case) of instances with one-hot encoding  
#       labels: 2D tensor (np.array), storing actual labels with one-hot encoding
#OUTPUT: accuracy in %
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


  
    

#%%
#initialize network parameters

numROI=499
num_channels=2
num_labels=2
image_size=numROI
batch_size = 4
patch_size = image_size
depth = 64
num_hidden = 96
keep_pr=0.6


combined_tensor=np.zeros((data_tensor1.shape[0],data_tensor1.shape[1],data_tensor1.shape[2],data_tensor1.shape[3]*num_channels))

combined_tensor[:,:,:,0]=normalize_tensor(data_tensor1[:,:,:,0])
combined_tensor[:,:,:,1]=normalize_tensor(data_tensor2[:,:,:,0])


subjects=np.unique(subjectIDs)


num_folds=10
IDs=create_train_and_test_folds(num_folds,subjects)
#IDs=np.load("F:/DTW/conv_net/IDs7_old.npy") #IDs can be loaded for repeated tests

            
test_labs=[]
test_preds=[]



#%%
#launch TensorFlow in each fold of cross-validation
for i in range(num_folds):
    
    train_data,train_labels,test_data,test_labels=create_train_and_test_data(i,IDs,subjectIDs,labels,combined_tensor)
    
    train_data=train_data[:,:image_size,:image_size,:]
    test_data=test_data[:,:image_size,:image_size,:]
    
    graph = tf.Graph()
    
    with graph.as_default():
    
      #input data placeholders
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      
      #test data is a constant
      tf_test_dataset = tf.constant(test_data)
      
      #network weight variables: Xavier initialization for better convergence in deep layers
      layer1_weights = tf.get_variable("layer1_weights", shape=[1, patch_size, num_channels, depth],
           initializer=tf.contrib.layers.xavier_initializer())
      layer1_biases = tf.Variable(tf.constant(0.001, shape=[depth]))
      layer2_weights = tf.get_variable("layer2_weights", shape=[patch_size, 1, depth, 2*depth],
           initializer=tf.contrib.layers.xavier_initializer())
      layer2_biases = tf.Variable(tf.constant(0.001, shape=[2*depth]))
      layer3_weights = tf.get_variable("layer3_weights", shape=[2*depth, num_hidden],
           initializer=tf.contrib.layers.xavier_initializer())
      layer3_biases = tf.Variable(tf.constant(0.01, shape=[num_hidden]))
      layer4_weights = tf.get_variable("layer4_weights", shape=[num_hidden, num_labels],
           initializer=tf.contrib.layers.xavier_initializer())
      layer4_biases = tf.Variable(tf.constant(0.01, shape=[num_labels]))
      
      #convolutional network architecture
      def model(data, keep_pr):
        #first layer: line-by-line convolution with ReLU and dropout
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv+layer1_biases),keep_pr)
        #second layer: convolution by column with ReLU and dropout
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.dropout(tf.nn.relu(conv+layer2_biases),keep_pr)
        #third layer: fully connected hidden layer with dropout and ReLU
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases),keep_pr)
        #fourth (output) layer: fully connected layer with logits as output
        return tf.matmul(hidden, layer4_weights) + layer4_biases
      
      #calculate loss-function (cross-entropy) in training
      logits = model(tf_train_dataset,keep_pr)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
      #optimizer definition
      learning_rate = 0.001
      optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
      
      
      #calculate predictions from training data
      train_prediction = tf.nn.softmax(logits)
      #calculate predictions from test data (keep_pr of dropout is 1!)
      test_prediction = tf.nn.softmax(model(tf_test_dataset,1))
      
      # nuber of iterations
      num_steps = 20001
    
    #start TensorFlow session
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      
      
      for step in range(num_steps):
          
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        
        if (offset == 0 ): #if we seen all train data at least once, re-randomize the order of instances
            train_data, train_labels = randomize_tensor(train_data, train_labels)
        
        #create batch    
        batch_data = train_data[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        #feed batch data to the placeholders
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        # at every 2000. step give some feedback on the progress
        if (step % 2000 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    
    
      #evaluate the trained model on the test data in the given fold
      test_pred=test_prediction.eval()
      print('Test accuracy: %.1f%%' % accuracy(test_pred, test_labels))
      
      #save test predictions and labels of this fold to a list
      test_labs.append(test_labels)
      test_preds.append(test_pred)

#create np.array to store all predictions and labels
l=test_labs[0]
p=test_preds[0]   
#iterate through the cross-validation folds    
for i in range(1,num_folds):
    l=np.vstack((l,test_labs[i]))
    p=np.vstack((p,test_preds[i]))

#calculate final accuracy    
print('Test accuracy: %.1f%%' % accuracy(p, l))

#save data
np.savez("predictions.npz",labels=l,predictions=p,splits=IDs)
