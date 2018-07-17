from __future__ import print_function

#import math

from IPython import display
#from matplotlib import cm
#from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
#from numpy import argmax
#from keras.utils import to_categorical
#from numpy import array

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

dataset = pd.read_csv("bearing_dataset_final.csv")

dataset= dataset.reindex(
    np.random.permutation(dataset.index))


def preprocess_features(dataset):
  """Prepares input features.

  Args:
    dataset: A Pandas DataFrame e
  Returns:
    A DataFrame that contains the features to be used for the model
  """
  selected_features = dataset[
    ["FE_0",
     "DE_0",
     "FE_1",
     "DE_1",
     "FE_2",
     "DE_2",
     "FE_3",
     "DE_3"]]

  processed_features = selected_features.copy()
  
  return processed_features


def preprocess_targets(dataset):
  """Prepares target features (i.e., labels) 
  Args:
    A Pandas DataFrame 
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets =dataset['label']
  #output_targets = pd.get_dummies(output_targets)
  
  return output_targets



training_examples = preprocess_features(dataset.head(100000))
training_targets = preprocess_targets(dataset.head(100000))

validation_examples = preprocess_features(dataset.tail(50000))
validation_targets = preprocess_targets(dataset.tail(50000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.
  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.     
  Returns:
    A `DNNRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  
  
  #my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNClassifier(
          feature_columns=construct_feature_columns(training_examples),
          hidden_units=hidden_units,
          optimizer=my_optimizer,
          n_classes = 2
      )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets, 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []

  for period in range (0, periods):
      
  # Train the model, starting from the prior state.

      dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period)
  
   
      # Take a break and compute predictions.
      training_probabilities = dnn_regressor.predict(input_fn=predict_training_input_fn)
      training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
                    
      validation_probabilities = dnn_regressor.predict(input_fn=predict_validation_input_fn)
      validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
                    
      training_log_loss = metrics.log_loss(training_targets, training_probabilities)
      validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
      # Occasionally print the current loss.
      print("  period %02d : %0.2f" % (period, training_log_loss))
                    # Add the loss metrics from this period to our list.
      training_log_losses.append(training_log_loss)
      validation_log_losses.append(validation_log_loss)
                    
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()


  evaluation_metrics = dnn_regressor.evaluate(input_fn=predict_validation_input_fn)
  print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
  print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])


           

  return dnn_regressor



_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.1),
    steps=5000,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=training_examples, 
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# Test set Evaluation

predict_x = pd.read_csv('bearing_test_final.csv')
predict_x = predict_x.reindex(
    np.random.permutation(predict_x.index))

test_x = preprocess_features(predict_x)
test_y = preprocess_targets(predict_x)

evaluation_metrics = _.evaluate(
        input_fn=lambda:eval_input_fn(test_x,test_y,
                                                batch_size=1))

print("Accuracy on the Test set: %0.2f" % evaluation_metrics['accuracy'])
print("AUC on the Test set: %0.2f" % evaluation_metrics['auc'])

# Test set predictions :

predict_x = pd.read_csv('bearing_test_final.csv')
test_x = preprocess_features(predict_x)
predictions = _.predict(
        input_fn=lambda:eval_input_fn(test_x, labels=None, batch_size=1))
class_ids = np.array([item['class_ids'][0] for item in predictions])