import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

###################
# the required callback for a Tensorflow transform
def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.compute_and_apply_vocabulary(s)
  x_centered_times_y_normalized = x_centered * y_normalized
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }
###################

# must define the schema for our data
raw_data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        's': tf.FixedLenFeature([], tf.string),
        'y': tf.FixedLenFeature([], tf.float32),
        'x': tf.FixedLenFeature([], tf.float32),
    }))

raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

transformed_dataset, transform_fn = ( (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn) )

# NOTE: AnalyzeAndTransformDataset is the amalgamation of two tft_beam functions: 
#  transformed_data, transform_fn = (my_data | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
# same as: 
# a = tft_beam.AnalyzeDataset(preprocessing_fn)
# transform_fn = a.expand(my_data)   # my_data is a dataset, applies preprocessing_fn, returns a transform_fn objA
#       transform_fn is a pure function that is applied to every row of incoming dataset
#       at this point, tf.Transform analyzers (like tft.mean() have already been computed and are constants, 
#       so transform_fn has constants for the mean of column x, the min and max of column y, i
#       and the vocabulary used to map the strings to integers
# all aggregation of data happens in AnalyzeDataset
# tranform_fun represented as a Tensorflow graph, so can be embedded into serving graph
transform_fn = my_data | tft_beam.AnalyzeDataset(preprocessing_fn)
# t = tft_beam.TransformDataset()   # instantiate this class
# transformed_data = t.expand( (my_data, transform_fn) )    # takes in a 2-tuple, outputs "dataset"
transformed_data = (my_data, transform_fn) | tft_beam.TransformDataset()
# where:
#  my_data is a "dataset": a typ
transformed_data, transformed_metadata = transformed_dataset
