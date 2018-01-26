# pyffe
A small Python library to setup, run, manage and monitor image classification experiments based on Caffe.

## Requirements

 - Caffe with Python interface (PyCaffe)
 
## Example of usage

```python
import pyffe
from pyffe.models import AlexNet

# Create datasets:
# 'data/dataset1' is a directory containing list files with image urls.
# Each list file define a different subset of data (e.g: train.txt, val.txt, test.txt)
dataset1 = pyffe.Dataset('data/dataset1')
dataset2 = pyffe.Dataset('data/dataset2')

# Define an input format for image pre-processing.
# Available options follow the 'image_data_param' definition in 'caffe.proto'
input_format = pyffe.InputFormat(
    new_width=256,
    new_height=256,
    crop_size=224,
    scale=1. / 256,
    mirror=True
)

# Create a model
model = AlexNet(input_format,
    num_output=2, # no. of classes
    batch_sizes=[64, 10] # batch sizes for train and val
)

# Define training hyper-parameters
# Similar to the definition of a Caffe Solver,
# but with epochs instead of iterations.
solver = pyffe.Solver(
    base_lr=0.01,
    train_epochs=10,
    lr_policy="step",
    gamma=0.1,
    stepsize_epochs=5,
    val_interval_epochs=1,
    val_epochs=1,
    display_per_epoch=30,
    snapshot_interval_epochs=1,
)

# Define a series of experiments
experiments = [
    # Standard training on one dataset having 3 splits
    pyffe.Experiment(model, solver, train=dataset1.train, val=dataset1.val, test=dataset1.test),
    # Train on whole dataset2, evaluate performance on dataset1 val split during validation, and finally test on dataset1 test split
    pyffe.Experiment(model, solver, train=dataset2.all, val=dataset1.val, test=dataset1.test),
    # Train on dataset1 train split, evaluate performance on dataset1 and dataset2 validation splits during training, finally report performance on dataset1 and dataset2 test splits.
    pyffe.Experiment(model, solver, train=dataset1.train, val=[dataset1.val, dataset2.val], test=[dataset1.test, dataset2.test])
]

# Setup and run experiments
for exp in experiments:
    exp.setup('runs/') # this will create an experiment folder and generate the needed prototxt files
    exp.run() # this will run the 'caffe' executable and monitor the training with live plots.

# Summarize the results (returns Pandas DataFrame)
dataframe = pyffe.summarize(experiments)
dataframe.to_csv('results.csv')


```
