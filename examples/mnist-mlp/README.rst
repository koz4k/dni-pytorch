MNIST with MLP
--------------

This example illustrates how to use ``BackwardInterface`` with a simple
multi-layer perceptron. Code is mostly copied from the official PyTorch
MNIST example:
https://github.com/pytorch/examples/blob/master/mnist/main.py

Classification model is replaced by a multi-layer perceptron with two hidden
layers, 256 neurons in each, batch normalization after every layer and ReLU
activation function. DNI is inserted between the last hidden layer and the
output layer (before activation).

Synthesizer used is an MLP with one hidden layer with 256 neurons and ReLU
activation function.

To install requirements::

    $ pip install -r requirements.txt

To train with regular backpropagation::

    $ python main.py

To train with DNI (no label conditioning)::

    $ python main.py --dni

To train with cDNI (label conditioning)::

    $ python main.py --dni --context
