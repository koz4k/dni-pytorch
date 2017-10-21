MNIST with CNN
--------------

This example illustrates how to implement a custom ``Synthesizer``.
Code is mostly copied from the official PyTorch MNIST example:
https://github.com/pytorch/examples/blob/master/mnist/main.py

Classification model is the same as in the original example (a CNN) with
batch normalization added on every layer and DNI inserted between the last
convolutional layer and the first fully-connected layer (before activation).

Synthesizer used is a CNN with three convolutional layers with padding, so
that sizes of the feature maps are kept constant, and ReLU activation function.

To install requirements::

    $ pip install -r requirements.txt

To train with regular backpropagation::

    $ python main.py

To train with DNI (no label conditioning)::

    $ python main.py --dni

To train with cDNI (label conditioning)::

    $ python main.py --dni --context
