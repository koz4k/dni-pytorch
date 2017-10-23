Word-level language modeling
----------------------------

This example illustrates how to use ``BackwardInterface`` with an RNN to
approximate gradient from an infinitely-unrolled sequence. Code is mostly
copied from the official PyTorch word-level language modeling example:
https://github.com/pytorch/examples/blob/master/word_language_model

Synthesizer used is an MLP with two hidden layers and ReLU activation function.

In the example training commands below, BPTT length was reduced to 5 to
highlight the ability to train on shorter sequences using DNI.

To install requirements::

    $ pip install -r requirements.txt

To train with regular backpropagation through time::

    $ python main.py --cuda --bptt 5 --epochs 6

To train with DNI::

    $ python main.py --cuda --bptt 5 --epochs 6 --dni
