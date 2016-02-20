==============================
seq2seq
==============================

What is this module ?
=======================

This module implements seqence to sequence (seq2seq) neural network models
with the aim to create dialogue systems.

This module uses `chainer <http://chainer.org/>`_ to implement neural network models.


Install
=================

Install

- chainer (1.6.1)
- gensim (0.12.4)
- Install cuda when using GPU

::

    $ pip3 install chainer==1.6.1
    $ pip3 install gensim==0.12.4
    $ pacman -S cuda  # for ArchLinux


Usage
======

First, train a model.

There are a configuration file `test.ini` and corpus files in `corpus_test` directory

- config file: `test.ini`
- dialogue corpus: `corpus_test/sent.txt` `corpus_test/conv.txt` .

Use `train.py` to train an encoder model.

::

    $ python train.py test.ini -g0 -tlstm

After finishing training, use `test.py` for talking with the model.

::

    $ python test.py test.ini -g0 -tlstm <./corpus_test/sent.char.txt

Use `-g` option to use GPU.

::

    # enable GPU to use -g option
    $ python train_rnn_encoder.py test.ini -g 0
    $ python train_rnn_decoder.py test.ini -g 0
    $ python test_rnn_decoder.py test.ini -g 0

There is Makefile for convenient.

::

    # train the model
    $ make train
    # test the model
    # mkae test

Model description
==================

- Word Embedding initializer: word2vec or random between [0, 1]
- Layers : embedding layer -> 2 hidden layers -> output layer -> Softmax
- Units/activation functions: LSTM or ReLU with dropout option
- Optimizer: ADAM with clipping
