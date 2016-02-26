==============================
seq2seq
==============================

Seq2seq is a Python library for modeling dialogue/conversational model with neural network.

This software implements seqence to sequence (seq2seq) neural network models
with the aim to create dialogue systems.

Install
=================

This software depends on chainer and gensim. You must install them first.

::

    $ pip3 install chainer==1.6.1
    $ pip3 install gensim==0.12.4

It is also recommended to install cuda to use GPU.
It is optional, but using GPU improves performance.

::

    $ pacman -S cuda  # for ArchLinux


Usage
======

You must learn a model to use seq2seq first with you own corpus.
It is easy to use your own corpus in this module.
The following description uses **test corpus** included in this software
to show you how to apply learning scripts to the corpus.

`corpus_test` directory has a test corpus `conv.txt`.
corpus should be the folllowing format.

::

    original_ sentence<TAB>reply_sentence

A sentence in the second column is reply to a sentence in the first column.
These sentences should be separated by `TAB`.

Each sentence should be divided into tokens such as words or characters.
You can utilize `corpus_test/Makefile` to split sentences into words and characters.

::

    $ cd corpus_test
    $ make char

Makefile splits `conv.txt` to `sent.char.txt` and `conv.char.txt`.
`sent.char.txt` has all texts in `conv.txt` splitted by characters.
`conv.char.txt` has all conversations in `conv.txt` splitted by characters.

Makefile can also split `conv.txt` to `sent.word.txt` and `conv.word.txt`
by using `make word`. When you use this, you should specify `TOKENIZER`
in `Makefile` first. The default is `mecab -Owakati`.
`sent.word.txt` has all texts in `conv.txt` splitted by words.
`conv.word.txt` has all conversations in `conv.txt` splitted by words.

This section uses `conv.sent.txt` and `sent.sent.txt` to descrive usage, but you can also use `conv.word.txt` and `sent.word.txt` instead.

After preparing a corpus, you can learn your model which predicts a reply sentence from a input sentence based on the corpus.
There are a configuration file `test.ini` which has parameters to learn a model from the test corpus.

Use `train.py` to learn your model.

::

    $ python train.py test.ini -tlstm

Specify `-g0` if you use GPU.

::

    $ python train.py test.ini -tlstm


After finishing training, use `test.py` for talking with the model.

::

    $ python test.py test.ini -tlstm <./corpus_test/sent.char.txt

Use `-g` option to use GPU.

::

    # enable GPU to use -g option
    $ python train.py test.ini -g0 -tlstm
    $ python test.py test.ini -g0 -tlstm

There is Makefile for convenient.

::

    # train the model
    $ make train
    # test the model
    # make test

Model description
==================

- Word Embedding initializer: word2vec or random between [0, 1]
- Layers : embedding layer -> 2 hidden layers -> output layer -> Softmax
- Units/activation functions: LSTM or ReLU with dropout option
- Optimizer: ADAM with clipping
