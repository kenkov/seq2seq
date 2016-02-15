==============================
seq2seq
==============================

What is this module ?
=======================

This module implements seqence to sequence (seq2seq) neural network models
with the aim to create dialogue systems.

It uses `chainer <http://chainer.org/>`_ to implement neural network models.


Install
=================

Install

- chainer (1.6.1)
- gensim (0.12.2)
- Install cuda when using GPU

    ::

        $ pacman -S cuda  # for ArchLinux


Usage
======

First, train encoder and decoder models.

There are a configuration and corpus files in `corpus_test` directory

- config file: `test.ini`
- dialogue corpus: `corpus_test/sent.txt` `corpus_test/conv.txt` .

Use `train_encoder.py` to train an encoder model.

::

    python train_encoder.py -t relu test.ini 2>/dev/null
    load model encoder_model_test/model.npz
    set optimizer clip threshold: 5.0
    loaded sentences from corpus_test/sent.word.txt
    data size: 12
    epoch 0
    epoch 0 batch 0: loss 3.6170058250427246, grad L2 norm: 3.9864994687553197, forward 0:00:00.014572, optimizer 0:00:00.009774
    epoch 0 batch 1: loss 3.6550519466400146, grad L2 norm: 2.3834681732152294, forward 0:00:00.010261, optimizer 0:00:00.003314
    epoch 0 batch 2: loss 3.7410669326782227, grad L2 norm: 5.000003863101264, forward 0:00:00.019999, optimizer 0:00:00.009585
    epoch 0 batch 3: loss 5.606266975402832, grad L2 norm: 4.9999995628779095, forward 0:00:00.031611, optimizer 0:00:00.015180
    epoch 0 batch 4: loss 3.6682329177856445, grad L2 norm: 1.7967707531089245, forward 0:00:00.003995, optimizer 0:00:00.002180
    epoch 0 batch 5: loss 18.82986831665039, grad L2 norm: 4.99999907185384, forward 0:00:00.025314, optimizer 0:00:00.012267
    finish epoch 0, loss 0.03911749291419983
    epoch 1
    epoch 1 batch 0: loss 3.553004503250122, grad L2 norm: 3.7215319124368187, forward 0:00:00.017583, optimizer 0:00:00.009865
    epoch 1 batch 1: loss 3.5388011932373047, grad L2 norm: 3.088856597811233, forward 0:00:00.013634, optimizer 0:00:00.004701
    epoch 1 batch 2: loss 6.547500133514404, grad L2 norm: 4.999992318087526, forward 0:00:00.026898, optimizer 0:00:00.012684
    epoch 1 batch 3: loss 3.7236814498901367, grad L2 norm: 5.00000295885751, forward 0:00:00.022071, optimizer 0:00:00.010529
    epoch 1 batch 4: loss 4.04321813583374, grad L2 norm: 4.999997366731293, forward 0:00:00.014172, optimizer 0:00:00.006885
    epoch 1 batch 5: loss 4.456284046173096, grad L2 norm: 5.000000392657223, forward 0:00:00.017445, optimizer 0:00:00.008506
    finish epoch 1, loss 0.025862489461898803
    epoch 2
    ...


Then, Use `train_decoder.py` to train a decoder model.

::

    $ python train_decoder.py -t relu test.ini
    load dictionary: 47 items
    load encoder model encoder_model_test/model.npz
    load decoder model decoder_model_test/model.npz
    set optimizer clip threshold: 5.0
    loaded sentences from corpus_test/conv.word.txt
    data size: 6
    running epoch 0
    epoch 0 batch 0: loss 14.778450012207031, grad L2 norm: 5.000001444721211, forward 0:00:00.032181, optimizer 0:00:00.015524
    epoch 0 batch 1: loss 13.42282772064209, grad L2 norm: 5.000006844430436, forward 0:00:00.057257, optimizer 0:00:00.025782
    epoch 0 batch 2: loss 4.566666126251221, grad L2 norm: 4.999997299629466, forward 0:00:00.021281, optimizer 0:00:00.009233
    finish epoch 0, loss 0.03276794385910034
    running epoch 1
    epoch 1 batch 0: loss 3.9690470695495605, grad L2 norm: 4.728466637951272, forward 0:00:00.015587, optimizer 0:00:00.007583
    epoch 1 batch 1: loss 5.986050605773926, grad L2 norm: 5.000004347569049, forward 0:00:00.036949, optimizer 0:00:00.017259
    epoch 1 batch 2: loss 16.204647064208984, grad L2 norm: 5.0000030326973395, forward 0:00:00.052537, optimizer 0:00:00.024978
    finish epoch 1, loss 0.026159744739532472
    running epoch 2
    epoch 2 batch 0: loss 9.967018127441406, grad L2 norm: 5.000000181427455, forward 0:00:00.040972, optimizer 0:00:00.019673
    epoch 2 batch 1: loss 6.41628885269165, grad L2 norm: 5.0000019923954495, forward 0:00:00.026851, optimizer 0:00:00.012867
    epoch 2 batch 2: loss 4.393733978271484, grad L2 norm: 5.000003581958471, forward 0:00:00.035179, optimizer 0:00:00.016866
    finish epoch 2, loss 0.02077704095840454


You can talk with the model after training encoder and decoder models.

::

    python test_decoder.py -t relu test.ini
    load dictionary: 47 items
    load encoder model encoder_model_test/model.npz
    load decoder model decoder_model_test/model.npz
    おはよう
    ['おはよう', '</S>']
    おはようござますす！！！


Use `-g` option to use GPU.

::

    # enable GPU to use -g オプション
    $ python train_rnn_encoder.py test.ini -g 1
    $ python train_rnn_decoder.py test.ini -g 1
    $ python test_rnn_decoder.py test.ini -g 1
