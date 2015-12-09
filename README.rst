==============================
seq2seq
==============================

概要
=====

LSTM を用いた sequence to sequence の encoder/decoder の実装です。
たとえば対話システムの作成に使うことができます。

インストール
=================

- Python 3.x.x をインストールしてください。
- Install python modules via pip
    - cython  http://cython.org/
    - numpy  https://github.com/numpy/numpy
    - scipy  https://github.com/scipy/scipy
    - chainer  https://github.com/pfnet/chainer
    - gensim  https://radimrehurek.com/gensim/

    ::

        $ pip install cython numpy scipy gensim chainer gensim

Python 3.5 よりバージョンが小さい場合には

- mypy-lang  http://mypy-lang.org/

もインストールしてください。

オプションですが、chainer で GPU を使うために
cuda をインストールしておいた方がよいです。

::

    $ pacman -S cuda  # for ArchLinux

GPU を使わないと学習に時間がかかりすぎるため、使うのは困難でしょう。

使い方
======

はじめに encoder, decoder をこの順に学習します。

以下はテスト用の設定ファイル `test.ini` を用います。
テスト用のコーパスは `corpus_test` 以下にはいっています。

encoder を訓練するには以下のコマンドを実行します。

::

    $ python train_rnn_encoder.py test.ini
    creating dictionary
    add start and end symbols in dictionary
    adding words from corpus_test/sent.txt to dictionary
    RNN structure: word -> 31 -> 700 -> 700 nodes, 6 LSTM layers -> 31, GPU flag: False
    loaded sentences from corpus_test/sent.txt
    data size: 8
    running epoch 0
    epoch 0 batch 0 loss 2.7052767276763916 forward 0:00:00.466268 backward 0:00:02.921779
    epoch 0 batch 1 loss 2.160247802734375 forward 0:00:00.289071 backward 0:00:01.900879
    epoch 0 batch 2 loss 2.4732136726379395 forward 0:00:00.479989 backward 0:00:03.109619
    epoch 0 batch 3 loss 2.813724994659424 forward 0:00:00.482876 backward 0:00:03.133984
    saved model as encoder_model_test/model_0.npz
    epoch: 0, loss 2.5381157994270325

1 エポックだけでは訓練が足りない場合は、 `for` ループで回しましょう。

::

    $ for _ in $(seq 10); do python train_rnn_encoder.py test.ini; done

decoder を訓練するには以下のコマンドを使います。

::

    $ python train_rnn_decoder.py test.ini
    load dictionary: 31 items
    RNN structure: word -> 31 -> 700 -> 700 nodes, 6 LSTM layers -> 31, GPU flag: False
    RNN structure: word -> 31 -> 700 -> 700 nodes, 6 LSTM layers -> 31, GPU flag: False
    loaded RNN model: encoder_model_test/model_0.npz
    loaded RNN model: decoder_model_test/model_0.npz
    loaded sentences from corpus_test/conv.txt
    data size: 4
    running epoch 0
    decoder epoch 0 batch 0 loss 1.1433789730072021 forward 0:00:01.137217 backward 0:00:07.309638
    decoder epoch 0 batch 1 loss 1.5498902797698975 forward 0:00:00.843419 backward 0:00:06.869120
    saved model as decoder_model_test/model_0.npz
    epoch: 0, loss 1.3466346263885498

1 エポックだけでは訓練が足りない場合は、 encoder の場合と同様に `for` ループで回しましょう。

::

    $ for _ in $(seq 10); do python train_rnn_decoder.py test.ini; done

テストするには以下のコマンドを実行します。

::

    $ python test_rnn_decoder.py test.ini
    load dictionary: 31 items
    RNN structure: word -> 31 -> 700 -> 700 nodes, 6 LSTM layers -> 31, GPU flag: False
    loaded RNN model: encoder_model_test/model_0.npz
    RNN structure: word -> 31 -> 700 -> 700 nodes, 6 LSTM layers -> 31, GPU flag: False
    loaded RNN model: decoder_model_test/model_0.npz
    おはよう！
    ['！', 'う', 'よ', 'は', 'お', '</S>']
    おんんばは

生成される文章がおかしい場合は、モデルの学習をくりかえしましょう。

モデルの学習が遅い場合は GPU を用いましょう（通常は使わないと遅すぎるでしょう）

::

    # -g オプションで GPU を ON にする
    $ python train_rnn_encoder.py test.ini -g 1
    $ python train_rnn_decoder.py test.ini -g 1
    $ python test_rnn_decoder.py test.ini -g 1

