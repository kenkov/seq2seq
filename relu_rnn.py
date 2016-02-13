#! /usr/bin/env python
# coding:utf-8

from chainer import Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import rnn
from rtype import State
from typing import List, Tuple, Dict
import numpy as np


# define classifier
Classifier = rnn.Classifier


class ReLURNN(Chain):
    def __init__(
            self,
            embed_dim: int,
            n_units: int=1000,
            gpu: int=-1,
    ):
        """
        3 層モデル
        layer 毎の初期化
          embed.W
            init: word2vec により初期化
          l1.W, l1.b, l2.W, l2.W, l3.W, l3.b
            init: [0, 1] uniform distribution により初期化
          h1.W, h1.b, h2.W, h2.W
            init: W は単位行列、b は 0 で初期化

        出力層以外は dropout する

        出力層は Softmax
        目的関数は cross-entropy
        optimizer は clipping-SGD ( clipping l2 norm = 5)
            (+ L2 したいが実装できていない)
        """

        super(ReLURNN, self).__init__(
            embed=L.EmbedID(embed_dim, n_units),  # word embedding
            l1=L.Linear(n_units, n_units),
            h1=L.Linear(n_units, n_units),
            l2=L.Linear(n_units, n_units),
            h2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, embed_dim),
        )
        self.embed_dim = embed_dim
        self.n_units = n_units
        self.gpu = gpu

        # S-RNN + ReLU の初期化
        #     W = E
        #     b = 0
        # のように初期化
        self.h1.W.data = self.xp.identity(n_units, dtype=self.xp.float32)
        self.h1.b.data = self.xp.zeros(n_units, dtype=self.xp.float32)
        self.h2.W.data = self.xp.identity(n_units, dtype=self.xp.float32)
        self.h2.b.data = self.xp.zeros(n_units, dtype=self.xp.float32)

        # その他は uniform distribution で初期化
        # for param in model.parameters:
        #     param[:] = np.random.uniform(-0.1, 0.1, param.shape)

    def set_word_embedding(self, array):
        self.embed.W.data = array

    def forward_one(
            self,
            word: Variable,
            state: State,
            dropout: bool=True,
            train: bool=False,
            dropout_ratio: float=0.4,
    ) -> Tuple[Variable, State]:
        if dropout:
            # 現状 batch 毎の dropout になっていない
            z0 = F.dropout(self.embed(word), ratio=dropout_ratio, train=train)
            u1 = self.l1(z0) + self.h1(state["h1"])
            z1 = F.dropout(F.relu(u1), ratio=dropout_ratio, train=train)
            u2 = self.l2(z1) + self.h2(state["h2"])
            z2 = F.dropout(F.relu(u2), ratio=dropout_ratio, train=train)
            u3 = self.l3(z2)
        else:
            z0 = self.embed(word)
            u1 = self.l1(z0) + self.h1(state["h1"])
            z1 = F.relu(u1)
            u2 = self.l2(z1) + self.h2(state["h2"])
            z2 = F.relu(u2)
            u3 = self.l3(z2)

        new_state = {
            "h1": z1,
            "h2": z2,
        }
        return u3, new_state

    def forward(
            self,
            words: List[Variable],
            state: State,
            dropout: bool=True,
            train: bool=False
    ) -> Tuple[List[Variable], State]:

        # state は 0 で初期化する
        state = state if state else {
            key: Variable(
                self.xp.zeros(
                    (1, self.n_units),
                    dtype=self.xp.float32)
            ) for key in ["h1", "h2"]
        }
        if state is None:
            for key in state:
                assert np.count_nonzero(state[key]) == 0

        ys = []
        for word in words:
            y, state = self.forward_one(
                word, state, dropout=dropout, train=train
            )
            ys.append(y)
        return ys, state

    def __call__(
            self,
            words: List[Variable],
            state: State,
            dropout: bool=True,
            train: bool=False
    ):
        return self.forward(
            words, state, dropout=dropout, train=train
        )


class Classifier(Chain):
    def __init__(
            self,
            predictor,
    ):
        super(Classifier, self).__init__(
            predictor=predictor,
        )

    def loss(
        self,
        words: [int],
        state: State,
        dropout: bool=True,
        train: bool=False
    ) -> Variable:
        if len(words) <= 1:
            raise Exception("word length error: >= 2")

        # convert words to variable
        _words = [
            Variable(self.xp.array([word], dtype=self.xp.int32))
            for word in words
        ]

        # predict next words
        ys, new_state = self.predictor(
            _words[:-1], state,
            dropout=dropout,
            train=train
        )

        # calculate loss
        loss = Variable(self.xp.zeros((), dtype=self.xp.float32))
        for y, t in zip(ys, _words[1:]):
            new_loss = F.softmax_cross_entropy(y, t)
            loss += new_loss

        len_words = Variable(self.xp.array(
            len(words) - 1,
            dtype=self.xp.float32
        ))
        return loss / len_words

    def __call__(
        self,
        words: int,
        state: State=None,
        dropout: bool=True,
        train: bool=False
    ) -> Tuple[Variable, Dict[str, Variable], Variable]:
        return self.loss(
            words, state,
            dropout=dropout,
            train=train
        )
