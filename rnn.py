#! /usr/bin/env python
# coding:utf-8

from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L
from typing import List, Tuple, Dict
from rtype import State


class RNN(Chain):
    def __init__(
            self,
            embed_dim: int,
            n_units: int=200,
            h_units: int=200,
            gpu: int=-1
    ):
        super(RNN, self).__init__(
            embed=L.EmbedID(embed_dim, n_units),  # word embedding
            l1=L.Linear(n_units, n_units),
            h1=L.Linear(n_units, n_units),
            l2=L.Linear(n_units, n_units),
            h2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, embed_dim),
        )
        self.embed_dim = embed_dim
        self.n_units = n_units
        self.h_units = h_units
        self.gpu = gpu

        # LSTM のかわりに S-RNN + ReLU を使う。
        # この場合、
        #     W = E
        #     b = 0
        # のように初期化する必要あり
        self.h1.W.data = self.xp.identity(n_units, dtype=self.xp.float32)
        self.h1.b.data = self.xp.zeros(n_units, dtype=self.xp.float32)
        self.h2.W.data = self.xp.identity(n_units, dtype=self.xp.float32)
        self.h2.b.data = self.xp.zeros(n_units, dtype=self.xp.float32)

    def forward_one(
            self,
            word: Variable,
            state: State,
            dropout: bool=False,
            train: bool=False
    ) -> Tuple[Variable, State]:
        y0 = self.l0(word)
        if dropout:
            y1 = F.relu(self.l1(F.dropout(y0, train=train)) + state["h1"])
            y2 = F.relu(self.l2(F.dropout(y1, train=train)) + state["h2"])
            y3 = self.l3(F.dropout(y2, train=train))
        else:
            y1 = F.relu(self.l1(y0) + state["h1"])
            y2 = F.relu(self.l2(y1) + state["h2"])
            y3 = self.l3(y2, train=train)

        new_state = {
            "h1": y1,
            "h2": y2
        }
        return y3, new_state

    def forward(
            self,
            words: List[Variable],
            state: State,
            dropout: bool=False,
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
            dropout: bool=False,
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
        # self.xp = predictor.gpu

    def loss(
        self,
        words: [int],
        state: State,
        dropout: bool=False,
        train: bool=False
    ) -> Variable:
        if len(words) <= 1:
            raise Exception("word length error: >= 2")

        # word convert
        _words = [
            Variable(self.xp.array([word], dtype=self.xp.int32))
            for word in words
        ]

        ys, new_state = self.predictor(
            _words[:-1], state,
            dropout=dropout,
            train=train
        )
        loss = Variable(self.xp.zeros((), dtype=self.xp.float32))

        norm_array = []

        for y, t in zip(ys, _words[1:]):
            new_loss = F.softmax_cross_entropy(y, t)
            y_norm = self.xp.sqrt(y.data[0].dot(y.data[0]))
            norm_array.append(float(y_norm))
            if self.xp.isnan(y_norm):
                print(y_norm)
            loss += new_loss
        # print(
        #     "y norm mean: {}".format(
        #         sum(norm_array) / len(norm_array)
        #     ),
        # )

        len_words = Variable(self.xp.array(
            len(words) - 1,
            dtype=self.xp.float32
        ))
        return loss / len_words

    def __call__(
        self,
        words: int,
        state: State=None,
        dropout: bool=False,
        train: bool=False
    ) -> Tuple[Variable, Dict[str, Variable], Variable]:
        return self.loss(
            words, state,
            dropout=dropout,
            train=train
        )
