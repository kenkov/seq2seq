#! /usr/bin/env python
# coding:utf-8

from chainer import Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import rnn
from rtype import State
from typing import List, Tuple


# define classifier
Classifier = rnn.Classifier


class LSTM(Chain):
    def __init__(
            self,
            embed_dim: int,
            n_units: int=1000,
            gpu: int=-1,
    ):
        super(LSTM, self).__init__(
            embed=L.EmbedID(embed_dim, n_units),  # word embedding
            l1=L.Linear(n_units, n_units * 4),
            h1=L.Linear(n_units, n_units * 4),
            l2=L.Linear(n_units, n_units * 4),
            h2=L.Linear(n_units, n_units * 4),
            l3=L.Linear(n_units, embed_dim),
        )
        self.embed_dim = embed_dim
        self.n_units = n_units
        self.gpu = gpu

    def set_word_embedding(self, array):
        self.embed.W.data = array

    def reset_state(self):
        self.l1.reset_state()

    def forward_one(
            self,
            word: Variable,
            state: State,
            dropout: bool=False,
            train: bool=False
    ) -> Tuple[Variable, State]:
        y0 = self.embed(word)
        if dropout:
            h1_in = self.l1(F.dropout(y0, train=train)) + self.h1(state["h1"])
            c1, h1 = F.lstm(state["c1"], h1_in)
            h2_in = self.l2(F.dropout(h1, train=train)) + self.h2(state["h2"])
            c2, h2 = F.lstm(state["c2"], h2_in)
            h3 = self.l3(F.dropout(h2, train=train))
        else:
            h1_in = self.l1(y0) + self.h1(state["h1"])
            c1, h1 = F.lstm(state["c1"], h1_in)
            h2_in = self.l2(h1) + self.h2(state["h2"])
            c2, h2 = F.lstm(state["c2"], h2_in)
            h3 = self.l3(h2)

        new_state = {
            "h1": h1, "c1": c1,
            "h2": h2, "c2": c2,
        }
        return h3, new_state

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
            ) for key in ["h1", "c1", "h2", "c2"]
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
