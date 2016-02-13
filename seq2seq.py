#! /usr/bin/env python
# coding:utf-8


import numpy as np
from datetime import datetime
from gensim import corpora
import seq2seq_config as config
from typing import List
import os
import sys

import chainer
from chainer import Variable, optimizers, serializers
from util import load_sentence, tokens2ids, load_conversation


def train_encoder(
    model,
    dictionary: corpora.Dictionary,
    sentence_file: str,
    model_dir: str,
    epoch_size: int=100,
    batch_size: int=30,
    dropout: bool=True,
    gpu: bool=False
) -> None:
    if gpu >= 0:
        model.to_gpu()
        print(model.xp)

    # setup SGD optimizer
    opt = optimizers.SGD()
    opt.setup(model)

    # optimizer hooks
    clip_threshold = 5.0
    print("set optimizer clip threshold: {}".format(clip_threshold))
    opt.add_hook(chainer.optimizer.GradientClipping(clip_threshold))

    # load conversation sentences
    sentences = load_sentence(sentence_file)
    data_size = len(sentences)

    print("data size: {}".format(data_size))
    for epoch in range(epoch_size):
        print("epoch {}".format(epoch))

        indexes = np.random.permutation(data_size)
        epoch_loss = 0  # int

        for bat_i in range(0, data_size, batch_size):
            forward_start_time = datetime.now()
            batch_loss = Variable(model.xp.zeros((), dtype=model.xp.float32))

            for index in indexes[bat_i:bat_i + batch_size]:
                input_words = sentences[index]
                # id のリストに変換する
                input_words_with_s = tokens2ids(
                    input_words,
                    dictionary,
                    verbose=False
                )

                # フォワード
                try:
                    new_loss = model(
                        input_words_with_s,
                        dropout=dropout,
                        state=None,
                        train=True
                    )
                    if model.xp.isnan(new_loss.data):
                        sys.exit(1)

                    batch_loss += new_loss
                except Exception:
                    print(index, input_words_with_s)
                    import traceback
                    traceback.print_exc()

            # 平均化
            batch_size_array = model.xp.array(
                batch_size,
                dtype=model.xp.float32
            )
            # if gpu:
            #     batch_size_array = cuda.to_gpu(batch_size_array)
            batch_loss = batch_loss / Variable(batch_size_array)
            epoch_loss += batch_loss.data

            # 時間計測
            forward_end_time = datetime.now()

            # 最適化
            opt_start_time = datetime.now()
            model.zerograds()
            batch_loss.backward()
            opt.update()
            opt_end_time = datetime.now()

            forward_delta = forward_end_time - forward_start_time
            opt_delta = opt_end_time - opt_start_time

            print_fmt = (
                "epoch {} batch {}: "
                "loss {}, grad L2 norm: {}, forward {}, optimizer {}"
            )
            print(print_fmt.format(
                epoch,
                int(bat_i / batch_size),
                batch_loss.data,
                opt.compute_grads_norm(),
                forward_delta,
                opt_delta,
            ))
            # save
            if ((bat_i / batch_size) + 1) % 100 == 0:
                serializers.save_npz(
                    os.path.join(
                        model_dir,
                        "model.npz"
                    ),
                    model
                )
            if ((bat_i / batch_size) + 1) % 1000 == 0:
                serializers.save_npz(
                    os.path.join(
                        model_dir,
                        "model_{}_{}_{}.npz".format(
                            epoch,
                            int(bat_i / batch_size) + 1,
                            datetime.now().strftime("%Y%m%d-%H%M%S")
                        )
                    ),
                    model
                )
        print("finish epoch {}, loss {}".format(
            epoch,
            epoch_loss / epoch_size
        ))
        # save
        serializers.save_npz(
            os.path.join(
                model_dir,
                "model.npz"
            ),
            model
        )
        serializers.save_npz(
            os.path.join(
                model_dir,
                "model_{}_{}_{}.npz".format(
                    epoch,
                    int(bat_i / batch_size) + 1,
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                )
            ),
            model
        )


def train_decoder(
    encoder_model,
    decoder_model,
    dictionary: corpora.Dictionary,
    conversation_file: str,
    decoder_model_dir: str,
    epoch_size: int=100,
    batch_size: int=30,
    dropout: bool=False,
    gpu: bool=False
) -> None:
    if gpu >= 0:
        encoder_model.to_gpu()
        decoder_model.to_gpu()
        print(encoder_model.xp)
        print(decoder_model.xp)

    # setup SGD optimizer
    opt = optimizers.SGD()
    opt.setup(decoder_model)

    # optimizer hooks
    clip_threshold = 5.0
    print("set optimizer clip threshold: {}".format(clip_threshold))
    opt.add_hook(chainer.optimizer.GradientClipping(clip_threshold))

    # load conversation sentences
    conversation = load_conversation(conversation_file, dictionary)
    data_size = len(conversation)

    print("data size: {}".format(data_size))
    for epoch in range(epoch_size):
        print("running epoch {}".format(epoch))
        indexes = np.random.permutation(range(data_size))
        epoch_loss = 0  # int

        for bat_i in range(0, data_size, batch_size):
            forward_start_time = datetime.now()
            batch_loss = Variable(decoder_model.xp.zeros((), dtype=np.float32))

            for index in indexes[bat_i:bat_i + batch_size]:
                pair_words = conversation[index]

                # encoder input words
                input_words_with_s = tokens2ids(pair_words[0], dictionary)
                ys, state = encoder_model.predictor.forward(
                    [Variable(
                        decoder_model.xp.array(
                            [word],
                            dtype=decoder_model.xp.int32
                        )
                    ) for word in input_words_with_s],
                    state=None,
                    dropout=False,
                    train=False
                )

                # decode
                output_words_with_s = tokens2ids(pair_words[1], dictionary)
                try:
                    new_loss = decoder_model(
                        output_words_with_s,
                        state=state,  # init_state を input の state にする
                        dropout=dropout,
                        train=True
                    )
                    batch_loss += new_loss
                except Exception:
                    print(index, input_words_with_s)
                    import traceback
                    traceback.print_exc()
            # 平均化
            batch_size_array = decoder_model.xp.array(
                batch_size,
                dtype=decoder_model.xp.float32
            )
            batch_loss = batch_loss / Variable(batch_size_array)
            epoch_loss += batch_loss.data

            # 時間計測
            forward_end_time = datetime.now()

            # 最適化
            opt_start_time = datetime.now()
            decoder_model.zerograds()
            batch_loss.backward()
            opt.update()
            opt_end_time = datetime.now()

            forward_delta = forward_end_time - forward_start_time
            opt_delta = opt_end_time - opt_start_time
            # print(
            #     ("decoder epoch {} batch {}: loss {}, "
            #      "forward {}, optimizer {},").format(
            #         epoch,
            #         int(bat_i / batch_size),
            #         batch_loss.data,
            #         forward_delta,
            #         opt_delta,
            #     )
            # )
            print_fmt = (
                "epoch {} batch {}: "
                "loss {}, grad L2 norm: {}, forward {}, optimizer {}"
            )
            print(print_fmt.format(
                epoch,
                int(bat_i / batch_size),
                batch_loss.data,
                opt.compute_grads_norm(),
                forward_delta,
                opt_delta,
            ))
            # save
            if ((bat_i / batch_size) + 1) % 100 == 0:
                serializers.save_npz(
                    os.path.join(
                        decoder_model_dir,
                        "model.npz"
                    ),
                    decoder_model
                )
            if ((bat_i / batch_size) + 1) % 1000 == 0:
                serializers.save_npz(
                    os.path.join(
                        decoder_model_dir,
                        "model_{}_{}_{}.npz".format(
                            epoch,
                            int(bat_i / batch_size) + 1,
                            datetime.now().strftime("%Y%m%d-%H%M%S")
                        )
                    ),
                    decoder_model
                )
        print("finish epoch {}, loss {}".format(
            epoch,
            epoch_loss / epoch_size
        ))
        # save
        serializers.save_npz(
            os.path.join(
                decoder_model_dir,
                "model.npz"
            ),
            decoder_model
        )
        serializers.save_npz(
            os.path.join(
                decoder_model_dir,
                "model_{}_{}_{}.npz".format(
                    epoch,
                    int(bat_i / batch_size) + 1,
                    datetime.now().strftime("%Y%m%d-%H%M%S")
                )
            ),
            decoder_model
        )


def decode(
        words: List[str],
        encoder_model,
        decoder_model,
        dictionary: corpora.Dictionary,
        dropout: bool=False,
) -> List[str]:

    input_words_with_s = tokens2ids(words, dictionary, verbose=True)
    # input words の hidden state を求める
    ys, state = encoder_model.predictor.forward(
        [Variable(
            decoder_model.xp.array([word], dtype=decoder_model.xp.int32)
        ) for word in input_words_with_s],
        state=None,
        dropout=dropout,
        train=False
    )

    word = dictionary.token2id[config.END_SYMBOL]
    lst = [config.END_SYMBOL]

    while True:
        y, state = decoder_model.predictor.forward_one(
            Variable(
                decoder_model.xp.array([word], dtype=decoder_model.xp.int32)
            ),
            state,
            dropout=dropout,
            train=False
        )
        word = y.data[0].argmax()

        lst.append(dictionary[word])
        if dictionary[word] == config.END_SYMBOL or len(lst) >= 100:
            return lst
