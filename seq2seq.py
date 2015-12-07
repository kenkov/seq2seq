#! /usr/bin/env python
# coding:utf-8


import numpy as np
from chainer import FunctionSet, Variable, optimizers, cuda
import chainer.functions as F
from datetime import datetime
from gensim import corpora
import seq2seq_config as config
import os
from typing import List, Tuple, Dict


# type definition
NArray = np.ndarray


def ia(lst: List[int]) -> NArray:
    return np.array(lst, dtype=np.int32)


def tokens2ids(
    tokens: List[str],
    dictionary: corpora.Dictionary,
    verbose: bool=False
) -> NArray:
    if verbose:
        not_found_lst = [
            word for word in tokens if word not in dictionary.token2id
        ]
        if not_found_lst:
            print("not found in dict: {}".format(
                not_found_lst
            ))

    # 未知語は UNK にする
    return ia([
        dictionary.token2id[word] if word in dictionary.token2id
        else dictionary.token2id[config.UNK_SYMBOL]
        for word in tokens
    ])


def create_dictionary(
    corpuses: List[str],
    min_freq: int=1,
    with_symbol=True
) -> corpora.Dictionary:
    """辞書を作成する。

    Args:
        corpuses ([str]): コーパスファイル名。一行が

                今日 は 疲れ た 。

            のように、単語毎にスペースで分割された文がはいっている必要がある。
        save_file (str): 保存するファイル名
        with_symbol (bool): START_SYMBOL, END_SYMBOL を追加するかどうか
    Returns:
        corpora.Dictionary: 辞書
    """
    # make and load dictionary
    dic = corpora.Dictionary()
    print("creating dictionary".format(len(dic.values())))
    if with_symbol:
        dic.add_documents([[config.START_SYMBOL]])
        dic.add_documents([[config.END_SYMBOL]])
        dic.add_documents([[config.UNK_SYMBOL]])
    print("add start and end symbols in dictionary".format(
        len(dic.values())
    ))
    for corpus in corpuses:
        print("adding words from {} to dictionary".format(corpus))
        dic.add_documents(
            line.split() for line in open(corpus)
        )

    # filter words
    ones_ids = {
        tokenid for tokenid, docfreq in dic.dfs.items() if docfreq <= min_freq
    }
    # start symbol と end symbold は含めない
    dic.filter_tokens(ones_ids - {dic.token2id[config.START_SYMBOL],
                                  dic.token2id[config.END_SYMBOL],
                                  dic.token2id[config.UNK_SYMBOL],
                                  })
    dic.compactify()
    if with_symbol:
        if config.START_SYMBOL in dic.token2id and \
                config.END_SYMBOL in dic.token2id and \
                config.UNK_SYMBOL in dic.token2id:
            pass
        else:
            raise Exception("START/END/UNK symbol are not in dictionary")

    return dic


def save_dictionary(
    dic: corpora.Dictionary,
    filename: str
) -> None:
    dic.save(filename)
    print("saved dictionary: {} items to {}".format(
        len(dic.values()), filename
    ))


def load_dictionary(filename: str) -> corpora.Dictionary:
    """辞書をロードする。

    Args:
        filename (str): ファイル名。
    Returns:
        corpora.Dictionary: 辞書。
    """
    dic = corpora.Dictionary.load(filename)
    # if with_symbol and \
    #         not (dic.token2id["<S>"] == 0 and dic.token2id["</S>"] == 1):
    #     raise Exception("<S> and </S> ids should be 0 and 1")

    print("load dictionary: {} items".format(len(dic.values())))
    # print([item for item in dic.items()][:10])
    return dic


def load_sentence(
    filename: str,
    dictionary: corpora.Dictionary,
    with_symbol: bool=True
) -> List[str]:
    """コーパスをロードする。"""
    if with_symbol:
        tokens = [
            # [config.START_SYMBOL] + sent.split() + [config.END_SYMBOL]
            list(reversed(sent.split())) + [config.END_SYMBOL]
            for sent in (_.strip() for _ in open(filename))
        ]
    else:
        tokens = [
            list(reversed(sent.split()))
            for sent in (_.strip() for _ in open(filename))
        ]

    print("loaded sentences from {}".format(filename))
    return tokens


def load_conversation(
    filename: str,
    dictionary: corpora.Dictionary,
    with_symbol: bool=True
) -> (Tuple[NArray, NArray]):
    """対話コーパスをロードする。

    Args:
        filename (str): コーパスファイル
            コーパスファイルの一行は

                何 が 好き です か ？,Python が 好き です 。

            のように、(source string, separator, target string) が単語ごとに
            スペースで分割された文字列が格納されている必要がある。
        dictionary:
        with_symbol:
    Returns:
        ([np.ndarray], [np.ndarray]): np.int32 の単語 id のリスト。

            ([1, 2, 3], [4, 5, 6])
    """
    # load conversation sentences
    if with_symbol:
        tokens = [(
            # [config.START_SYMBOL] + src.split() + [config.END_SYMBOL],
            # [config.START_SYMBOL] + dst.split() + [config.END_SYMBOL]
            list(reversed(src.split())) + [config.END_SYMBOL],
            [config.END_SYMBOL] + dst.split() + [config.END_SYMBOL]
        ) for src, dst in (sent.split(config.SEPARATOR)
                           for sent in open(filename))
        ]
    else:
        tokens = [
            (list(reversed(src.split())), dst.split())
            for src, dst in (sent.split(config.SEPARATOR)
                             for sent in open(filename))
        ]

    print("loaded sentences from {}".format(filename))
    return tokens


class RNN:
    def __init__(
        self,
        embed_dim: int,
        n_units: int=250,
        gpu=False
    ):

        model = self.define_model(embed_dim, n_units)
        # initialize parameter
        for param in model.parameters:
            param[:] = np.random.uniform(-0.1, 0.1, param.shape)

        if gpu:
            model = model.to_gpu()
            print("converted RNN model to use GPU")

        optimizer = optimizers.SGD(lr=1.)
        optimizer.setup(model)

        self.model = model
        self.embed_dim = embed_dim
        self.n_units = n_units
        self.optimizer = optimizer
        self.gpu = gpu
        self.mod = cuda if gpu else np

        lstm_nodes = int((len(self.model.parameters) - 3) / 4)

        print(
            ("RNN structure: "
             "word -> {} -> {} -> {} nodes, {} LSTM layers -> {}, GPU flag: {}"
             ).format(
                 embed_dim, n_units, n_units, lstm_nodes, embed_dim, gpu
            )
        )

    def define_model(
        self,
        embed_dim: int,
        n_units: int
    ):
        # Prepare RNNLM model
        model = FunctionSet(
            embed=F.EmbedID(embed_dim, n_units),
            l1_x=F.Linear(n_units, 4 * n_units),
            l1_h=F.Linear(n_units, 4 * n_units),
            l2_x=F.Linear(n_units, 4 * n_units),
            l2_h=F.Linear(n_units, 4 * n_units),
            l3_x=F.Linear(n_units, 4 * n_units),
            l3_h=F.Linear(n_units, 4 * n_units),
            l4_x=F.Linear(n_units, 4 * n_units),
            l4_h=F.Linear(n_units, 4 * n_units),
            l5_x=F.Linear(n_units, 4 * n_units),
            l5_h=F.Linear(n_units, 4 * n_units),
            l6_x=F.Linear(n_units, 4 * n_units),
            l6_h=F.Linear(n_units, 4 * n_units),
            l7=F.Linear(n_units, embed_dim)
        )
        return model

    def save(
        self,
        directory: str,
        suffix: str=""
    ) -> None:
        """
        """
        params = self.model.parameters
        filename = "{}_{}".format(
            "model",
            suffix
        )
        save_path = os.path.join(directory, filename)
        np.savez(
            save_path,
            *[cuda.to_cpu(array) if self.gpu else array for array in params]
        )
        print("saved model as {}.npz".format(save_path))

    def load(
        self,
        filename: str
    ) -> None:
        load_data = np.load(filename)
        sorted_data = [
            val for key, val in
            sorted(
                load_data.items(),
                key=lambda x: int(x[0].split("_")[1])
            )
        ]

        self.model.copy_parameters_from(
            sorted_data
        )
        print("loaded RNN model: {}".format(filename))

    def forward_one_step(
        self,
        cur_word: str,
        next_word: str,
        state: Dict[str, Variable],
        train: bool=True
    ) -> Tuple[Variable, Dict[str, Variable], Variable]:
        if self.gpu:
            cur_word = cuda.to_gpu(cur_word)
            next_word = cuda.to_gpu(next_word)

        x = Variable(cur_word)
        t = Variable(next_word)
        h0 = self.model.embed(x)
        # h1_in = self.model.l1_x(F.dropout(h0, train=train)) + \
        #     self.model.l1_h(state['h1'])
        h1_in = self.model.l1_x(h0) + self.model.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        # h2_in = self.model.l2_x(F.dropout(h1)) + self.model.l2_h(state['h2'])
        # c2, h2 = F.lstm(state['c2'], h2_in)
        h2_in = self.model.l2_x(h1) + self.model.l2_h(state['h2'])
        c2, h2 = F.lstm(state['c2'], h2_in)

        # h3_in = self.model.l3_x(F.dropout(h2)) + self.model.l3_h(state['h3'])
        # c3, h3 = F.lstm(state['c3'], h3_in)
        h3_in = self.model.l3_x(h2) + self.model.l3_h(state['h3'])
        c3, h3 = F.lstm(state['c3'], h3_in)

        # h4_in = self.model.l4_x(F.dropout(h3)) + self.model.l4_h(state['h4'])
        # c4, h4 = F.lstm(state['c4'], h4_in)
        h4_in = self.model.l4_x(h3) + self.model.l4_h(state['h4'])
        c4, h4 = F.lstm(state['c4'], h4_in)

        h5_in = self.model.l5_x(h4) + self.model.l5_h(state['h5'])
        c5, h5 = F.lstm(state['c5'], h5_in)
        h6_in = self.model.l6_x(h5) + self.model.l6_h(state['h6'])
        c6, h6 = F.lstm(state['c6'], h6_in)

        # y = self.model.l5(F.dropout(h1, train=train))
        y = self.model.l7(h6)
        state = {'c1': c1, 'h1': h1,
                 'c2': c2, 'h2': h2,
                 'c3': c3, 'h3': h3,
                 'c4': c4, 'h4': h4,
                 'c5': c5, 'h5': h5,
                 'c6': c6, 'h6': h6,
                 }

        # y はソフトマックス関数にわたしていない
        return F.softmax_cross_entropy(y, t), state, y

    def make_initial_state(self) -> Dict[str, Variable]:
        return {
            name: Variable(self.mod.zeros((1, self.n_units), dtype=np.float32))
            for name in (
                'c1', 'h1',
                'c2', 'h2',
                'c3', 'h3',
                'c4', 'h4',
                'c5', 'h5',
                'c6', 'h6',
            )
        }

    def forward(
        self,
        words: List[str],
        init_state: Dict[str, Variable]=None,
        train: bool=True
    ) -> Tuple[Variable, Dict[str, Variable], Variable]:

        if len(words) <= 1:
            raise Exception("word length error: >= 2")

        # recurrent のノードは 0 で初期化しておく
        # h = Variable(np.zeros((1, 500), dtype=np.float32))
        state = init_state if init_state else self.make_initial_state()

        loss = Variable(self.mod.zeros((), dtype=np.float32))
        for cur_word, next_word in zip(words, words[1:]):
            new_loss, state, output_var = self.forward_one_step(
                cur_word.reshape((1,)), next_word.reshape((1,)), state, train
            )
            loss += new_loss
        # loss 計算には関係ないが、最後の要素をながす
        # _, state, output_var = self.forward_one_step(
        #     next_word.reshape((1,)),
        #     next_word.reshape((1,)),  # ここは何でもいい
        #     state,
        #     train
        # )

        word_len_array = np.array(len(words) + 1, dtype=np.float32)
        if self.gpu:
            word_len_array = cuda.to_gpu(word_len_array)
        word_len_var = Variable(
            word_len_array
        )

        return (
            loss / word_len_var,
            state,
            output_var
        )


def train_encoder(
    model: FunctionSet,
    dictionary: corpora.Dictionary,
    sentence_file: str,
    model_dir: str,
    epoch_size: int=100,
    batch_size: int=30,
    gpu: bool=False
) -> None:
    mod = cuda if gpu else np

    # load conversation sentences
    sentences = load_sentence(sentence_file, dictionary)
    data_size = len(sentences)

    print("data size: {}".format(data_size))
    for epoch in range(epoch_size):
        print("running epoch {}".format(epoch))

        indexes = np.random.permutation(range(data_size))
        epoch_loss = 0  # int

        for bat_i in range(0, data_size, batch_size):
            forward_start_time = datetime.now()

            batch_loss = Variable(mod.zeros((), dtype=np.float32))
            for index in indexes[bat_i:bat_i + batch_size]:
                input_words = sentences[index]
                # print("{}".format(input_words))

                # id のリストに変換する
                input_words_with_s = tokens2ids(
                    input_words,
                    dictionary,
                    verbose=False
                )

                # フォワード
                new_loss, _, _ = model.forward(input_words_with_s, train=True)
                batch_loss += new_loss
            # 平均化
            batch_size_array = np.array(batch_size, dtype=np.float32)
            if gpu:
                batch_size_array = cuda.to_gpu(batch_size_array)
            batch_loss = batch_loss / Variable(batch_size_array)
            epoch_loss += batch_loss.data * batch_size

            # 時間計測
            forward_end_time = datetime.now()

            # 最適化
            opt_start_time = datetime.now()
            model.optimizer.zero_grads()
            batch_loss.backward()
            model.optimizer.update()
            opt_end_time = datetime.now()

            print("epoch {} batch {} loss {} forward {} backward {}".format(
                epoch,
                int(bat_i / batch_size),
                batch_loss.data,
                forward_end_time - forward_start_time,
                opt_end_time - opt_start_time,
            ))
            # save
            if ((bat_i / batch_size) + 1) % 100 == 0:
                model.save(
                    directory=model_dir,
                    suffix="{}".format(epoch)
                )

            # save
            if ((bat_i / batch_size) + 1) % 10000 == 0:
                model.save(
                    directory=model_dir,
                    suffix="{}_{}_{}".format(
                        epoch,
                        int(bat_i / batch_size) + 1,
                        datetime.now().strftime("%Y%m%d-%H%M%S")
                    )
                )
        # save
        model.save(
            directory=model_dir,
            suffix="{}".format(epoch)
        )
        # epoch log
        print("epoch: {}, loss {} ".format(
            epoch,
            epoch_loss / data_size
        ))


def train_decoder(
    encoder_model: FunctionSet,
    decoder_model: FunctionSet,
    dictionary: corpora.Dictionary,
    conversation_file: str,
    decoder_model_dir: str,
    epoch_size: int=100,
    batch_size: int=30,
    gpu: bool=False
) -> None:
    mod = cuda if gpu else np

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

            batch_loss = Variable(mod.zeros((), dtype=np.float32))
            for index in indexes[bat_i:bat_i + batch_size]:
                pair_words = conversation[index]
                # print("input: {}\noutput:{}".format(
                #     pair_words[0], pair_words[1]
                # ))
                # input
                input_words_with_s = tokens2ids(pair_words[0], dictionary)
                _, state, _ = encoder_model.forward(
                    input_words_with_s,
                    train=False  # train しないでながす
                )

                # output
                output_words_with_s = tokens2ids(pair_words[1], dictionary)
                new_loss, _, _ = decoder_model.forward(
                    output_words_with_s,
                    init_state=state,  # init_state を input の state にする
                    train=True
                )
                batch_loss += new_loss
            # 平均化
            batch_size_array = np.array(batch_size, dtype=np.float32)
            if gpu:
                batch_size_array = cuda.to_gpu(batch_size_array)
            batch_loss = batch_loss / Variable(batch_size_array)
            epoch_loss += batch_loss.data * batch_size

            # 時間計測
            forward_end_time = datetime.now()

            # 最適化
            opt_start_time = datetime.now()
            decoder_model.optimizer.zero_grads()
            batch_loss.backward()
            decoder_model.optimizer.update()
            opt_end_time = datetime.now()

            print(
                ("decoder epoch {} batch {} loss {} "
                 "forward {} backward {}").format(
                     epoch,
                     int(bat_i / batch_size),
                     batch_loss.data,
                     forward_end_time - forward_start_time,
                     opt_end_time - opt_start_time)
            )
            if ((bat_i / batch_size) + 1) % 100 == 0:
                decoder_model.save(
                    directory=decoder_model_dir,
                    suffix="{}".format(epoch)
                )
            # save
            if ((bat_i / batch_size) + 1) % 1000 == 0:
                decoder_model.save(
                    directory=decoder_model_dir,
                    suffix="{}_{}_{}".format(
                        epoch,
                        int(bat_i / batch_size) + 1,
                        datetime.now().strftime("%Y%m%d-%H%M%S")
                    )
                )

        decoder_model.save(
            directory=decoder_model_dir,
            suffix="{}".format(epoch)
        )
        print("epoch: {}, loss {} ".format(
            epoch,
            epoch_loss / data_size
        ))


def encode(
    words: List[str],
    encoder_model,
    dictionary,
) -> List[str]:

        input_words_with_s = tokens2ids(words, dictionary, verbose=True)
        # input words の hidden state を求める
        _, state, output_var = encoder_model.forward(
            input_words_with_s,
            train=False
        )

        next_word = output_var.data.argmax()
        lst = [w for w in words] + [dictionary[next_word]]
        while True:
            if dictionary[next_word] == config.END_SYMBOL or len(lst) >= 100:
                return lst
            cur_word = next_word
            _, state, output_var = encoder_model.forward_one_step(
                cur_word.reshape((1,)).astype(np.int32),
                cur_word.reshape((1,)).astype(np.int32),  # ここは使わない
                state,
                train=False
            )
            next_word = output_var.data.argmax()
            lst.append(dictionary[next_word])


def decode(
    words: List[str],
    encoder_model: FunctionSet,
    decoder_model: FunctionSet,
    dictionary: corpora.Dictionary,
) -> List[str]:

        input_words_with_s = tokens2ids(words, dictionary, verbose=True)
        # input words の hidden state を求める
        _, state, _ = encoder_model.forward(input_words_with_s, train=False)

        next_word = ia(dictionary.token2id[config.END_SYMBOL])
        lst = [config.END_SYMBOL]
        while True:
            cur_word = next_word
            _, state, output_var = decoder_model.forward_one_step(
                cur_word.reshape((1,)).astype(np.int32),
                cur_word.reshape((1,)).astype(np.int32),  # ここはなんでもいい
                state,
                train=False
            )
            next_word = output_var.data[0].argmax()

            # if len(lst) == 0 and dictionary[next_word] == config.END_SYMBOL:
            #     next_word = output_var.data[0].argsort()[-2]

            # for idx in output_var.data[0].argsort()[-5:]:
            #     print(idx, dictionary[idx], output_var.data[0][idx])
            # print("---")

            lst.append(dictionary[next_word])
            if dictionary[next_word] == config.END_SYMBOL or len(lst) >= 100:
                return lst
