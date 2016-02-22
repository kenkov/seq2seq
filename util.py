#! /usr/bin/env python
# coding:utf-8


import numpy as np
from gensim import corpora
import seq2seq_config as config
from typing import List, Tuple
from rtype import NIntArray


def ia(lst: List[int]) -> NIntArray:
    return np.array(lst, dtype=np.int32)


def tokens2ids(
    tokens: List[str],
    dictionary: corpora.Dictionary,
    verbose: bool=False
) -> List[int]:
    if verbose:
        not_found_lst = [
            word for word in tokens if word not in dictionary.token2id
        ]
        if not_found_lst:
            print("not found in dict: {}".format(
                not_found_lst
            ))
        for word in tokens:
            if word in dictionary and dictionary.token2id[word] < 0:
                raise("word id < 0: {}".format(word))

    # 未知語は UNK にする
    return [
        dictionary.token2id[word] if word in dictionary.token2id
        else dictionary.token2id[config.UNK_SYMBOL]
        for word in tokens
    ]


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
    with_symbol: bool=True
) -> List[str]:
    """コーパスをロードする。"""
    if with_symbol:
        tokens = [
            list(sent.split()) + [config.END_SYMBOL]
            for sent in (_.strip() for _ in open(filename))
        ]
    else:
        tokens = [
            list(sent.split())
            for sent in (_.strip() for _ in open(filename))
        ]

    print("loaded sentences from {}".format(filename))
    return tokens


def load_conversation(
    filename: str,
    dictionary: corpora.Dictionary,
    with_symbol: bool=True
) -> (Tuple[List[int], List[int]]):
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
        Tuple[List[int], List[int]]
            ([1, 2, 3], [4, 5, 6])
    """
    # load conversation sentences
    if with_symbol:
        tokens = [(
            # [config.START_SYMBOL] + src.split() + [config.END_SYMBOL],
            # [config.START_SYMBOL] + dst.split() + [config.END_SYMBOL]
            list(src.split()) + [config.END_SYMBOL],
            [config.END_SYMBOL] + dst.split() + [config.END_SYMBOL]
        ) for src, dst in (sent.split(config.SEPARATOR)
                           for sent in open(filename))
        ]
    else:
        tokens = [
            (list(src.split()), dst.split())
            for src, dst in (sent.split(config.SEPARATOR)
                             for sent in open(filename))
        ]

    print("loaded sentences from {}".format(filename))
    return tokens
