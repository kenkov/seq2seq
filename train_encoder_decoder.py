#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":
    import argparse
    from seq2seq import train_encoder_decoder
    from util import load_dictionary, load_sentence
    import configparser
    import os
    from chainer import serializers
    from gensim.models import word2vec
    import logging
    import relu_rnn

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    # GPU config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_file', type=str,
                        help='config file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--type', '-t', default="relu", type=str,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    gpu_flag = args.gpu if args.gpu >= 0 else -1

    config_file = args.config_file
    parser_config = configparser.ConfigParser()
    parser_config.read(config_file)

    # params
    config = parser_config["CONFIG"]
    model_dir = config.get("model_dir")
    dict_file = config.get("dict_file")
    sent_file = config.get("sent_file")
    conv_file = config.get("conv_file")

    word2vec_init = config.getboolean("word2vec_init")
    word2vec_model_file = config.get("word2vec_model_file")

    min_freq = config.getint("min_freq")
    n_units = config.getint("n_units")
    epoch_size = config.getint("epoch_size")
    batch_size = config.getint("batch_size")
    dropout = config.getboolean("dropout")

    print("### options ###")
    print("model_dir: {}".format(model_dir))
    print("sent_file: {}".format(sent_file))
    print("conv_file: {}".format(conv_file))
    print("dict_file: {}".format(dict_file))
    print("word2vec_init: {}".format(word2vec_init))
    print("word2vec_model_file: {}".format(word2vec_model_file))
    print("min_freq: {}".format(min_freq))
    print("n_units: {}".format(n_units))
    print("epoch_size: {}".format(epoch_size))
    print("batch_size: {}".format(batch_size))
    print("dropout: {}".format(dropout))
    print("##############")

    # 辞書
    if os.path.exists(dict_file):
        dictionary = load_dictionary(dict_file)
    else:
        from util import create_dictionary
        dictionary = create_dictionary(
            [sent_file],
            min_freq=min_freq
        )
        dictionary.save(dict_file)

    # Prepare encoder RNN model
    dim = len(dictionary.keys())
    model_type = args.type
    if model_type == "relu":
        model = relu_rnn.Classifier(
            relu_rnn.ReLURNN(
                embed_dim=dim,
                n_units=n_units,
                gpu=args.gpu
            )
        )
    elif model_type == "lstm":
        import lstm
        model = lstm.Classifier(
            lstm.LSTM(
                embed_dim=dim,
                n_units=n_units,
                gpu=args.gpu
            )
        )
    else:
        raise Exception("model argment should be relu or lstm")

    # load model
    init_model_name = os.path.join(
        model_dir,
        "model.npz"
    )
    if os.path.exists(init_model_name):
        serializers.load_npz(init_model_name, model)
        print("load model {}".format(init_model_name))

    elif word2vec_init:
        # initialize embedding layer by word2vec
        import numpy as np

        if os.path.exists(word2vec_model_file):
            print("load word2vec model")
            word2vec_model = word2vec.Word2Vec.load(word2vec_model_file)
        else:
            print("start learning word2vec model")
            word2vec_model = word2vec.Word2Vec(
                load_sentence(sent_file),
                size=n_units,
                window=5,
                min_count=1,
                workers=4
            )
            print("save word2vec model")
            word2vec_model.save(word2vec_model_file)

        # initialize word embedding layer with word2vec
        initial_W = np.array([
            word2vec_model[dictionary[wid]]
            if dictionary[wid] in word2vec_model
            else np.array(
                [np.random.random() for _ in range(n_units)],
                dtype=np.float32
            )
            for wid in range(dim)],
            dtype=np.float32
        )
        not_found_words = []
        for wid in range(dim):
            if dictionary[wid] not in word2vec_model:
                not_found_words.append(dictionary[wid])
        print("{} are not found in word2vec model".format(not_found_words))
        model.predictor.set_word_embedding(initial_W)
        # print(initial_W)
        print("finish initializing word embedding with word2vec")

    train_encoder_decoder(
        model,
        dictionary,
        conv_file,
        model_dir,
        epoch_size,
        batch_size,
        dropout,
        gpu=gpu_flag
    )
