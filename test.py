#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":

    import sys
    import argparse
    from seq2seq import decode
    from util import load_dictionary
    import configparser
    import os
    from chainer import serializers

    # GPU config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_file', type=str,
                        help='config file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--type', '-t', default="relu", type=str,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    gpu_flag = True if args.gpu >= 0 else False

    config_file = args.config_file
    parser_config = configparser.ConfigParser()
    parser_config.read(config_file)
    config = parser_config["CONFIG"]
    # config["SEPARATOR"] = bytes(
    #     config["DEFAULT"]["SEPARATOR"], "utf-8"
    # ).decode("unicode_escape")

    # params
    model_dir = config["model_dir"]
    n_units = int(config["n_units"])

    # load conversation sentences
    dictionary = load_dictionary(config["dict_file"])

    # Prepare encoder RNN model
    dim = len(dictionary.keys())
    model_type = args.type
    if model_type == "relu":
        import relu_rnn
        model = relu_rnn.Classifier(
            relu_rnn.ReLURNN(
                embed_dim=dim,
                n_units=int(config["n_units"]),
                gpu=args.gpu
            )
        )
    elif model_type == "lstm":
        import lstm
        model = lstm.Classifier(
            lstm.LSTM(
                embed_dim=dim,
                n_units=int(config["n_units"]),
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
    else:
        raise Exception("learn model first")

    for text in (_.strip() for _ in sys.stdin):
        ws = text.split()
        print("> {}".format(" ".join(ws)))
        for order in range(1, 5):
            decoded_words = decode(
                ws,
                model,
                model,
                dictionary,
                order=order
            )

            answer_text = "".join(decoded_words[1:-1])
            print("{}".format(answer_text))
