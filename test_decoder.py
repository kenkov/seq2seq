#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":

    import sys
    import argparse
    from seq2seq import decode
    from util import load_dictionary
    import seq2seq_config
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
    encoder_model_dir = config["encoder_model_dir"]
    decoder_model_dir = config["decoder_model_dir"]
    n_units = int(config["n_units"])

    # load conversation sentences
    dictionary = load_dictionary(config["dict_file"])

    # Prepare encoder RNN model
    dim = len(dictionary.keys())
    model_type = args.type
    if model_type == "relu":
        import relu_rnn
        encoder_model = relu_rnn.Classifier(
            relu_rnn.ReLURNN(
                embed_dim=dim,
                n_units=int(config["n_units"]),
                gpu=args.gpu
            )
        )
        decoder_model = relu_rnn.Classifier(
            relu_rnn.ReLURNN(
                embed_dim=dim,
                n_units=int(config["n_units"]),
                gpu=args.gpu
            )
        )
    elif model_type == "lstm":
        import lstm
        encoder_model = lstm.Classifier(
            lstm.LSTM(
                embed_dim=dim,
                n_units=int(config["n_units"]),
                gpu=args.gpu
            )
        )
        decoder_model = lstm.Classifier(
            lstm.LSTM(
                embed_dim=dim,
                n_units=int(config["n_units"]),
                gpu=args.gpu
            )
        )
    else:
        raise Exception("model argment should be relu or lstm")

    # load model

    init_encoder_model_name = os.path.join(
        encoder_model_dir,
        "model.npz"
    )
    if os.path.exists(init_encoder_model_name):
        serializers.load_npz(init_encoder_model_name, encoder_model)
        print("load encoder model {}".format(init_encoder_model_name))
    else:
        raise Exception("learn encoder model first")

    init_decoder_model_name = os.path.join(
        decoder_model_dir,
        "model.npz"
    )
    if os.path.exists(init_decoder_model_name):
        serializers.load_npz(init_decoder_model_name, decoder_model)
        print("load decoder model {}".format(init_decoder_model_name))
    else:
        raise Exception("learn decoder model first")

    for text in (_.strip() for _ in sys.stdin):
        ws = text.split()
        ws_with_s = ws + [seq2seq_config.END_SYMBOL]
        print(ws_with_s)
        decoded_words = decode(
            ws_with_s,
            encoder_model,
            decoder_model,
            dictionary,
        )

        answer_text = "".join(decoded_words[1:-1])
        print(answer_text)
