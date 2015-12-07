#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":

    import sys
    import argparse
    from seq2seq import load_dictionary, RNN, decode
    import seq2seq_config
    import configparser

    # GPU config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_file', type=str,
                        help='config file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
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

    # encoder model
    dim = len(dictionary.keys())
    encoder_model = RNN(
        embed_dim=dim,
        n_units=n_units,
        gpu=gpu_flag
    )
    encoder_model.load("{}/model_0.npz".format(encoder_model_dir))

    # decoder model
    decoder_model = RNN(
        embed_dim=dim,
        n_units=n_units,
        gpu=gpu_flag
    )
    decoder_model.load("{}/model_0.npz".format(decoder_model_dir))

    for text in (_.strip() for _ in sys.stdin):
        ws = list(text)
        ws_with_s = list(reversed(ws)) + [seq2seq_config.END_SYMBOL]
        print(ws_with_s)
        decoded_words = decode(
            ws_with_s,
            encoder_model,
            decoder_model,
            dictionary,
        )

        answer_text = "".join(decoded_words[1:-1])
        print(answer_text)
