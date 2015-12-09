#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":
    from seq2seq import load_dictionary, RNN, train_decoder
    import argparse
    import configparser
    import os

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

    # load conversation sentences
    dictionary = load_dictionary(config["dict_file"])

    # encoder model
    dim = len(dictionary.keys())
    encoder_model = RNN(
        embed_dim=dim,
        n_units=int(config["n_units"]),
        gpu=gpu_flag
    )

    # decoder model
    decoder_model = RNN(
        embed_dim=dim,
        n_units=int(config["n_units"]),
        gpu=gpu_flag
    )

    encoder_model.load(
        "{}/model_0.npz".format(encoder_model_dir)
    )

    decoder_init_model_name = "{}/model_0.npz".format(decoder_model_dir)
    if os.path.exists(decoder_init_model_name):
        decoder_model.load(decoder_init_model_name)

    train_decoder(
        encoder_model,
        decoder_model,
        dictionary,
        config["conv_file"],
        decoder_model_dir,
        epoch_size=int(config["epoch_size"]),
        batch_size=int(config["batch_size"]),
        gpu=gpu_flag
    )
