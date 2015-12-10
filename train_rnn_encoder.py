#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":
    import argparse
    from seq2seq import RNN, train_encoder, load_dictionary
    import seq2seq_config as config
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

    # files
    encoder_model_dir = config["encoder_model_dir"]

    # 辞書
    if os.path.exists(config["dict_file"]):
        dictionary = load_dictionary(config["dict_file"])
    else:
        from seq2seq import create_dictionary
        dictionary = create_dictionary(
            [config["sent_file"],
             ],
            min_freq=int(config["min_freq"])
        )
        dictionary.save(config["dict_file"])

    # Prepare RNNLM model
    dim = len(dictionary.keys())
    model = RNN(
        embed_dim=dim,
        n_units=int(config["n_units"]),
        gpu=gpu_flag
    )

    init_model_name = "{}/model_0.npz".format(encoder_model_dir)
    if os.path.exists(init_model_name):
        model.load(init_model_name)

    train_encoder(
        model,
        dictionary,
        config["sent_file"],
        encoder_model_dir,
        epoch_size=int(config["epoch_size"]),
        batch_size=int(config["batch_size"]),
        gpu=gpu_flag
    )
