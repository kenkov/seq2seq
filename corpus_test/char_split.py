#! /usr/bin/env python
# coding:utf-8

import sys

if __name__ == '__main__':
    source_fd = open(sys.argv[1])
    sent_fd = open(sys.argv[2], "w")
    sent_char_fd = open(sys.argv[3], "w")
    conv_char_fd = open(sys.argv[4], "w")
    for line in (_.strip() for _ in source_fd):
        orig, reply = line.split("\t")
        print(orig, file=sent_fd)
        print(reply, file=sent_fd)
        print(" ".join(orig), file=sent_char_fd)
        print(" ".join(reply), file=sent_char_fd)
        print(
            "{}\t{}".format(
                " ".join(orig),
                " ".join(reply)
            ),
            file=conv_char_fd
        )
