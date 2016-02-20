PYTHON=python

.PHONY: test clean

train:
	${PYTHON} train.py test.ini -g0 -tlstm

test:
	${PYTHON} test.py test.ini -g0 -tlstm <./corpus_test/sent.char.txt

clean:
	rm model_test/*
