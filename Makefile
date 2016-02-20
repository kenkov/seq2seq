PYTHON=python

.PHONY: test clean

test:
	${PYTHON} train_encoder_decoder.py test.ini -g0 -tlstm

clean:
	rm model_test/*
