TOKENIZER=mecab -Owakati

char: split _char remove_tmp remove_char_tmp
word: split _word remove_tmp remove_word_tmp
all:  split _char _word remove_tmp remove_char_tmp remove_word_tmp

split:
	cut -f1 conv.txt >_orig.tmp.txt
	cut -f2 conv.txt >_reply.tmp.txt

_word:
	${TOKENIZER} _orig.tmp.txt >_orig.word.tmp.txt
	${TOKENIZER} _reply.tmp.txt >_reply.word.tmp.txt
	paste _orig.word.tmp.txt _reply.word.tmp.txt >conv.word.txt
	cat _orig.word.tmp.txt _reply.word.tmp.txt >sent.word.txt

_char:
	sed -e "s/\(.\)/\1 /g" _orig.tmp.txt | sed -e "s/\s$$//g" >_orig.char.tmp.txt
	sed -e "s/\(.\)/\1 /g" _reply.tmp.txt | sed -e "s/\s$$//g" >_reply.char.tmp.txt
	paste _orig.char.tmp.txt _reply.char.tmp.txt >conv.char.txt
	cat _orig.char.tmp.txt _reply.char.tmp.txt >sent.char.txt

remove_tmp:
	rm {_orig,_reply}.tmp.txt

remove_word_tmp:
	rm {_orig,_reply}.word.tmp.txt

remove_char_tmp:
	rm {_orig,_reply}.char.tmp.txt

clean:
	rm sent.txt {sent,conv}.word.txt {sent,conv}.char.txt
