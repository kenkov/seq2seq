main: char word
char:
	python char_split.py conv.txt sent.txt sent.char.txt conv.char.txt

word:
	kytea sent.txt -notags >sent.word.txt
	kytea conv.txt -notags >conv.word.txt

clean:
	rm sent.txt {sent,conv}.word.txt {sent,conv}.char.txt
