all:
	make -C src

test:
	nosetests -v -s

.PHONY: all test bench
