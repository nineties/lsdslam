CFLAGS=-W -Wall -g -O2 -ffast-math -std=c99 -D_GNU_SOURCE -shared -FPIC
LIBS=-lm
liblsdslam.so: lsdslam.c
	gcc -o $@ $^ $(CFLAGS) $(LIBS)
