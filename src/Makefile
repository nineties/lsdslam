CFLAGS=-W -Wall -g -O2 -ffast-math -std=c99 -D_GNU_SOURCE -shared -fPIC
LIBS=-lm

all: liblsdslam.so liblsdslam_test.so

liblsdslam.so: lsdslam.c
	gcc -o $@ $^ $(CFLAGS) $(LIBS)

liblsdslam_test.so: lsdslam.c
	gcc -o $@ $^ $(CFLAGS) $(LIBS) -DBUILD_FOR_TEST
