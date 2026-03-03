SRC = .
CFLAGS = -O3 -ffast-math -march=native
LDFLAGS = 
CC = gcc

OBJ1 = o/iba.o o/misc.o

all : iba

iba: $(OBJ1)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

o/iba.o : $(SRC)/iba.c $(SRC)/iba.h $(SRC)/Makefile
	$(CC) $(CFLAGS) -c $(SRC)/iba.c -o $@

o/misc.o : $(SRC)/misc.c $(SRC)/iba.h $(SRC)/Makefile
	$(CC) $(CFLAGS) -c $(SRC)/misc.c -o $@

