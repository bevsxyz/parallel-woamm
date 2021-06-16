CC=nvcc -std=c++11

all: whale

whale: *.o
	$(CC) -o philox4 philox4.o

*.o: philox4.cu
	$(CC) -o philox4.o -c philox4.cu

clean:
	rm -f philox4.o philox4