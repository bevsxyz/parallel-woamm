CC=nvcc -std=c++11

all: whale

whale: *.o
	$(CC) -o philox4 philox4.o
	$(CC) -o MTGP32 MTGP32.o
	$(CC) -o MRG32k3a MRG32k3a.o

*.o: philox4.cu MTGP32.cu MRG32k3a.cu
	$(CC) -o philox4.o -c philox4.cu
	$(CC) -o MTGP32.o -c MTGP32.cu
	$(CC) -o MRG32k3a.o -c MRG32k3a.cu

clean:
	rm -f philox4.o philox4 MTGP32.o MTGP32 MRG32k3a.o MRG32k3a