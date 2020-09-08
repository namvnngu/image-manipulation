clean: 
	rm -rf ./bin/*.o
build-omp:
	export OMP_NUM_THREADS=$(t) && g++ main.cpp -fopenmp -Wall -g -o bin/output.o
build-ocv: main.cpp
	g++ main.cpp -Wall -g -o bin/output.o `pkg-config --cflags --libs opencv`
run: ./bin/output.o
	./bin/output.o

