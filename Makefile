# Define constants
CXX = g++
INPUT = ./src/*.cpp 
OUTPUT = putput.o
HEADER = ./include/*.h 
BIN_DIR = ./bin

# Flags
OPEN_CV_FLAG = `pkg-config --cflags --libs opencv`
OPENMP_FLAG = -fopenmp
CFLAGS = -O2 -Wall -g -Iinclude 

# Build
OPENMP_BUILD = ${CXX} ${INPUT} ${OPENMP_FLAG} ${CFLAGS} -o ${BIN_DIR}/${OUTPUT} ${OPEN_CV_FLAG}
OPENCV_BUILD = ${CXX} ${INPUT} ${CFLAGS} -o ${BIN_DIR}/${OUTPUT} ${OPEN_CV_FLAG}
MPI_BUILD = mpicxx ${INPUT} ${OPENMP_FLAG} ${CFLAGS} -o ${BIN_DIR}/${OUTPUT} ${OPEN_CV_FLAG}

# Commands
clear: 
	rm -rf ${BIN_DIR}/*.o ./output/*.jpg ./output/*.avi ./output/*.jpeg
build-omp: ${INPUT} ${HEADER}
	@echo Building..
	export OMP_NUM_THREADS=$(t) && ${OPENMP_BUILD}
	@echo Building complete
build-ocv: ${INPUT} ${HEADER}
	@echo Building..
	${OPENCV_BUILD}
	@echo Building complete
build: ${INPUT} ${HEADER}
	@echo Building..
	export OMP_NUM_THREADS=$(t) && ${MPI_BUILD}
	@echo Building complete
run: ${BIN_DIR}/${OUTPUT}
	mpirun -np ${n} ${BIN_DIR}/${OUTPUT}
run-omp:
	${BIN_DIR}/${OUTPUT} $(s)


