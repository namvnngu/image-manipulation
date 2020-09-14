# Define constants
CXX = g++
INPUT = ./src/*.cpp 
OUTPUT = output.o
HEADER = ./include/*.h 
BIN_DIR = ./bin

# Flags
OPEN_CV_FLAG = `pkg-config --cflags --libs opencv`
OPENMP_FLAG = -fopenmp
CFLAGS = -O2 -Wall  -Iinclude 

# Build
OPENMP_BUILD = ${CXX} ${INPUT} ${OPENMP_FLAG} ${CFLAGS} -o ${BIN_DIR}/${OUTPUT}
OPENCV_BUILD = ${CXX} ${INPUT} ${CFLAGS} -o ${BIN_DIR}/${OUTPUT} ${OPEN_CV_FLAG}

# Commands
clear: 
	rm -rf ${BIN_DIR}/*.o
build-omp: ${INPUT} ${HEADER}
	export OMP_NUM_THREADS=$(t) && ${OPENMP_BUILD}
build-ocv: ${INPUT} ${HEADER}
	@echo Building..
	${OPENCV_BUILD}
	@echo Building complete
run: ${BIN_DIR}/${OUTPUT}
	${BIN_DIR}/${OUTPUT} $(s)


