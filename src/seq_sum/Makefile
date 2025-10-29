# Simple Makefile for tutorial04

NVCC = /usr/local/cuda-12.2/bin/nvcc
CXXFLAGS = -O2

TARGET = vector_sum
SRC = vector_sum.cu block_reduce.cu cpu_sum.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
