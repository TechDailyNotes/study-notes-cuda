.PHONY: c c_exec c_obj c_obj_exec cpp cpp_exec cuda cuda_exec java python clean

# Compilers
GCC = gcc
G++ = g++
NVCC = nvcc
NVCC_ARGS = -arch=sm_86

# Objects
C_OUT = c.out
C_OBJ = c.obj
CPP_OUT = cpp.out
CUDA_OUT = cuda.out

c:
	$(GCC) c.c -o $(C_OUT)

c_exec: c
	@./$(C_OUT)

c_obj:
	$(GCC) -c c.c -o $(C_OBJ)

c_obj_exec: c_obj
	@$(GCC) $(C_OBJ) -o $(C_OUT)
	@./$(C_OUT)

cpp:
	$(G++) cpp.cpp -o $(CPP_OUT)

cpp_exec: cpp
	@./$(CPP_OUT)

cuda:
	$(NVCC) $(NVCC_ARGS) cuda.cu -o $(CUDA_OUT)

cuda_exec: cuda
	@./$(CUDA_OUT)

java:
	@javac Java.java
	@java Java

python:
	@python python.py

clean:
	rm -f $(C_OUT) $(C_OBJ) $(CPP_OUT) $(CUDA_OUT) *.class
