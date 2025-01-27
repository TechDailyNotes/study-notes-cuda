.PHONY: vector_add_1 vector_add_2 matmul_core matmul_full clean

vector_add_1:
	nvcc -arch=sm_86 cuda_vector_addition_1.cu -o vector_add_1.out

vector_add_2:
	nvcc -arch=sm_86 cuda_vector_addition_2.cu -o vector_add_2.out

matmul_core:
	nvcc -arch=sm_86 cuda_matmul_core.cu -o matmul_core.out

matmul_full:
	nvcc -arch=sm_86 cuda_matmul_full.cu -o matmul_full.out

clean:
	rm -f *.out
