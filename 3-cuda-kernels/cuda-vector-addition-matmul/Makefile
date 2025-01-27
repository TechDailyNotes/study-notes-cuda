.PHONY: vector_add_1 vector_add_2 clean

vector_add_1:
	nvcc -arch=sm_86 cuda_vector_addition_1.cu -o vector_add_1.out

vector_add_2:
	nvcc -arch=sm_86 cuda_vector_addition_2.cu -o vector_add_2.out

clean:
	rm -f *.out
