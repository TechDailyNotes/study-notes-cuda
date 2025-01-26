.PHONY: vector_add clean

vector_add:
	nvcc -arch=sm_86 cuda_vector_addition_1.cu -o vector_add.out

clean:
	rm -f *.out
