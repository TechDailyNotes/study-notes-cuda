.PHONY: test_exec clean

test_exec:
	@nvcc -arch=sm_86 test.cu -o test.out
	@./test.out

clean:
	rm -f *.out
