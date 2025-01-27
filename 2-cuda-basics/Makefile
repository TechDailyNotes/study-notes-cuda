.PHONY: indexing_exec memory_exec_clean clean

indexing_exec:
	@nvcc indexing.cu -o indexing.out
	@./indexing.out

memory_exec_clean:
	@nvcc -arch=sm_86 memory.cu -o memory.out
	@./memory.out
	@rm -f ./memory.out

clean:
	rm -f *.out
