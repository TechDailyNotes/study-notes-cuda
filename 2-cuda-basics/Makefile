.PHONY: indexing_exec clean

indexing_exec:
	@nvcc indexing.cu -o indexing.out
	@./indexing.out

clean:
	rm -f *.out
