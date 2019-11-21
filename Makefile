bloom: bloom.c
	gcc -Wall -g -o bloom bloom.c

clean:  
	@rm -rf *.o *~ *.dSYM bloom
	@echo Made clean.
