a.out: main.c
	gcc -mavx512f main.c

.PHONY: run
run: a.out
	aprun -q ./a.out
