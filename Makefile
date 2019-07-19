a.out: main.c
	gcc -mavx512f main.c

.PHONY: run
run: a.out
	aprun -q ./a.out

main.s:
	gcc -S -mavx512f main.c
