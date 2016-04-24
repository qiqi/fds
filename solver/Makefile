lorenz:
	gfortran -c lorenz.f03
	gfortran solver.f03 lorenz.o -o solver
	gfortran finite-diff.f03 lorenz.o -o finite-diff
	echo '1 1 28' > u0
	./finite-diff > fd.txt

circular:
	gfortran -c circular.f03
	gfortran solver.f03 circular.o -o solver
	gfortran finite-diff.f03 circular.o -o finite-diff
	echo '1 1 1' > u0
	./finite-diff > fd.txt
