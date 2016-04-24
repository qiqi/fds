lorenz:
	gfortran -c lorenz.f03
	gfortran solver.f03 lorenz.o -o solver
	gfortran finite-diff.f03 lorenz.o -o finite-diff
	echo 3 > n
	./finite-diff > fd.txt

circular:
	gfortran -c circular.f03
	gfortran solver.f03 circular.o -o solver
	gfortran finite-diff.f03 circular.o -o finite-diff
	echo 3 > n
	./finite-diff > fd.txt
