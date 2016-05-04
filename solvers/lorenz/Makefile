solver:
	gfortran -c lorenz.f03
	gfortran ../common/solver.f03 lorenz.o -o solver
	gfortran ../common/finite-diff.f03 lorenz.o -o finite-diff
	echo '1 1 28' > u0
	./finite-diff > fd.txt
