solver:	circular.f03 ../common/finite-diff.f03 ../common/solver.f03
	gfortran -c circular.f03
	gfortran ../common/solver.f03 circular.o -o solver
	gfortran ../common/finite-diff.f03 circular.o -o finite-diff
	echo '1 1' > u0
	./finite-diff > fd.txt
