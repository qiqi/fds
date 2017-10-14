solver:	lorenz.f03 ../common/finite-diff.f03 ../common/tan-adj.f03 ../common/solver.f03 ../common/adjoint.f03
	gfortran -c lorenz.f03
	gfortran ../common/solver.f03 lorenz.o -o solver
	gfortran ../common/adjoint.f03 lorenz.o -o adjoint
	gfortran ../common/finite-diff.f03 lorenz.o -o finite-diff
	gfortran ../common/tan-adj.f03 lorenz.o -o tan-adj
	echo '1 1 28' > u0
	./finite-diff > tan-fd.txt
	./tan-adj > tan-adj.txt
