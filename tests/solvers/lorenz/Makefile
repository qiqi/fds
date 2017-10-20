solver:	lorenz.f03 ../common/finite-diff.f03 ../common/fd-tan.f03 ../common/tan-adj.f03 ../common/solver.f03 ../common/tangent.f03 ../common/adjoint.f03
	gfortran -c lorenz.f03
	gfortran ../common/solver.f03 lorenz.o -o solver
	gfortran ../common/tangent.f03 lorenz.o -o tangent
	gfortran ../common/adjoint.f03 lorenz.o -o adjoint
	gfortran ../common/finite-diff.f03 lorenz.o -o finite-diff
	gfortran ../common/fd-tan.f03 lorenz.o -o fd-tan
	gfortran ../common/tan-adj.f03 lorenz.o -o tan-adj
	echo '1 1 28' > u0
	./finite-diff > fd.txt
	./fd-tan > fd-tan.txt
	./tan-adj > tan-adj.txt
