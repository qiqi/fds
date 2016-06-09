default:	tests/solvers/lorenz/solver tests/solvers/circular/solver

tests/solvers/lorenz/solver:
	cd tests/solvers/lorenz; make

tests/solvers/circular/solver:
	cd tests/solvers/circular; make
