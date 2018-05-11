TESTS = tests/solvers/vanderpol/solver tests/solvers/lorenz/solver tests/solvers/circular/solver
BINARIES = tests/solvers/mock_fun3d/final.data.0 $(TESTS)

default:	$(BINARIES)

tests:	$(TESTS)

tests/solvers/lorenz/solver:
	cd tests/solvers/lorenz; make

tests/solvers/circular/solver:
	cd tests/solvers/circular; make

tests/solvers/vanderpol/solver:
	cd tests/solvers/vanderpol; make

tests/solvers/mock_fun3d/final.data.0:
	cd tests/solvers/mock_fun3d; make

# tools/openfoam4/pisoFoam/pisoFoam:	tools/openfoam4/pisoFoam/*.C tools/openfoam4/pisoFoam/*.H
# 	cd tools/openfoam4/pisoFoam/; wmake

clean:
	rm -f $(BINARIES)
