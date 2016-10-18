
OBJECTS1 = Lorenz96.o ensemble_tangent.o 
OBJECTS2 = Lorenz96.o test1.o 
MODULES = Lorenz96.mod
LFLAGS = -lblas -llapack
FFLAGS = -g
NUM_PROCS ?= 4   

FC = mpif90

.PHONY: test1 enstan clean 
enstan: enstan.exe
	mpiexec -n $(NUM_PROCS) ./enstan.exe

enstan.exe: $(MODULES) $(OBJECTS1)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS1) -o enstan.exe

test1: test1.exe
	mpiexec -n $(NUM_PROCS) ./test1.exe

test1.exe: $(MODULES) $(OBJECTS2)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS2) -o test1.exe

%.o : %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c  $< 

%.mod: %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c $<

clean:
	rm -f *.o *.exe *.mod

