
OBJECTS = Lorenz96.o test1.o
MODULES = Lorenz96.mod
LFLAGS = -lblas -llapack
FFLAGS = -g
NUM_PROCS ?= 4   

FC = mpif90

.PHONY: test1 clean 

test1: test1.exe
	mpiexec -n $(NUM_PROCS) ./test1.exe

test1.exe: $(MODULES) $(OBJECTS)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS) -o test1.exe

%.o : %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c  $< 

%.mod: %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c $<

clean:
	rm -f *.o *.exe *.mod

