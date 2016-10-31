
OBJECTS1 = Lorenz96.o ensemble_tangent.o 
OBJECTS2 = Lorenz96.o test1.o 
OBJECTS3 = Lorenz96.o longtime_average.o 
MODULES = Lorenz96.mod
LFLAGS = -lblas -llapack
FFLAGS = -g #-fcheck=all -Wall
NUM_PROCS ?= 4   

FC = mpif90

.PHONY: test1 enstan lta clean 
enstan: enstan.exe
	mpiexec -n $(NUM_PROCS) ./enstan.exe

lta: lta.exe
	mpiexec -n $(NUM_PROCS) ./lta.exe

enstan.exe: $(MODULES) $(OBJECTS1)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS1) -o enstan.exe

lta.exe: $(MODULES) $(OBJECTS3)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS3) -o lta.exe

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

