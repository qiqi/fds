
OBJECTS1 = Lorenz96.o ensemble_tangent.o 
OBJECTS3 = Lorenz96.o longtime_average.o 
MODULES = Lorenz96.mod
LFLAGS = -lblas -llapack
FFLAGS = -g #-fcheck=all -Wall
NUM_PROCS ?= 4   
TEST_ARG = 1
FC = mpif90

.PHONY: test1 enstan lta clean 
enstan: enstan.exe
	mpiexec -n $(NUM_PROCS) ./enstan.exe $(TEST_ARG)

lta: lta.exe
	mpiexec -n $(NUM_PROCS) ./lta.exe

enstan.exe: $(MODULES) $(OBJECTS1)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS1) -o enstan.exe

lta.exe: $(MODULES) $(OBJECTS3)
	$(FC) $(FFLAGS) $(LFLAGS) $(OBJECTS3) -o lta.exe

%.o : %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c  $< 

%.mod: %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c $<

clean:
	rm -f *.o *.exe *.mod

