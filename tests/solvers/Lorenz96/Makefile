# $UWHPSC/codes/fortran/newton/Makefile

OBJECTS = Lorenz96.o test1.o
MODULES = Lorenz96.mod
LFLAGS = -lblas -llapack
FFLAGS = -g

.PHONY: test1 clean 

test1: test1.exe
	./test1.exe

test1.exe: $(MODULES) $(OBJECTS)
	gfortran $(FFLAGS) $(LFLAGS) $(OBJECTS) -o test1.exe

%.o : %.f90
	gfortran $(FFLAGS) $(LFLAGS) -c  $< 

%.mod: %.f90
	gfortran $(FFLAGS) $(LFLAGS) -c $<

clean:
	rm -f *.o *.exe *.mod

