ifndef F90C
F90C=gfortran
endif
ifndef CC
CC=gcc
endif



solver: \
rijke.o solver.o
	${F90C} -o $@ $^ 

tangent: \
rijke.o tangent.o 
	${F90C} -o $@ $^ 

adjoint: \
rijke.o adjoint.o 
	${F90C} -o $@ $^ 


test_adjoint: \
rijke.o tan-adj.o 
	${F90C} -o $@ $^ 

write_dcheb: \
write_dcheb.o rijke.o 
	${F90C} -o $@ $^ 


%.o : %.f03
	${F90C} -o $@ -c $< 

%.o : %.f90
	${F90C} -o $@ -c $< 

%.o : %.f
	${F90C} ${F90FLAGS} -g -O -o $@ -c $< 

%.o : %.c
	${CC} -o $@ -c $< 


clean: 
	rm -f *.o *.mod
.PHONY: clean
