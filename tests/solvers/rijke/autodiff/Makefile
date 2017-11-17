ifndef F90C
F90C=gfortran
endif
ifndef CC
CC=gcc
endif
RTSUPP_rev=w2f__types OAD_active_custom OAD_cp OAD_tape OAD_rev   
#RTSUPP=w2f__types OAD_active OAD_cp OAD_tape OAD_rev   
RTSUPP_for=w2f__types OAD_active_custom  


#Interface with nilss

tangent: $(addsuffix .o, $(RTSUPP_for)) \ \
rijke_oad_passive.o numCorei.pre.xb.x2w.w2f.post.o driver_iht_nilss.o
	${F90C} -o $@ $^

adjoint: $(addsuffix .o, $(RTSUPP_rev)) \ \
rijke_oad_passive.o numCorea.pre.xb.x2w.w2f.post.o driver_adjoint_nilss.o
	${F90C} -o $@ $^



solver_primal: \
rijke_oad_passive.o driver_primal.o 
	${F90C} -o $@ $^ 




numCorei.pre.xb.x2w.w2f.post.f90 $(addsuffix .f90,  $(RTSUPP_for)) iaddr.c : toolChaini 
	
numCorea.pre.xb.x2w.w2f.post.f90 $(addsuffix .f90,  $(RTSUPP_rev)) iaddr.c : toolChaina 
	



toolChaini : numCorei.f90
	openad -c -m f $<
	rsync -av ${OPENADROOT}/runTimeSupport/vector/OAD_active.f90 ./
	cat OAD_active.f90 | sed 's/100/33/' > OAD_active_custom.f90	

numCorei.f90: rijke_oad.f90 head.prepped.inhomo.f90
	cat $^ > $@


toolChaina : numCorea.f90
	openad -c -m rj $<
	rsync -av ${OPENADROOT}/runTimeSupport/vector/OAD_active.f90 ./
	cat OAD_active.f90 | sed 's/100/33/' > OAD_active_custom.f90	

numCorea.f90: rijke_oad.f90 head.prepped.adjoint.f90
	cat $^ > $@



rijke_oad_passive.f90: rijke_oad.f90
	cat $^ | sed 's/rijke/rijke_passive/' > $@
  

ad_inline.f:toolChain




%.o : %.f90
	${F90C} -o $@ -c $< 

%.o : %.f
	${F90C} ${F90FLAGS} -g -O -o $@ -c $< 

%.o : %.c
	${CC} -o $@ -c $< 


clean: 
	rm -f ad_template* ad_inline.f OAD_* w2f__*  iaddr* 
	rm -f head.prepped.pre.* *.B *.xaif *.o *.mod driver driverE *~
	rm -f numCore* *.mod*
	rm -f rijke_passive.* 
.PHONY: clean toolChain
