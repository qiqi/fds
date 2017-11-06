ifndef F90C
F90C=gfortran
endif
ifndef CC
CC=gcc
endif
RTSUPP=w2f__types OAD_active_custom OAD_cp OAD_tape OAD_rev   
#RTSUPP=w2f__types OAD_active OAD_cp OAD_tape OAD_rev   
#RTSUPP=w2f__types OAD_active  


#Interface with nilss
solver_ht: $(addsuffix .o, $(RTSUPP)) \
combustion_juniper_passive.o numCoreh.pre.xb.x2w.w2f.post.o driver_ht_nilss.o
	${F90C} -o $@ $^

solver_iht: $(addsuffix .o, $(RTSUPP)) \ \
combustion_juniper_passive.o numCorei.pre.xb.x2w.w2f.post.o driver_iht_nilss.o
	${F90C} -o $@ $^


solver_primal: \
combustion_juniper_passive.o driver_primal.o 
	${F90C} -o $@ $^ 

solver_attractor: \
combustion_juniper_passive.o driver_attractor.o 
	${F90C} -o $@ $^ 

solver_phaseportrait: \
combustion_juniper_passive.o phase_portrait.o 
	${F90C} -o $@ $^ 

test_primal: \
rijke.o test_primal.o 
	${F90C} -o $@ $^ 

test_adjoint: \
../common/tan-adj.f03 rijke.f90
	${F90C} -o $@ $^ 

numCoreh.pre.xb.x2w.w2f.post.f90 $(addsuffix .f90, $(RTSUPP)) iaddr.c : toolChainh 

numCorei.pre.xb.x2w.w2f.post.f90 $(addsuffix .f90,  $(RTSUPP)) iaddr.c : toolChaini 
	


toolChainh : numCoreh.f90
	openad -c -m rj $<
	rsync -av ${OPENADROOT}/runTimeSupport/vector/OAD_active.f90 ./
	cat OAD_active.f90 | sed 's/100/33/' > OAD_active_custom.f90		

numCoreh.f90: combustion_juniper.f90 head.prepped.homo.f90
	cat $^ > $@

toolChaini : numCorei.f90
	openad -c -m rj $<
	rsync -av ${OPENADROOT}/runTimeSupport/vector/OAD_active.f90 ./
	cat OAD_active.f90 | sed 's/100/33/' > OAD_active_custom.f90	

numCorei.f90: combustion_juniper.f90 head.prepped.inhomo.f90
	cat $^ > $@


combustion_juniper_passive.f90: combustion_juniper.f90
	cat $^ | sed 's/combustion_juniper/combustion_juniper_passive/' > $@
  

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
	rm -f combustion_juniper_passive.* 
.PHONY: clean toolChain
# the following include has explicit rules that could replace the openad script
include MakeExplRules.inc
