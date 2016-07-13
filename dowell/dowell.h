#ifndef PDE_H
#define PDE_H

#include<assert.h>

void
ddt(const double * u, double * dudt);

double
Obj(const double * u, int obj_ind);

void
stepPrimal(const double * u0, double * u, double dt);

void
run_primal(double * u, double * s, double * J, int n_steps, int n_grid, double dt);

extern int N_GRID;
extern double S_CONST[5];
extern double DT_STEP;
extern double JBAR[5];
extern double T_TOTAL;
#endif
