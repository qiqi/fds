#ifndef PDE_H
#define PDE_H

#include<assert.h>

void
du_dx(const double * u, double * dudt);

void
du2_dx(const double * u, double * dudt);

void
d2u_dx2(const double * u, double * dudt);

void
d4u_dx4(const double * u, double * dudt);

void
ddt(const double * u, double * dudt);

double
Obj(const double * u, int obj_ind);

void
stepPrimal(const double * u0, double * u, double dt);

void
run_primal(double * u, double * s, int n_steps, int n_grid, double dt);

extern int N_GRID;
extern double S_CONST[5];
extern double DT_STEP;
extern double JBAR[5];
extern double T_TOTAL;
#endif
