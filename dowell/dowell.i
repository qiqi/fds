%module dowell
%{
#define SWIG_FILE_WITH_INIT
#include "dowell.h"
%}


%include "numpy.i"

%init %{
import_array();
%}


%apply (double* INPLACE_ARRAY1, int DIM1) {(double* u, int n)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* s, int n_s)}

%inline %{
    void
    c_run_primal(double * u, int n, double * s, int n_s, int n_steps, int n_grid, double dt)
    {
        assert (n == n_grid);
        assert (n_s == 5);
        run_primal(u, s, n_steps, n_grid, dt); 
    }
%}

%immutable;
int N_GRID;
double DT_STEP;
double T_TOTAL;
%mutable;
