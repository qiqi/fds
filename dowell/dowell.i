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
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* J, int n_J)}

%inline %{
    void
    c_run_primal(double * u, int n, double * s, int n_s, double * J, int n_J, int n_steps, int n_grid, double dt)
    {
        assert (n == n_grid);
        assert (n_s == 5);
        assert (n_J == n_steps);
        run_primal(u, s, J, n_steps, n_grid, dt); 
    }
%}

%immutable;
int N_GRID;
double DT_STEP;
double T_TOTAL;
%mutable;
