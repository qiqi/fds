#include<assert.h>
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"cblas.h"

#include"dowell.h"

int N_GRID=0;
double S_CONST[5]={ 0.0 };
double DT_STEP=0;
double JBAR[5]={ 0.0 };
double T_TOTAL=0;

// Runge Kutta coefficients
double RK[3][3] = {{ 1./2,  0,    0 },
                   {-1./6,  1./3, 0 },
                   { 0,    -2./3, 1.}};

// PI
double PI = 3.141592653589793;
double PI2 = 9.869604401089358;
double PI4 = 97.40909103400242;


void
ddt(const double * u, double * dudt)
{
    // A1 = u[0], A2 = u[1], A3 = u[2], A4 = u[3],
    // A1' = u[4], A2' = u[5], A3' = u[6], A4' = u[7]
    // Rx = s[0], Lambda = s[1], nu = s[2], (mu/M)**0.5 = s[3], P = s[4]
    double a = - 6.0 * (1 - pow(S_CONST[2], 2)) * PI2;
    double b = - 2.0 * pow(S_CONST[1],0.5) * S_CONST[3];
    double sigma = 0.5 * PI2 * (u[0] * u[0] + 4.0 * u[1] * u[1] 
                                + 9.0 * u[2] * u[2] + 16.0 * u[3] * u[3]);
    
    dudt[0] = u[4]; 
    dudt[1] = u[5];
    dudt[2] = u[6];
    dudt[3] = u[7];
    dudt[4] = (a * sigma - S_CONST[0] * PI2 - PI4) * u[0]
            - S_CONST[1] * (-8./3. * u[1] - 16./15. * u[3])
            + b * u[4]
            + 4. * S_CONST[4] / PI;

    dudt[5] = (a * 4. * sigma - S_CONST[0] * 4. * PI2 - 16. * PI4) * u[1]
            - S_CONST[1] * (8./3. * u[0] - 24./5. * u[2])
            + b * u[5];

    dudt[6] = (a * 9. * sigma - S_CONST[0] * 9. * PI2 - 81. * PI4) * u[2]
            - S_CONST[1] * (24./5. * u[1] - 48./7. * u[3])
            + b * u[6]
            + 4. * S_CONST[4] / (3. * PI);

    dudt[7] = (a * 16. * sigma - S_CONST[0] * 16. * PI2 - 256. * PI4) * u[3]
            - S_CONST[1] * (16./15. * u[0] + 48./7. * u[2])
            + b * u[7];
}


// Objective function J 
double
Obj(const double * u, int obj_ind)
{
    double J = 0;
    double W = 0.0;
    if(obj_ind == 1)
    // time averaged W(0.75)^2
    {
        for (int i = 0; i < 4; ++i) {
            W = W + u[i] * sin((i+1.) * PI * 0.75);
        }
        J = pow(W, 2);

    }
    else if(obj_ind == 2)
    // time averaged 1st mode
    {
        J = u[0];
    }
    else if(obj_ind == 3)
    // time averaged 1st mode squared
    {
        J = pow(u[0],2);
    }
    else if(obj_ind == 4)
    // time averaged variance of W(0.75) 
    {
        for (int i = 0; i < 4; ++i) {
            W = W + u[i] * sin((i+1.) * PI * 0.75);
        }
        J = pow(W-JBAR[0], 2);
    }
    else
    {
    // time averaged W(0.75)
    	for (int i = 0; i < 4; ++i) {
            J = J + u[i] * sin((i+1.) * PI * 0.75);
        }
    }

	return J;
}


// dual consistent explicit RK (See Shan Yang's thesis)
void
stepPrimal(const double * u0, double * u, double dt)
{
    double dudt0[N_GRID], dudt1[N_GRID], *dudt2 = dudt0;

    memmove(u, u0, sizeof(double) * N_GRID);

    ddt(u, dudt0);
    cblas_daxpy(N_GRID, dt * RK[0][0], dudt0, 1, u, 1);

    ddt(u, dudt1);
    cblas_daxpy(N_GRID, dt * RK[1][0], dudt0, 1, u, 1);
    cblas_daxpy(N_GRID, dt * RK[1][1], dudt1, 1, u, 1);

    ddt(u, dudt2);
    cblas_daxpy(N_GRID, dt * RK[2][1], dudt1, 1, u, 1);
    cblas_daxpy(N_GRID, dt * RK[2][2], dudt2, 1, u, 1);
}


// This function initializes this module, must be called from Python
// before using any other functionality.
void
run_primal(double * u, double * s, double * J, int n_steps, int n_grid, double dt)
{
    S_CONST[0] = s[0];
    S_CONST[1] = s[1];
    S_CONST[2] = s[2];
    S_CONST[3] = s[3];
    S_CONST[4] = s[4];


    assert(n_grid > 0);
    N_GRID = n_grid;

    assert(n_steps > 0);

    for (int i = 0; i < n_steps; ++ i)
    {
        J[i] = Obj(u,4);
        stepPrimal(u, u, dt);
    
    }
}

