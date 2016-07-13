#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include"cblas.h"
#include<math.h>
#include"dowell.h"


int
main()
{
    // Rx, Lam, nu, muOverM^0.5, P
    double s[5] = {-5.3 * 9.869604401089358,150.0,0.3,0.1,0.0}; 
	double dt = 0.001;
    const int n_grid = 8; 
    double  u0[n_grid];
    //for (int i = 0; i < n_grid; ++ i) u0[i] = rand() / (double)RAND_MAX;
    for (int i = 0; i < n_grid; ++ i) u0[i] = 0.0;
    u0[0] = 0.01;

    int n_steps = 10000;
    run_primal(u0, s, n_steps, n_grid, dt);

	
    // Print final solution to file
    FILE *fp;
    fp = fopen("u_fin.dat", "w");
    for(int j = 0; j < N_GRID; j++) {
        fprintf(fp,"%d %f \n",j, u0[j]);
    }
	fclose(fp);

    return 0;
}
