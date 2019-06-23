Follow below steps to compute sensitivities and draw shadowing directions related figures.
Readers should adjust this instruction to their own situations.
For computing and plotting Lyapunov exponents and vectors, please check the charles_cylinder3D_Lyapunov folder.


1.
In charles.cpp, add to temporalHook() an output function,
which print the objective function to a file called "cylinderforce.txt" at every step.
Later we use this file in fds.
Then compile the charles code from Cascade Technology.


2.
Use charles.exe to run several simulations with different parameters to get corresponding objectives.


3.
Add charles.exe built from step 1 to the ref folder.
Run fluid simulation by charles for some time, add the les file of the last step to the ref folder:
this will be the initial.les file used by fds code.
Move the ref folder to the REF_WORK_PATH given in charles.py


4.
Move every file in the main folder to the fds/apps folder.
The parallel_job.sh files are subscripts for submitting jobs to Slurm queue,
should change to the particular job submission method on your own computer.


5.
Run main/charles.py, this should generate a folder with many checkpoint files of fds.


6.
Run draw_shadowing_djds/plot_djds_hist.py, this requires only the last checkpoint file generated in step 5.
This should generate the convergence history of confidence interval of sensitivities.
The final sensitivities should be printed to screen.


7.
With sensitivities computed from previous step and averaged objectives computed in step 2,
we can use the plot_J_s file to plot objective vs parameter figure, with sensitivities on it.


8.
Run draw_shadowing_djds/drawshadowing.py, this should generate:
1) a folder with many .vtu files containing flow fields of shadowing directions, v^\perp.
2) a norms.p file which contains the history of the norm of shadowing direction, which we can use drawnorm.py to plot.


9. 
Install paraview.
Move for_paraview/paraviewVperp.py and averageVperp.py to the folder of .vtu files generated in the above step.
Use paraviewVperp.py to plot .png files of v^perp at the end of each time segment;
further use the make_avi script to assemble .png files to videos.
Use averageVperp.py to plot .png files of v^perp averaged over time.
