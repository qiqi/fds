Follow below steps to compute sensitivities and draw shadowing directions related figures.
Readers should adjust this instruction to their own situations.


1.
In charles.cpp, add to temporalHook() an output function,
which print the objective function to a file called "cylinderforce.txt" at every step.
Later we use this file in fds.
Then compile the charles code from Cascade Technology.


2.
Use charles to run several simulations with different parameters to get corresponding objectives.


3.
Add charles.exe built from step 1 to the ref folder.
Run fluid simulation by charles for some time, add the les file of the last step to the ref folder:
this will be the initial.les file used by fds code.
Move the ref folder to the REF_WORK_PATH given in charles.py


4.
Move every file in the main folder to the fds/apps folder.
Note: the parallel_job.sh files are subscripts for submitting jobs to Slurm queue.
Should change to the particular job submission method on your own computer.


5.
Run charles.py in the main folder, this should generate 
1) a folder with many checkpoint files of fds.
2) screen output of the sensitivities computed by FD-NILSS after each time segment


6.
Run drawshadowing.py in the draw_shadowing_direction folder.
This should generate:
1) a folder with many .vtu files containing flow fields of shadowing directions, v^\perp.
2) a norms.p file which contains the norm of shadowing direction, which we can use drawnorm.py to plot.
Moreover, with sensitivities computed from previous step and averaged objectives computed in step 2,
we can use the plot.py file to plot objective vs parameter figure, with sensitivities on it.


7. 
Install paraview.
Move  paraviewVperp.py and averageVperp.py to the folder of .vtu files.
Use paraviewVperp.py to plot .png files of v^perp at the end of each time segment.
Use averageVperp.py to plot .png files of v^perp averaged over time.
