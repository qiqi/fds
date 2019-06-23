Follow below steps for computing and plotting Lyapunov exponents and vectors of the 3D cylinder testcase simulated by charles.
Readers should adjust this instruction to their own situations.
For computing and plotting sensitivities and shadowing directions, please check the charles_cylinder3D folder.


1.
In charles.cpp, add to temporalHook() an output function,
which print the objective function to a file called "cylinderforce.txt" at every step.
Later we use this file in fds.
Then compile the charles code from Cascade Technology.


2.
Add charles.exe built from step 1 to the ref folder.
Run fluid simulation by charles for some time, add the les file of the last step to the ref folder:
this will be the initial.les file used by fds code.
Move the ref folder to the REF_WORK_PATH given in charles.py
Note: theoretically, we no longer need a charles.template file for Lyapunov analysis,
but we still have it here so that we don't need to modify the code too much.


3.
Replace fds/fds.py by apps/charles_cylinder3D_Lyapunov/fds/fds.py.
The main change is to comment out the 4 lines where we previously called the function time_dil.project(),
which is a function for computing the ^perp, the orthogonal projection of a vector to the perpendicular subspace of trajectory.
For NILSS we need this ^perp, but for Lyapunov analysis we do not.


4.
Move every file in the main folder to the fds/apps folder.
The parallel_job.sh files are subscripts for submitting jobs to Slurm queue,
should change to the particular job submission method on your own computer.


5.
Run main/charles.py, this should generate a folder with many checkpoint files of fds.


6.
Run draw_Lyapunov/drawLE.py, this requires only the last checkpoint file generated in step 5.
This should compute and plot confidence intervals of all LEs and their converging history
The final Lyapunov exponents should be printed to screen and save to a LE.p file.


7.
Run draw_Lyapunov/draw_angles.py, this requires only the last checkpoint file generated in step 5.
This should compute and plot angles between CLVs.


8.
Run draw_Lyapunov/drawCLV.py, this requires all checkpoint files generated in step 5.
This should generate a folder with many .vtu files containing flow fields of CLVs.


9. 
Install paraview.
Move for_paraview/paraviewCLV.py and averageCLV.py to the folder of .vtu files generated in the above step.
Use paraviewCLV.py to plot .png files of CLVs at the end of each time segment;
further use the make_avi script to assemble .png files to videos.
Use averageCLV.py to plot .png files of CLVs averaged over time.
