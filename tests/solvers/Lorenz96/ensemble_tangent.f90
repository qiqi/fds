! Lorenz'96

program ensemble_tangent

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,v,Xp,Xpnp1_res,Xnp1_res
	real(kind=8), dimension(:), allocatable :: dXdt,g,thetaEA_mean, thetaEA_var	
	real(kind=8), dimension(:,:), allocatable :: dfdX_res, vnp1_res
	integer :: i, me, ierr, nprocs, Dproc, D, ns, ns_proc, j, Dext !, you_old, you_new
    integer :: istart, iend, lproc, rproc, max_iter	
	integer, allocatable :: seed(:)
	integer :: rsize, req1, req2
	real(kind=8), pointer, dimension(:) :: p
	real(kind=8) :: dt, dXavgds, dXavgds_avg_proc, L1, dXavgds_fd,F,dF
	integer :: thefile, T, tau,k,k1,N, ntau, k2
	real(kind=8), dimension(:), allocatable :: dXavgds_all,thetaEA
	integer, dimension(MPI_STATUS_SIZE) :: status
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

		
	D = 38	
	ntau =2
	dt = 0.01d0
	T = 5000
	
	ns = 100
	ns_proc = ns/nprocs
	Dext = D+3
	dF = 0.01d0	
	F = 4.99	
	allocate(X(1:Dext),v(1:Dext),vnp1_res(1:D,1),Xnp1_res(1:D), &
	Xpnp1_res(1:D), Xp(1:Dext), g(1:D), dXavgds_all(1:ns), &
	thetaEA_mean(1:ntau), thetaEA_var(1:ntau))
	if(me==0) then
		k=0
	end if
	istart = 3
	iend = Dext - 1
	g = 1.d0/D


	if(me==0) then	
			open(unit=20, file='EthetaEA_test.dat')
			open(unit=21, file='VthetaEA_test.dat')
            open(unit=22, file='Dynamics.dat')
	end if

	!Finite Difference
	
	if(me==0) then
		max_iter = 1000000
		dXavgds_fd = 0.d0

		!call RANDOM_SEED(SIZE=rsize)
		!allocate(seed(rsize))
		!call RANDOM_SEED(PUT=seed)	
		call RANDOM_NUMBER(X)
		!deallocate(seed)
		v = 0.d0
		Xp = X
		print *, X
        !Xp(istart) = Xp(istart) + 0.01d0
		do i = 1,max_iter

				

			X(1) = X(Dext-2)
			X(2) = X(Dext-1)
			X(Dext) = X(istart)

			Xp(1) = Xp(Dext-2)
			Xp(2) = Xp(Dext-1)
			Xp(Dext) = Xp(istart)

			call Xnp1(X,Dext,Xnp1_res,F)
			call Xnp1(Xp,Dext,Xpnp1_res,F+dF)
	
			if(i > 400000) then
               
				dXavgds_fd = dXavgds_fd + & 
				(SUM(Xpnp1_res) - SUM(Xnp1_res))/D/dF
			    
                !if(MOD(i,5)==0) then
                        write(22, *), Xpnp1_res(1)
                !end if
            end if
			
			X(3:Dext-1) = Xnp1_res
			Xp(3:Dext-1) = Xpnp1_res
		end do
		dXavgds_fd = dXavgds_fd/(max_iter-400000)/dt
		print *, "fds sens... ", dXavgds_fd
	end if 





	!Ensemble Tangent	
	do k2 = 1,ntau
			
        print *, "tau = ", tau
		tau = 100 + (k2-1)*4930/(ntau-1)	
	
		if(me==0) then
				print *, "Beginning tau = ", tau
				k = 0
		endif	
			N = T/tau
			allocate(thetaEA(1:ns/N))
			dXavgds_avg_proc = 0.d0	
			do j = 1, ns_proc
					
					!print *, "Starting with sample number ", j, "at tau = ", tau
					!print *, "I am processor, ", me	
					!call RANDOM_SEED(SIZE=rsize)
					!allocate(seed(rsize))
					!call RANDOM_SEED(PUT=seed)	
					call RANDOM_NUMBER(X)
					v = 0.d0
					Xp = X

					v(istart) = 1.d0
					Xp(istart) = Xp(istart) + 0.001d0
				

				!	call MPI_FILE_OPEN(MPI_COMM_WORLD, 'test', &
				!			MPI_MODE_WRONLY + MPI_MODE_CREATE, &
				!			MPI_INFO_NULL, thefile, ierr)
				!			open(unit=30+me, file='initialdata.dat')
				!			write(30+me,*) X(istart:iend)	
				!			close(30+me)	

					dXavgds = 0.d0
					
					do i = 1, tau	
						


						v(1) = v(Dext-2)
						v(2) = v(Dext-1)
						v(Dext) = v(istart)


				
						X(1) = X(Dext-2)
						X(2) = X(Dext-1)
						X(Dext) = X(istart)

					
						call Xnp1(X,Dext,Xnp1_res,F)
				
						call rk45_full(X,Dext,v,vnp1_res)

						!if(me == 0) then
						!Compute lift and drag.	
						!end if
						X(istart:iend) = Xnp1_res
						Xp(istart:iend) = Xpnp1_res
						v(istart:iend) = vnp1_res(:,1)
										
					!	if(me==0 .and. i==1000) then	
					!		open(unit=20, file='X.dat')
					!		write(20,*) X(istart:iend)	
					!		open(unit=21, file='Xp.dat')
					!		write(21,*) Xp(istart:iend)		
					!	end if


						dXavgds = dXavgds + DOT_PRODUCT(v(istart:iend),g)
					enddo
					!close(20)
					!close(21)
					dXavgds = dXavgds/tau/dt
					if(me==1) then
						print *, "Tangent sens... ", dXavgds	
					endif 
					call MPI_SEND(dXavgds,1,MPI_DOUBLE_PRECISION, &
									0, me, MPI_COMM_WORLD, ierr)

					if(me==0) then
					!	you_old = 50
						do k1 = 1,nprocs
							k = k + 1
							call MPI_RECV(dXavgds_all(k), &
								1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, &
								MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)
								
!							!you_new = status(MPI_SOURCE)	
!							!if(tau==70 .and. j==1) then
							!	if(you_old==you_new) then
							!		print *, "busted..."
							!	end if
							!	you_old = you_new
							!end if
						enddo
					end if      
					!deallocate(seed)
			enddo

	
			
		if(me==0) then
				do k1 = 1,ns/N
					thetaEA(k1) = sum(dXavgds_all((k1-1)*N+1:k1*N))/N
				enddo
			
			
				thetaEA_mean(k2) = sum(thetaEA)/(ns/N)
				thetaEA_var(k2) = 0.d0
			
				do k1 = 1,ns/N
			
					thetaEA_var(k2) = thetaEA_var(k2) + (thetaEA(k1)-thetaEA_mean(k2))**2.0
				enddo
				print *, thetaEA_var(k2)
				thetaEA_var(k2) = thetaEA_var(k2)/(ns/N)	
		end if
		deallocate(thetaEA)
	
	end do

	if(me==0) then

		do k2=1,ntau
	
			write(20, *) thetaEA_mean(k2)
			write(21, *) thetaEA_var(k2)	

		end do

	close(20)
	close(21)


	end if

		call mpi_finalize(ierr)	
	
end program ensemble_tangent
