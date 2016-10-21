! Lorenz'96

program ensemble_tangent

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,v,Xp,Xpnp1_res,Xnp1_res
	real(kind=8), dimension(:), allocatable :: dXdt,g
	real(kind=8), dimension(:,:), allocatable :: dfdX_res, vnp1_res
	integer :: i, me, ierr, nprocs, Dproc, D, ns, ns_proc, j, Dext
    integer :: istart, iend, lproc, rproc	
	integer, allocatable :: seed(:)
	integer :: rsize, req1, req2
	integer, dimension(MPI_STATUS_SIZE) :: mpistatus
 	real(kind=8), pointer, dimension(:) :: p
	real(kind=8) :: dt, dXavgds, dXavgds_avg_proc, L1, thetaEA_mean, thetaEA_var	
	integer :: thefile, T, tau,k,k1,s
	real(kind=8), dimension(:), allocatable :: dXavgds_all,thetaEA
	integer, dimension(MPI_STATUS_SIZE) :: status
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

	tau = 1000
	D = 40	
	dt = 0.01d0
	T = 100000
	L1 = 0.d0
	ns = 10000
	ns_proc = ns/nprocs
	Dext = D+3
	s = T/tau
	allocate(X(1:Dext),v(1:Dext),vnp1_res(1:D,1),Xnp1_res(1:D), &
	Xpnp1_res(1:D), Xp(1:Dext), g(1:D), dXavgds_all(1:ns),thetaEA(1:ns/s))
	if(me==0) then
		k=0
	endif

	istart = 3
	iend = Dext - 1
	g = 0.d0
	g(D) = 1.d0
	
	dXavgds_avg_proc = 0.d0	
	do j = 1, ns_proc

			
			call RANDOM_SEED(SIZE=rsize)
			allocate(seed(rsize))
			call RANDOM_SEED(PUT=seed)	
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

                Xp(1) = Xp(Dext-2)
                Xp(2) = Xp(Dext-1)
                Xp(Dext) = Xp(istart)

				call Xnp1(X,Dext,Xnp1_res)
				call Xnp1(Xp,Dext,Xpnp1_res)
				call rk45_full(X,Dext,v,vnp1_res)

				if(me == 0) then
				!Compute lift and drag.	
				end if
				X(istart:iend) = Xnp1_res
				Xp(istart:iend) = Xpnp1_res
				v(istart:iend) = vnp1_res(:,1)
								
				if(me==0 .and. i==1000) then	
					open(unit=20, file='X.dat')
					write(20,*) X(istart:iend)	
					open(unit=21, file='Xp.dat')
					write(21,*) Xp(istart:iend)		
				end if


				dXavgds = dXavgds + DOT_PRODUCT(v(istart:iend),g)
			enddo
			close(20)
			close(21)
			dXavgds = dXavgds/tau
			
			call MPI_SEND(dXavgds,1,MPI_DOUBLE_PRECISION, &
							0, me, MPI_COMM_WORLD, ierr)

			if(me==0) then
				do k1 = 1,nprocs
					k = k + 1
					call MPI_RECV(dXavgds_all(k), &
						1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, &
						MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)
				end do
			end if      
			deallocate(seed)
	enddo

	
	
	if(me==0) then
		do k1 = 1,ns/s
			thetaEA(k1) = sum(dXavgds_all((k1-1)*s+1:k1*s))/s
		end do
		thetaEA_mean = sum(thetaEA)/(ns/s)
		thetaEA_var = 0.d0
		do k1 = 1,ns/s
			thetaEA_var = thetaEA_var + (thetaEA(k1)-thetaEA_mean)**2.0
		end do
		thetaEA_var = thetaEA_var/(ns/s)
	end if


	call mpi_finalize(ierr)	
	
end program ensemble_tangent
