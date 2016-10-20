! Lorenz'96

program ensemble_tangent

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,v,Xp,Xpnp1_res,Xnp1_res
	real(kind=8), dimension(:), allocatable :: dXdt,g
	real(kind=8), dimension(:,:), allocatable :: dfdX_res, vnp1_res
	integer :: i, me, ierr, nprocs, Dproc, D, ns, ns_proc, j, Dext
    integer :: istart, iend, ncyc, lproc, rproc	
	integer, allocatable :: seed(:)
	integer :: rsize, req1, req2
	integer, dimension(MPI_STATUS_SIZE) :: mpistatus
 	real(kind=8), pointer, dimension(:) :: p
	real(kind=8) :: dt, dXavgds, dXavgds_avg_proc, dXavgds_avg, L1	
	integer :: thefile, T
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

	ncyc = 60000
	D = 40	
	dt = 0.01d0
	T = 720000
	L1 = 0.d0
	ns = T/ncyc
	ns_proc = ns/nprocs
	Dext = D+3
	allocate(X(1:Dext),v(1:Dext),vnp1_res(1:D,1),Xnp1_res(1:D), &
	Xpnp1_res(1:D), Xp(1:Dext), g(1:D))

	istart = 3
	iend = Dext - 1
	g = 0.d0
	g(D) = 1.d0
	dXavgds_avg = 0.d0	
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
			
			do i = 1, ncyc	
				


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
			dXavgds = dXavgds/ncyc
			dXavgds_avg_proc = dXavgds_avg_proc + dXavgds
			deallocate(seed)
	enddo
	
	call MPI_REDUCE(dXavgds_avg_proc, dXavgds_avg, &
					1, MPI_DOUBLE_PRECISION,MPI_SUM,0, &
					MPI_COMM_WORLD,ierr)

	if(me==0) then
		dXavgds_avg = dXavgds_avg/ns
		print *,  dXavgds_avg
	end if


	call mpi_finalize(ierr)	
	
end program ensemble_tangent
