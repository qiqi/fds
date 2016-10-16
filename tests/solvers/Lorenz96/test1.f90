! Lorenz'96

program test1

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,v, Xnp1_res
	real(kind=8), dimension(:), allocatable :: dXdt
	real(kind=8), dimension(:,:), allocatable :: dfdX_res, vnp1_res
	integer :: i, me, ierr, nprocs, Dproc, D
    integer :: istart, iend, ncyc, lproc, rproc	
	integer, allocatable :: seed(:)
	integer :: rsize, req1, req2
	integer, dimension(MPI_STATUS_SIZE) :: mpistatus
 	real(kind=8), pointer, dimension(:) :: p	
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

	ncyc = 1
	D = 8
	Dproc =	D/nprocs	

	istart = me*Dproc + 1
	iend = min((me + 1)*Dproc, D)
	lproc = MOD(me + nprocs - 1, nprocs)
	rproc = MOD(me + 1, nprocs)

	allocate(X(istart-2:iend+1),v(istart-2:iend+1),vnp1_res(istart:iend,1),Xnp1_res(istart:iend))

	call RANDOM_SEED(SIZE=rsize)
	allocate(seed(rsize))
	call RANDOM_SEED(PUT=seed)	
	call RANDOM_NUMBER(X)
	v = 0.d0
	if(me == 0) then 
		v(istart) = 1.d0
	end if
	print *, me, X

	do i = 1, ncyc	

		call mpi_isend(X(istart), 1,  &
		MPI_DOUBLE_PRECISION,         &
		lproc,  &
		1, MPI_COMM_WORLD, req1, ierr)
			
		call mpi_isend(X(iend-1:iend), &
		2, MPI_DOUBLE_PRECISION,       &
		rproc,   &
		2, MPI_COMM_WORLD, req2, ierr)


		call mpi_recv(X(istart-2:istart-1), &
		2, MPI_DOUBLE_PRECISION, lproc, &
	    2, MPI_COMM_WORLD, mpistatus, ierr)			 			
		
		call mpi_recv(X(iend+1), &
		1, MPI_DOUBLE_PRECISION, rproc, &
	    1, MPI_COMM_WORLD, mpistatus, ierr)			 			

		call mpi_isend(v(istart), 1,  &
		MPI_DOUBLE_PRECISION,         &
		lproc,  &
		3, MPI_COMM_WORLD, req1, ierr)
			
		call mpi_isend(v(iend-1:iend), &
		2, MPI_DOUBLE_PRECISION,       &
		rproc,   &
		4, MPI_COMM_WORLD, req2, ierr)


		call mpi_recv(X(istart-2:istart-1), &
		2, MPI_DOUBLE_PRECISION, lproc, &
	    4, MPI_COMM_WORLD, mpistatus, ierr)			 			
		
		call mpi_recv(X(iend+1), &
		1, MPI_DOUBLE_PRECISION, rproc, &
	    3, MPI_COMM_WORLD, mpistatus, ierr)		


		
		call Xnp1(X,Dproc+3,Xnp1_res)
		call rk45(X,Dproc+3,v,vnp1_res)

		if(me == 0) then
		!Compute lift and drag.	
		end if
		X(istart:iend) = Xnp1_res
		
		v(istart:iend) = vnp1_res(:,1)
							
		
		
	enddo
	call mpi_finalize(ierr)	
	
end program test1
