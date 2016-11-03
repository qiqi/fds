! Lorenz'96

program ensemble_tangent

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,Xnp1_res,F,Xavg	
	integer :: i, me, ierr, nprocs, D, j, Dext !, you_old, you_new
    integer :: istart, iend, max_iter	
	integer, allocatable :: seed(:)
	integer :: rsize, nF, nF_sent, sender, jj
	real(kind=8) :: Xavg_F, dF,Fproc
	integer, dimension(MPI_STATUS_SIZE) :: status
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

		
	D = 40	
	Dext = D+3
	dF = 0.1
	nF = 100	
	
	allocate(X(1:Dext), Xnp1_res(1:D), &
	F(1:nF))
	istart = 3
	iend = Dext - 1
	
	if(me==0) then	
			!open(unit=20, file='EthetaEA_test.dat')
			!open(unit=21, file='VthetaEA_test.dat')
	end if

	!Direct Integration
	Xavg = 0.d0	
	if(me==0) then

		nF_sent = 0

		!Initialize F
		F(1) = 5.d0
		do j = 2, nF

			F(j) = F(j-1) + dF
		end do

		do j=1,min(nprocs-1,nF)

			call MPI_SEND(F(j), 1, MPI_DOUBLE_PRECISION, &
							j, j, MPI_COMM_WORLD, ierr)
			
			nF_sent = nF_sent + 1	
				
		end do
	
		do j = 1,nF
			
			call MPI_RECV(Xavg_F, 1, MPI_DOUBLE_PRECISION, &
				MPI_ANY_SOURCE, MPI_ANY_TAG, &
				MPI_COMM_WORLD, status, ierr)
		
			sender = status(MPI_SOURCE)
			jj = status(MPI_TAG)
			Xavg(jj) = Xavg_F

			if(nF_sent < nF) then
		
				nF_sent = nF_sent + 1
				call MPI_SEND(F(nF_sent), 1, MPI_DOUBLE_PRECISION, &
					sender, nF_sent, MPI_COMM_WORLD, ierr)
							
			else
			
				call MPI_SEND(F(nF_sent), 1, MPI_DOUBLE_PRECISION, &
					sender, 0, MPI_COMM_WORLD, ierr)

				

			end if	
			 	

		end do	
		
		
	end if 

		
!Code for workers

	if(me /= 0) then

		if(me > nF) then 
			go to 99
		end if

		do while (.true.)

			call MPI_RECV(Fproc, 1, MPI_DOUBLE_PRECISION, &
						0, MPI_ANY_TAG, MPI_COMM_WORLD, &
						status, ierr)
			j = status(MPI_TAG)
		
			if(j==0) then 
				go to 99
 			end if
		
			call RANDOM_SEED(SIZE=rsize)
			allocate(seed(rsize))
			call RANDOM_SEED(PUT=seed)	
			call RANDOM_NUMBER(X)
			deallocate(seed)	
	
			max_iter = 1000000
			Xavg_F = 0.d0

			do i = 1,max_iter

						

					X(1) = X(Dext-2)
					X(2) = X(Dext-1)
					X(Dext) = X(istart)
	
					call Xnp1(X,Dext,Xnp1_res)
			
			
					if(i > 300000) then
						Xavg = Xavg + SUM(Xnp1_res)/D 
					end if


			end do
			Xavg = Xavg/(max_iter-300000)
			print *, "Long term average is : ", Xavg
			
	end do  


	end if

99 continue
		call mpi_finalize(ierr)	
	
end program ensemble_tangent
