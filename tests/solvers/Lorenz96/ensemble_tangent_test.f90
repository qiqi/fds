! Lorenz'96

program ensemble_tangent

    use Lorenz96
    use mpi 

    implicit none
	real(kind=8), dimension(:), allocatable ::X,v,Xp,Xpnp1_res,Xnp1_res
	real(kind=8), dimension(:), allocatable :: g,thetaEA_mean, thetaEA_var	
	real(kind=8), dimension(:,:), allocatable :: dfdX_res, vnp1_res
	integer :: i, me, ierr, nprocs, Dproc, D, ns, ns_proc, j, Dext !, you_old, you_new
    integer :: istart, iend, lproc, rproc, max_iter, ns_sent	
	integer, allocatable :: seed(:)
	integer :: rsize, req1, req2, sender
	real(kind=8), pointer, dimension(:) :: p
	real(kind=8) :: dt, dXavgds, dXavgds_avg_proc, L1, dXavgds_fd,F,dF
	integer :: thefile, T, tau,k,k1,N, ntau, k2
	real(kind=8), dimension(:), allocatable :: dXavgds_all,thetaEA
	integer, dimension(MPI_STATUS_SIZE) :: status
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)

		
	D = 40	
	ntau = 2
	dt = 0.01d0
	T = 1000000
	
	ns = 10000
	ns_proc = ns/nprocs
	Dext = D+3
	dF = 0.01d0	
	F = 8.0d0	
    istart = 3
	iend = Dext - 1

    if(ns < T/100) then
        print *, "too few samples..."
    end if 


	!Finite Difference
!	
!	if(me==0) then
!		max_iter = 1000000
!		dXavgds_fd = 0.d0
!
!		!call RANDOM_SEED(SIZE=rsize)
!		!allocate(seed(rsize))
!		!call RANDOM_SEED(PUT=seed)	
!		call RANDOM_NUMBER(X)
!		!deallocate(seed)
!		v = 0.d0
!		Xp = X
!		print *, X
!        !Xp(istart) = Xp(istart) + 0.01d0
!		do i = 1,max_iter
!
!				
!
!			X(1) = X(Dext-2)
!			X(2) = X(Dext-1)
!			X(Dext) = X(istart)
!
!			Xp(1) = Xp(Dext-2)
!			Xp(2) = Xp(Dext-1)
!			Xp(Dext) = Xp(istart)
!
!			call Xnp1(X,Dext,Xnp1_res,F)
!			call Xnp1(Xp,Dext,Xpnp1_res,F+dF)
!	
!			if(i > 400000) then
!               
!				dXavgds_fd = dXavgds_fd + & 
!				(SUM(Xpnp1_res) - SUM(Xnp1_res))/D/dF
!			    
!                !if(MOD(i,5)==0) then
!                        write(22, *), Xpnp1_res(1)
!                !end if
!            end if
!			
!			X(3:Dext-1) = Xnp1_res
!			Xp(3:Dext-1) = Xpnp1_res
!		end do
!		dXavgds_fd = dXavgds_fd/(max_iter-400000)/dt
!		print *, "fds sens... ", dXavgds_fd
!	end if 


    !Ensemble Tangent
    !Master process
    if(me==0) then
		k=0
        allocate(X(1:Dext),thetaEA_mean(1:ntau), thetaEA_var(1:ntau))
        open(unit=20, file='EthetaEA_TEST_1e6.dat')
        open(unit=21, file='VthetaEA_TEST_1e6.dat')
        open(unit=22, file='Dynamics.dat')
        do k2 = 1,ntau
			tau = 23 + (k2-1)*400/(ntau-1)
            print *, "Short Integration Time, tau = ", tau	
			N = T/tau
            !ns/N : number of expts
            print *, "Number of expts: ", ns/N
			allocate(thetaEA(1:ns/N),dXavgds_all(1:ns))
            ns_sent = 0
		    do j=1,min(nprocs-1,ns)
                
                call RANDOM_NUMBER(X)
                !write(22, *) X
			    call MPI_SEND(X, Dext, MPI_DOUBLE_PRECISION, &
							j, tau, MPI_COMM_WORLD, ierr)
			
			    ns_sent = ns_sent + 1	
				
		    end do	
            do j = 1, ns
			    !call RANDOM_SEED(SIZE=rsize)
				!allocate(seed(rsize))
				!call RANDOM_SEED(PUT=seed)	
				call RANDOM_NUMBER(X)
                !write(22, *) X
                call MPI_RECV(dXavgds_all(j), &
                     1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, &
                     MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)

                if(dXavgds_all(j) /= dXavgds_all(j)) then
                    print *, "NaN detected! "
                end if 

                sender = status(MPI_SOURCE)

                if(ns_sent < ns) then
                
                    ns_sent = ns_sent + 1
                    call MPI_SEND(X, Dext, MPI_DOUBLE_PRECISION, &
							sender, tau, MPI_COMM_WORLD, ierr)
                else
                
                    call MPI_SEND(X, Dext, MPI_DOUBLE_PRECISION, &
							sender, 0, MPI_COMM_WORLD, ierr)
                end if
   
                if(MOD(j,25)==0) then
                    print *, "Sample Number: ", j, "tau = ", tau, &
                            "theta = ", dXavgds_all(j)
                end if

                
               !deallocate(seed)            
            end do
            if(ns/N==0) then 
                thetaEA_mean(k2) = sum(dXavgds_all(1:ns))/ns
                thetaEA_var(k2) = sum((dXavgds_all(1:ns) - thetaEA_mean(k2))**2.d0)/(ns**2.d0)
            else 
            do k1 = 1,ns/N
               
                thetaEA(k1) = sum(dXavgds_all((k1-1)*N+1:k1*N))/N
                 print *, dXavgds_all(1:15) 
            enddo
            thetaEA_mean(k2) = sum(thetaEA)/(ns/N)
            thetaEA_var(k2) = 0.d0
    
            do k1 = 1,ns/N
                thetaEA_var(k2) = thetaEA_var(k2) + (thetaEA(k1)-thetaEA_mean(k2))**2.0
            end do
            thetaEA_var(k2) = thetaEA_var(k2)/(ns/N)	
            end if
            
            
            print *, "For tau = ", tau, " E[theta_{EA}] = ", thetaEA_mean(k2)
             print *, "For tau = ", tau, " Var[theta_{EA}] = ", thetaEA_var(k2) 
            deallocate(thetaEA,dXavgds_all)

            write(20, *) thetaEA_mean(k2)
            write(21, *) thetaEA_var(k2)	

      
             
        end do
        close(20)
        close(21)
        close(22)

    end if





	!Ensemble Tangent
    ! Worker processes
    if(me /= 0) then
        allocate(X(1:Dext),v(1:Dext),vnp1_res(1:D,1),Xnp1_res(1:D), &
                	g(1:D))
        do k = 1,ntau
            g = 1.d0/D
            do while (.true.) 
           
	            call MPI_RECV(X, Dext, MPI_DOUBLE_PRECISION, &
						0, MPI_ANY_TAG, MPI_COMM_WORLD, &
						status, ierr)
			    j = status(MPI_TAG)
                !print *,"Tag sent is: ",j, "from process", me
	            tau = j	
			    if(j==0) then 
				    go to 99
                end if	
			    !Run until attractor is reached
                do i = 1, 120000
                    X(1) = X(Dext-2)
                    X(2) = X(Dext-1)
                    X(Dext) = X(istart)
                    call Xnp1(X,Dext,Xnp1_res,F)
                    X(istart:iend) = Xnp1_res
                end do
                !print *, "X is " , X
                v = 0.d0
                dXavgds = DOT_PRODUCT(v(istart:iend), g)
                do i = 1, tau	

                    v(1) = v(Dext-2)
                    v(2) = v(Dext-1)
                    v(Dext) = v(istart)
                    X(1) = X(Dext-2)
                    X(2) = X(Dext-1)
                    X(Dext) = X(istart)
                    call Xnp1(X,Dext,Xnp1_res,F)
                    call rk45_full(X,Dext,v,vnp1_res)
                    v(istart:iend) = vnp1_res(:,1)
                    X(istart:iend) = Xnp1_res 
                    dXavgds = dXavgds + DOT_PRODUCT(v(istart:iend),g)
                end do
				dXavgds = dXavgds/tau
			    call MPI_SEND(dXavgds,1,MPI_DOUBLE_PRECISION, &
									0, me, MPI_COMM_WORLD, ierr)
            end do
            99 continue
            !print *, "this is the ", k, "th tau for process", me
        end do
    end if
			
    call mpi_finalize(ierr)	
	
end program ensemble_tangent
