program driver
	use OAD_active
	use OAD_rev
	use rijke_passive
	implicit none 
	external head_homogeneous
	type(active), dimension(:), allocatable :: x
    type(active), dimension(:,:), allocatable :: y
	double precision, dimension(d) :: X0, X1, Xorig
	double precision, dimension(:,:), allocatable :: v0,v,vorig
	integer :: t, nSteps, subspace_dimension, t1, t2
	double precision :: eps
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(Nparams) :: s
	character(len=128) :: arg1, arg2


	our_rev_mode%tape=.TRUE.
	eps = 1.d-4
!	c2 = 0.01d0
!	xf = 0.3d0
!	beta = 0.75d0
!	tau = 0.02d0

	
	if (command_argument_count() .ne. 2) then
        print *, "Need number of time steps and subspace dimension"
        call exit(-1)
    end if

	call get_command_argument(1, arg1)
    Read(arg1, '(i10)') nSteps

	call get_command_argument(2, arg2)
    Read(arg2, '(i10)') subspace_dimension


	Open(1, file="input.bin", form="unformatted", access="stream", &
           status="old", convert='big_endian')
    Read(1) X0
    Close(1)
	
	Xorig = X0
	
		
	allocate(v0(d,subspace_dimension))
	allocate(vorig(d,subspace_dimension))
	allocate(v(d,subspace_dimension))

	allocate(y(d,subspace_dimension))
	allocate(x(subspace_dimension))
	
	Open(1, file="tan-input.bin", form="unformatted", access="stream", &
          status="old", convert='big_endian')
    do t = 1, subspace_dimension, 1
		Read(1) v0(:,t)		
 	end do
	Close(1)
	
	vorig = v0

	Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
	do t = 1, Nparams, 1
 	 	Read(1) s(t)
	end do
    Close(1)
		
	Open(1, file="tan-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
	
 
	call cheb_diff_matrix(Dcheb)
	do t = 1, subspace_dimension, 1
		do t2 = 1, d, 1
			y(t2,t)%d = 0.d0
			y(t2,t)%d(t2) = 1.d0
			x(t)%d(t2) = 0.d0
		end do
		x(t)%v = eps
		call head_homogeneous(x(t),y(:,t),s,X0,v0(:,t),nsteps,Dcheb)

		print *, x(t)%d(1:d)
		Write(1) x(t)%d(1:d)		


		

	end do
    Close(1)
		

end program driver
