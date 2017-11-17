program driver
	use OAD_active
	use OAD_rev
	use rijke_passive
	implicit none 
	external head_homogeneous
	type(active), dimension(Nparams+d) :: x
    type(active) :: y
	double precision, dimension(d) :: X0
	double precision, dimension(d) :: v0
	integer :: t, nSteps
	double precision :: eps
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(Nparams) :: ss
	character(len=128) :: arg1, arg2


	our_rev_mode%tape=.TRUE.

	
	if (command_argument_count() .ne. 1) then
        print *, "Need number of time steps"
        call exit(-1)
    end if

	call get_command_argument(1, arg1)
    Read(arg1, '(i10)') nSteps


	Open(1, file="input.bin", form="unformatted", access="stream", &
           status="old", convert='big_endian')
    Read(1) X0
    Close(1)
	
	
	Open(1, file="adj-input.bin", form="unformatted", access="stream", &
          status="old", convert='big_endian')
    
		Read(1) v0		
 	Close(1)
	

	Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
	do t = 1, Nparams, 1
 	 	Read(1) ss(t)
	end do
    Close(1)
		
	Open(1, file="adj-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
	
  	Open(2, file="dJds.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')

	call cheb_diff_matrix(Dcheb)
	y%d = 1.d0
	x%v = 0.d0
	!print *, v0	
	call head_adjoint(x,y,ss,X0,v0,nsteps,Dcheb)

	do t = 1, d, 1
		!print *, x(t)%d(1)
		Write(1) x(t)%d(1)	
	end do
	Close(1)
	do t = 1, Nparams, 1
		!print *, x(t+d)%d(1)
		Write(2) x(t+d)%d(1)	
	end do
	Close(2)
		

	
    
		

end program driver
