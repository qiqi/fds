program driver
	use OAD_active
	!use OAD_rev
	use rijke_passive
	implicit none 
	external head_inhomogeneous
	type(active), dimension(Nparams) :: x
    type(active), dimension(d) :: y
	double precision, dimension(d) :: X0, X1, Xorig
	double precision, dimension(d) :: v0,v,vorig
	integer :: t, nSteps, subspace_dimension, poi
	double precision :: eps
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(Nparams) :: ss, ds
	character(len=128) :: arg1


	!our_rev_mode%tape=.TRUE.
	!eps = 1.d-4
	
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
	
	Xorig = X0
	
		
	
	Open(1, file="tan-input.bin", form="unformatted", access="stream", &
          status="old", convert='big_endian')
    	Read(1) v0		
	Close(1)
	
	vorig = v0

	Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
	do t = 1, Nparams, 1
 	 	Read(1) ss(t)
	end do
    Close(1)
	
	Open(1, file="tan-param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
	do t = 1, Nparams, 1
 	 	Read(1) ds(t)
		if(ds(t) .gt. 1.e-6) then
			poi = t
		end if
	end do
    Close(1)

	
	Open(1, file="tan-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
	
 
		call cheb_diff_matrix(Dcheb)
		x%v = 0.d0
		x(poi)%d = 1.d0		
		do t = 1, d, 1
			y(t)%d = 0.d0
		end do
		call head_inhomogeneous(x,y,ss,ds,X0,v0,nsteps,Dcheb)
		
		do t = 1, d, 1
			print *, y(t)%d(poi)
		end do
		!Write(1) y		


		

	
    Close(1)
		

end program driver
