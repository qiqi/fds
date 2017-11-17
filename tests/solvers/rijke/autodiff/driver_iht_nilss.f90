program driver
	use OAD_active
	!use OAD_rev
	use rijke_passive
	implicit none 
	external head_inhomogeneous
	type(active), dimension(Nparams+1) :: x
    type(active), dimension(d+1) :: y
	double precision, dimension(d) :: X0, X1, Xorig
	double precision, dimension(d) :: v0,v,vorig
	integer :: t, nSteps, subspace_dimension, poi
	double precision :: eps
	double precision, dimension(:), allocatable :: J, dJ 
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(Nparams) :: ss, ds
	double precision :: Jt, dJt
	character(len=128) :: arg1


	!our_rev_mode%tape=.TRUE.
	!eps = 1.d-4
	
	if (command_argument_count() .ne. 1) then
        print *, "Need number of time steps"
        call exit(-1)
    end if

	call get_command_argument(1, arg1)
    Read(arg1, '(i10)') nSteps

	allocate(J(nsteps+1))
	allocate(dJ(nsteps+1))

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
	poi = 0	
	Open(1, file="tan-param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
	do t = 1, Nparams, 1
 	 	Read(1) ds(t)
		if(ds(t) .gt. 1.e-6) then
			poi = t
		end if
	end do
    Close(1)

	
	
    Open(1, file="output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Open(2, file="tan-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
  
    Open(3, file="J.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
 
    Open(4, file="dJ.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
   


 
		call cheb_diff_matrix(Dcheb)
		x%v = 0.d0
		if(poi .eq. 0) then 
			poi = Nparams + 1
		end if
		
		x(poi)%d = 1.d0
				
		do t = 1, d+1, 1
			y(t)%d = 0.d0
		end do
	
		do t = 1, nsteps, 1 
			call head_inhomogeneous(x,y,ss,ds,X0,v0,Dcheb)
			v0 = y(1:d)%d(poi)
			X0 = y(1:d)%v
			J(t) = y(d+1)%v
			dJ(t) = y(d+1)%d(poi)
		end do
		call head_inhomogeneous(x,y,ss,ds,X0,v0,Dcheb)
		J(t) = y(d+1)%v
		dJ(t) = y(d+1)%d(poi)
		
		write(1) X0
		write(2) v0
		write(3) J
		write(4) dJ

	
    Close(1)
	Close(2)
	Close(3)
	Close(4)	

end program driver
