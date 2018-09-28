subroutine head_adjoint(eps,y,ss,X0,v0,nsteps,Dcheb)
	use rijke
	implicit none 
	double precision, dimension(d)  :: X0, v0, X1
	double precision, intent(out) :: y
	integer :: t
	integer :: nSteps
	double precision, dimension(Nparams+d), intent(in) :: eps
	double precision, dimension(Nparams) :: ss, dd
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	
	
!$openad INDEPENDENT(eps)

	do t = 1, d, 1 
		X1(t) = X0(t) + eps(t)
	end do
	do t = 1, Nparams, 1 
		dd(t) = ss(t) + eps(d+t)
	end do

	y = 0.5d0/nsteps*Objective(X1,dd)	
	do t = 1, nsteps-1, 1 
		call step(X1, dd, Dcheb)
		y = y + 1.d0/nsteps*Objective(X1,dd)
	end do 		
	call step(X1,dd,Dcheb)
	y = y + 0.5d0/nsteps*Objective(X1,dd)
	do t = 1, d, 1
		y = y + X1(t)*v0(t)
	end do



!$openad DEPENDENT(y)

end subroutine


