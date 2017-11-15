subroutine head_homogeneous(eps,y,s,X0,v0,nsteps,Dcheb)
	use rijke
	implicit none 
	!double precision, dimension(d) :: X
	double precision, dimension(d)  :: X0, v0
	double precision, dimension(d), intent(out) :: y
	integer :: t, t1, t2
	integer :: nSteps
	double precision, intent(in) :: eps
	double precision, dimension(Nparams) :: s
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	
	
!$openad INDEPENDENT(eps)

	do t = 1, d, 1
		y(t) = X0(t) + eps*v0(t)
	end do

	do t = 1, nsteps, 1
	
		call step(y,s,Dcheb)

	end do
	



!$openad DEPENDENT(y)

end subroutine


