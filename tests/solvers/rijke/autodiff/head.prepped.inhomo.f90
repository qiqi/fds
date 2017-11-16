subroutine head_inhomogeneous(eps,y,ss,ds,X0,v0,nsteps,Dcheb)
	use rijke
	implicit none 
	
	double precision, dimension(d)  :: X0, v0
	double precision, dimension(d), intent(out) :: y
	integer :: t, t1, t2
	integer :: nSteps
	double precision, dimension(Nparams), intent(in) :: eps
	double precision, dimension(Nparams) :: ss, dd, ds
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	
	
!$openad INDEPENDENT(eps)
	
	do t = 1, d, 1
		y(t) = X0(t)
		do t1 = 1, Nparams, 1
			y(t) = y(t) + eps(t1)*v0(t)
		end do
	end do
	
	do t = 1, Nparams, 1
		!ss(t) = ss(t) + eps(t)*ds(t)
		dd(t) = ss(t) + eps(t)*ds(t)
	end do
	!print *, "After ", dd	
	do t = 1, nsteps, 1
		call step(y,dd,Dcheb)
	end do
	!y(1) = 3.d0*eps(6) + eps(5)
	!y(2) = 4.d0*eps(6)
	!print *, "After ", dd	
	!print *, "y = ", y
	


!$openad DEPENDENT(y)

end subroutine


