subroutine head_inhomogeneous(eps,y,ss,ds,X0,v0,Dcheb)
	use rijke
	implicit none 
	
	double precision, dimension(d)  :: X0, v0
	double precision, dimension(d+1), intent(out) :: y
	integer :: t, t1
	double precision, dimension(Nparams+1), intent(in) :: eps
	double precision, dimension(Nparams) :: ss, dd, ds
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	
	
!$openad INDEPENDENT(eps)
	
	do t = 1, d, 1
		y(t) = X0(t)
		do t1 = 1, Nparams + 1, 1
			y(t) = y(t) + eps(t1)*v0(t)
		end do
	end do

	do t = 1, Nparams, 1
		dd(t) = ss(t) + eps(t)*ds(t)
	end do	
	y(d+1) = objective(y(1:d),dd)
	call step(y(1:d),dd,Dcheb)
	

!$openad DEPENDENT(y)

end subroutine


