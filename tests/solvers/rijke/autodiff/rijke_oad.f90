!Simple model of combustion in gas turbines

module rijke
	implicit none
	REAL, PARAMETER :: Pi = 3.1415927
	double precision, parameter :: dt = 0.0001d0
	integer, parameter :: chaos_flag = 1
	integer, parameter :: N = 10, Ncheb = 10
	integer, parameter :: d = 2*N + Ncheb + 3*chaos_flag
	integer, parameter :: Nparams = 10	
    !double precision, parameter :: S0(Nparams) = (/ 10.d0,10.d0, &
	!28.d0, &
	!2.667d0,0.01d0,0.3d0,0.865d0,0.05d0,0.01d0,0.04d0/) 
contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Primal solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine step(X,ss,Dcheb)

	implicit none
	double precision, dimension(Nparams) :: ss
	double precision, dimension(d):: X, Xnp1
    double precision, dimension(d):: k1, k2, k3, k4
	double precision, dimension(d):: ddt
	double precision, dimension(Ncheb+1,Ncheb+1),optional :: Dcheb
	integer :: i, imax
			
	!if(present(Dcheb) .eqv. .false.) then
    !    call cheb_diff_matrix(Dcheb)
    !endif
	call dXdt(X,ddt,ss,Dcheb)
   	do i = 1, d, 1
		X(i) = X(i) + dt*ddt(i)
	end do
	!do i = 1, d, 1
	!	k1(i) = dt*ddt(i)
	!	Xnp1(i) = X(i) + 0.5d0*k1(i) 
	!end do
	!call dXdt(Xnp1,ddt,s,Dcheb)
    !do i = 1, d, 1
	!	k2(i) = dt*ddt(i)
	!	Xnp1(i) = X(i) + 0.5d0*k2(i) 
	!end do
	!call dXdt(Xnp1,ddt,s,Dcheb)
    !do i = 1, d, 1
	!	k3(i) = dt*ddt(i)
	!	Xnp1(i) = X(i) + k3(i) 
	!end do
	!call dXdt(Xnp1,ddt,s,Dcheb)
	!do i = 1, d, 1
	!	k4(i) = dt*ddt(i) 
	!end do
  
	!do i = 1, d, 1
	!	X(i) = X(i) + 1.d0/6.d0*k1(i) + &
    !          1.d0/3.d0*k2(i) + 1.d0/3.d0*k3(i) + &
    !           1.d0/6.d0*k4(i)   

	!end do


end subroutine step
double precision function Objective(X,ss)
	implicit none
	double precision, intent(in), dimension(d) :: X
	double precision, dimension(Nparams) :: ss
	integer :: t
	double precision :: heat_release

	heat_release = ss(7)*qdot(X(2*N+Ncheb))
	Objective = 0.d0
	do t = 1, N, 1
		Objective = Objective - X(N+t)*sin(t*pi*ss(6))
	end do
    Objective = Objective*heat_release
end function Objective
double precision function uf(X,xf)
	
	implicit none
	double precision, dimension(N) :: X
	double precision :: uf0, xf
	integer :: i

	uf0 = 0.d0
	do i = 1, N, 1
	
		uf0 = uf0 + X(i)*cos(i*pi*xf)	

	end do			
	uf = uf0
end function uf 
double precision function zeta(i,c1,c2)
	
	implicit none
	integer :: i
	double precision :: c1, c2
	zeta = c1*(i**2.0) + c2*(i**0.5d0)

end function zeta
double precision function qdot(delayed_velocity)
	implicit none
	double precision :: delayed_velocity
	double precision, dimension(5) :: coeffs
	integer :: i 
	coeffs(1) = 0.5d0
	coeffs(2) = -0.108d0
	coeffs(3) = -0.044d0
	coeffs(4) = 0.059d0
	coeffs(5) = -0.012d0
	qdot = 0.d0
	do i = 1, 5, 1
			qdot = qdot + coeffs(i)*(delayed_velocity**i)
	end do
end function qdot
real(kind=8) function cheb_pts(k,n)
		implicit none
		integer:: k,n
		cheb_pts = cos(k*pi/n)
end function cheb_pts
subroutine cheb_diff_matrix(chebD)
	implicit none
	real(kind=8), dimension(Ncheb + 1, Ncheb + 1) :: Dcheb, chebD
	real(kind=8), dimension(Ncheb + 1) :: x
	!real(kind=8), dimension(Ncheb + 1, Ncheb + 1):: cheb_diff_matrix
	integer :: i,j,k
	Dcheb = 0.d0

	do k = 1, Ncheb + 1, 1
		x(k) = cheb_pts(k-1,Ncheb) 
	end do
	if(Ncheb > 1) then
		do i = 2, Ncheb, 1
			Dcheb(i,i) = -1.d0*x(i)/(1.d0 - x(i)**2.d0)/2.d0
			do j  = i+1, Ncheb, 1
				Dcheb(i,j) = (-1.d0)**(i+j)/(x(i)-x(j))
				Dcheb(j,i) = -1.d0*Dcheb(i,j)
			end do
		end do
	end if
	Dcheb(1,1) = (2.0*(Ncheb**2.0)+1.d0)/6.d0
	Dcheb(Ncheb+1,Ncheb+1) = -1.d0*Dcheb(1,1)
	do j = 2, Ncheb + 1, 1
		Dcheb(1,j) = 2.d0*(-1)**(j-1)/(x(1)-x(j))
	end do
	Dcheb(1,Ncheb+1) = Dcheb(1,Ncheb+1)/2.d0
	Dcheb(2:Ncheb+1,1) = -0.25d0*Dcheb(1,2:Ncheb+1)
	Dcheb(Ncheb+1,1) = 4.d0*Dcheb(Ncheb+1,1)
	do j = 2, Ncheb, 1
		Dcheb(Ncheb+1,j) = 2.d0*(-1)**(Ncheb+j-1)/(x(Ncheb+1)-x(j))
	end do
	Dcheb(2:Ncheb,Ncheb+1) = -1.d0*Dcheb(Ncheb+1,2:Ncheb)/4.d0 
	do j = 1, Ncheb + 1, 1
		do i = 1, Ncheb + 1, 1	
			chebD(j,i) = Dcheb(Ncheb - j + 2,Ncheb-i + 2)
		end do
	end do
end subroutine cheb_diff_matrix 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Primal step
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine dXdt(X,dXdt_res,ss,Dcheb)
	implicit none
	double precision, dimension(Nparams) :: ss
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision :: velocity_fluctuation
	double precision :: velocity_flame
	double precision, dimension(d) :: X
	double precision :: heat_release
	double precision, intent(out), dimension(d) :: dXdt_res
	integer :: i,j

	dXdt_res(d-2) = 1.d0/ss(1)*(ss(2)*(X(d-1)-X(d-2)))
	dXdt_res(d-1) = 1.d0/ss(1)*(X(d-2)*(ss(3)-X(d)) - X(d-1))
	dXdt_res(d) = 1.d0/ss(1)*(X(d-2)*X(d-1) - ss(4)*X(d))
	velocity_fluctuation = ss(5)*X(d-2)/(ss(3) - 1.d0)
	velocity_flame = uf(X,ss(6)) + velocity_fluctuation
	heat_release = ss(7)*qdot(X(2*N+Ncheb))		

	do i = 1, N, 1
		dXdt_res(i) = i*pi*X(N+i)
		dXdt_res(N+i) = -1.d0*i*pi*X(i) - zeta(i,ss(8),ss(9))*X(N+i) &
						- 2.d0*heat_release*sin(i*pi*ss(6))
	end do
	do i = 1, Ncheb, 1
		dXdt_res(2*N+i) = -2.d0/ss(10)*Dcheb(i+1,1)*velocity_flame 
		do j = 2, Ncheb+1, 1 
			dXdt_res(2*N+i) = dXdt_res(2*N+i) &
							 - 2.d0/ss(10)*X(2*N +j-1)*Dcheb(i+1,j)
		end do	
	end do
 
end subroutine dXdt
end module rijke
