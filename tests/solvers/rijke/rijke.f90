!Simple model of combustion in gas turbines

module equations
	implicit none
	REAL, PARAMETER :: Pi = 3.1415927
	double precision, parameter :: dt = 0.001d0
	integer, parameter :: chaos_flag = 1
	integer, parameter :: N = 10, Ncheb = 10
	integer, parameter :: d = 2*N + Ncheb + 3*chaos_flag
	integer, parameter :: Nparams = 10	
    double precision, parameter :: S0(Nparams) = (/ 10.d0,10.d0,28.d0,8.d0/3.d0,0.01d0,0.3d0,0.865d0,0.05d0,0.01d0,0.04d0/) 
contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Primal solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine step(X,Xnp1,s,Dcheb)

	implicit none
	double precision, dimension(Nparams) :: s
	double precision, dimension(d):: X, Xnp1
    double precision, dimension(d):: k1, k2, k3, k4
	double precision, dimension(d):: ddt
	double precision, dimension(Ncheb+1,Ncheb+1),optional :: Dcheb
	integer :: i, imax
			
	if(present(Dcheb) .eqv. .false.) then
        Dcheb = cheb_diff_matrix()
    endif
	call dXdt(X,ddt,s,Dcheb)
    do i = 1, d, 1
		k1(i) = dt*ddt(i)
		Xnp1(i) = X(i) + 0.5d0*k1(i) 
	end do
	call dXdt(Xnp1,ddt,s,Dcheb)
    do i = 1, d, 1
		k2(i) = dt*ddt(i)
		Xnp1(i) = X(i) + 0.5d0*k2(i) 
	end do
	call dXdt(Xnp1,ddt,s,Dcheb)
    do i = 1, d, 1
		k3(i) = dt*ddt(i)
		Xnp1(i) = X(i) + k3(i) 
	end do
	call dXdt(Xnp1,ddt,s,Dcheb)
	do i = 1, d, 1
		k4(i) = dt*ddt(i) 
	end do
  
	do i = 1, d, 1
		Xnp1(i) = X(i) + 1.d0/6.d0*k1(i) + &
               1.d0/3.d0*k2(i) + 1.d0/3.d0*k3(i) + &
                1.d0/6.d0*k4(i)   

	end do


end subroutine step
double precision function Objective(X,s)
	implicit none
	double precision, intent(in), dimension(d) :: X
	double precision, dimension(Nparams) :: s
	integer :: t
	double precision :: heat_release

	heat_release = s(7)*qdot(X(2*N+Ncheb))
	Objective = 0.d0
	do t = 1, N, 1
		Objective = Objective - X(N+t)*sin(t*pi*s(6))
	end do
    Objective = Objective*heat_release
end function Objective
double precision function TangentObjective(X,s,v,ds)
	implicit none
	double precision, intent(in), dimension(d) :: X,v
	double precision, dimension(Nparams) :: s,ds
	integer :: t
	double precision :: heat_release

	heat_release = s(7)*qdot(X(2*N+Ncheb))
	TangentObjective = 0.d0
	do t = 1, N, 1
		TangentObjective = TangentObjective - X(N+t)*sin(t*pi*s(6))
	end do
    TangentObjective = TangentObjective*heat_release
end function TangentObjective



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
double precision function dqdot(delayed_velocity)
	implicit none
	double precision :: delayed_velocity
	double precision, dimension(5) :: coeffs
	integer :: i 
	coeffs(1) = 0.5d0
	coeffs(2) = -0.108d0
	coeffs(3) = -0.044d0
	coeffs(4) = 0.059d0
	coeffs(5) = -0.012d0
	dqdot = 0.d0
	do i = 1, 5, 1
			dqdot = dqdot + i*coeffs(i)*(delayed_velocity**(i-1))
	end do
end function dqdot


real(kind=8) function cheb_pts(k,n)
		implicit none
		integer:: k,n
		cheb_pts = cos(k*pi/n)
end function cheb_pts
function cheb_diff_matrix()
	implicit none
	real(kind=8), dimension(Ncheb + 1, Ncheb + 1) :: Dcheb
	real(kind=8), dimension(Ncheb + 1) :: x
	real(kind=8), dimension(Ncheb + 1, Ncheb + 1):: cheb_diff_matrix
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
			cheb_diff_matrix(j,i) = Dcheb(Ncheb - j + 2,Ncheb-i + 2)
		end do
	end do
end function cheb_diff_matrix 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Tangent solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine tangentstep(X,s,v,ds,vnp1,Dcheb)

	implicit none
	double precision, dimension(Nparams) :: s, ds
	double precision, dimension(d):: X, v, vnp1
    double precision, dimension(d):: k1, k2, k3, k4
	double precision, dimension(d):: ddt
	double precision, dimension(Ncheb+1,Ncheb+1),optional :: Dcheb
	integer :: i, imax

	if(present(Dcheb) .eqv. .false.) then
        Dcheb = cheb_diff_matrix()
    endif
	call dvdt(X,s,v,ds,ddt,Dcheb)
    do i = 1, d, 1
		k1(i) = dt*ddt(i)
		vnp1(i) = v(i) + 0.5d0*k1(i) 
	end do
	call dvdt(X,s,vnp1,ds,ddt,Dcheb)
    do i = 1, d, 1
		k2(i) = dt*ddt(i)
		vnp1(i) = v(i) + 0.5d0*k2(i) 
	end do
	call dvdt(X,s,vnp1,ds,ddt,Dcheb)
    do i = 1, d, 1
		k3(i) = dt*ddt(i)
		vnp1(i) = v(i) + k3(i) 
	end do
	call dvdt(X,s,vnp1,ds,ddt,Dcheb)
	do i = 1, d, 1
		k4(i) = dt*ddt(i) 
	end do
  
	do i = 1, d, 1
		vnp1(i) = v(i) + 1.d0/6.d0*k1(i) + &
               1.d0/3.d0*k2(i) + 1.d0/3.d0*k3(i) + &
                1.d0/6.d0*k4(i)   

	end do


end subroutine tangentstep
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Adjoint Solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine adjointstep(X,s,y,ynp1,Dcheb)

	implicit none
	double precision, dimension(Nparams) :: s
	double precision, dimension(d):: X, ynp1, y
    double precision, dimension(d):: k1, k2, k3, k4
	double precision, dimension(d):: ddt
	double precision, dimension(Ncheb+1,Ncheb+1),optional :: Dcheb
	integer :: i, imax
			
	if(present(Dcheb) .eqv. .false.) then
        Dcheb = cheb_diff_matrix()
    endif
	call dydt(X,s,y,ddt,Dcheb)
    do i = 1, d, 1
		k1(i) = dt*ddt(i)
		ynp1(i) = y(i) + 0.5d0*k1(i) 
	end do
	call dydt(X,s,ynp1,ddt,Dcheb)
    do i = 1, d, 1
		k2(i) = dt*ddt(i)
		ynp1(i) = y(i) + 0.5d0*k2(i) 
	end do
	call dydt(X,s,ynp1,ddt,Dcheb)
    do i = 1, d, 1
		k3(i) = dt*ddt(i)
		ynp1(i) = y(i) + k3(i) 
	end do
	call dydt(X,s,ynp1,ddt,Dcheb)
	do i = 1, d, 1
		k4(i) = dt*ddt(i) 
	end do
  
	do i = 1, d, 1
		ynp1(i) = y(i) + 1.d0/6.d0*k1(i) + &
               1.d0/3.d0*k2(i) + 1.d0/3.d0*k3(i) + &
                1.d0/6.d0*k4(i)   

	end do


end subroutine adjointstep


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Primal step
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine dXdt(X,dXdt_res,s,Dcheb)
	implicit none
	double precision, dimension(Nparams) :: s
	double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision :: velocity_fluctuation
	double precision :: velocity_flame
	double precision, dimension(d) :: X
	double precision :: heat_release
	double precision, intent(out), dimension(d) :: dXdt_res
	integer :: i,j

	dXdt_res(d-2) = 1.d0/s(1)*(s(2)*(X(d-1)-X(d-2)))
	dXdt_res(d-1) = 1.d0/s(1)*(X(d-2)*(s(3)-X(d)) - X(d-1))
	dXdt_res(d) = 1.d0/s(1)*(X(d-2)*X(d-1) - s(4)*X(d))
	velocity_fluctuation = s(5)*X(d-2)/(s(3) - 1.d0)
	velocity_flame = uf(X,s(6)) + velocity_fluctuation
	heat_release = s(7)*qdot(X(2*N+Ncheb))		

	do i = 1, N, 1
		dXdt_res(i) = i*pi*X(N+i)
		dXdt_res(N+i) = -1.d0*i*pi*X(i) - zeta(i,s(8),s(9))*X(N+i) &
						- 2.d0*heat_release*sin(i*pi*s(6))
	end do
	do i = 1, Ncheb, 1
		dXdt_res(2*N+i) = -2.d0/s(10)*Dcheb(i+1,1)*velocity_flame 
		do j = 2, Ncheb+1, 1 
			dXdt_res(2*N+i) = dXdt_res(2*N+i) &
							 - 2.d0/s(10)*X(2*N +j-1)*Dcheb(i+1,j)
		end do	
	end do
 
end subroutine dXdt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Tangent step
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine dvdt(X,s,v,ds,dvdt_res,Dcheb)

	implicit none
    double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(d) :: X
	double precision, dimension(d) :: dvdt_res
	double precision, dimension(d) :: v
	double precision, dimension(Nparams) :: s,ds
	double precision :: heat_release, velocity_flame
	integer :: i, j		

	dvdt_res(d-2) = s(2)/s(1)*(v(d-1)-v(d-2)) + &
		ds(2)/s(1)*(X(d-1) - X(d-2)) &
		- ds(1)/s(1)/s(1)*s(2)*(X(d-1)-X(d-2))
      
	dvdt_res(d-1) = -1.d0*ds(1)/s(1)/s(1)*((s(3)-X(d))*X(d-2) &
			- X(d-1)) + &
			+ 1.d0/s(1)*ds(3)*X(d-2) &
			+ 1.d0/s(1)*((s(3)-X(d))*v(d-2) - X(d-2)*v(d) &
				- v(d-1))
			 
	dvdt_res(d) = -1.d0*ds(1)/s(1)/s(1)*(X(d-2)*X(d-1) - s(4)*X(d)) &
			  - ds(4)/s(1)*X(d) &
			  + 1.d0/s(1)*(v(d-2)*X(d-1) + v(d-1)*X(d-2) - s(4)*v(d))
	heat_release = s(7)*qdot(X(2*N+Ncheb))
	do i = 1, N, 1
		dvdt_res(i) = i*pi*v(N+i)
		dvdt_res(N+i) = -i*pi*v(i) - zeta(i,s(8),s(9))*v(N+i) &
					-2.d0*i*pi*ds(6)*heat_release*cos(i*pi*s(6)) &
					- i*i*ds(8)*X(N+i) - ds(9)*X(N+i)/(i**0.5d0) &
					- 2.d0*sin(i*pi*s(6))*qdot(X(2*N+Ncheb))*ds(7) &
					-2.d0*s(7)*sin(i*pi*s(6))*dqdot(X(2*N+Ncheb))*v(2*N+Ncheb)	  
		
	end do
	velocity_flame = uf(X,s(6)) + s(5)*X(d-2)/(s(3) - 1.d0)

	
	do i = 1, Ncheb, 1
	
		dvdt_res(2*N+i) = 2.d0/s(10)/s(10)*ds(10)*Dcheb(i+1,1)*velocity_flame 
		do j = 2, Ncheb + 1, 1	
		
			dvdt_res(2*N+i) = dvdt_res(2*N+i) + 2.d0/s(10)/s(10)*ds(10)*Dcheb(i+1,j)*X(2*N+j-1) &
		- 2.d0/s(10)*Dcheb(i+1,j)*v(2*N+j-1) 		
		end do

		dvdt_res(2*N+i) = dvdt_res(2*N+i) + X(d-2)*ds(5)/(s(3)-1.d0) + &
					s(5)/(s(3)-1.d0)*v(d-2) &
					- s(5)/(s(3)-1.d0)/(s(3)-1.d0)*X(d-2)
		do j = 1, N, 1
		
			dvdt_res(2*N+i) = dvdt_res(2*N+i) - j*pi*ds(6)*X(j)*sin(j*pi*s(6)) &
			+ sin(j*pi*s(6))*v(j) 	

		end do

	end do	

end subroutine dvdt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Adjoint Step
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine dydt(X,s,y,dydt_res,Dcheb)

	implicit none
    double precision, dimension(Ncheb+1,Ncheb+1) :: Dcheb
	double precision, dimension(d) :: X
	double precision, dimension(d) :: dydt_res
	double precision, dimension(d) :: y
	double precision, dimension(Nparams) :: s
	double precision :: heat_release, velocity_flame
	double precision :: dheat_release
	integer :: i, j		

	dydt_res(d-2) = 2.d0/s(10)*Dcheb(Ncheb+1,1)*s(5)/(s(3)-1.d0)*y(2*N+Ncheb) - &
	(s(3)-X(d))/s(1)*y(d-1) + &
	s(2)/s(1)*y(d-2) - &
	X(d-1)/s(1)*y(d)	
       
	dydt_res(d-1) = -1.d0*s(2)/s(1)*y(d-2) &
	+ 1.d0/s(1)*y(d-1) &
	- X(d-2)/s(1)*y(d)  
   			 
	dydt_res(d) = X(d-2)/s(1)*y(d-1) &
	+ s(4)/s(1)*y(d)	
	
	heat_release = s(7)*qdot(X(2*N+Ncheb))
	dheat_release = s(7)*dqdot(X(2*N+Ncheb))
	do i = 1, N, 1
		dydt_res(i) = i*pi*y(N+i) 
		do j = 1, Ncheb, 1
			dydt_res(i) = dydt_res(i) + &
				2.d0/s(10)*Dcheb(j+1,1)*cos(i*pi*s(6))*y(2*N+j)
		end do	

		dydt_res(N+i) = - i*pi*y(i) + zeta(i,s(8),s(9))*y(N+i)  
	end do
	!velocity_flame = uf(X,s(6)) + s(5)*X(d-2)/(s(3) - 1.d0)

	do i = 1, Ncheb, 1
		
		dydt_res(2*N+i) = 0.d0
		do j = 2, Ncheb + 1, 1	
			dydt_res(2*N+i) = dydt_res(2*N+i) + &
				2.d0/s(10)*Dcheb(j,i+1)*y(2*N+j-1) 
		end do
	end do	

	do j = 1, N, 1
		dydt_res(2*N + Ncheb) = dydt_res(2*N + Ncheb) &
		+ 2.d0*dheat_release*sin(j*pi*s(6))*y(N+j)	
	end do
end subroutine dydt

end module equations
