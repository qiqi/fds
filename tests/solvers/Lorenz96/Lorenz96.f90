! Lorenz ' 96 system

module Lorenz96

    ! system parameters:
    real(kind=8), parameter :: M = 10.d0
contains

subroutine Xnp1(X,D,Xnp1_res)

        
    implicit none
	integer, intent(in) :: D
	real(kind=8), intent(in), dimension(D) :: X
	real(kind=8), intent(out), dimension(D-3) :: Xnp1_res
	integer :: i
	real(kind=8) :: dt

	dt = 0.001d0
	do i=3,D-1
		Xnp1_res(i-2) = (-X(i-2) + X(i+1))*X(i-1) - X(i) + M	
	end do
	!dXdt(1) = (-X(D-1) + X(2))*X(D) - X(1) + M
  	!dXdt(2) = (-X(D) + X(3))*X(1) - X(2) + M
	!dXdt(D) = (-X(D-2) + X(1))*X(D-1) - X(D) + M
	Xnp1_res = X(1:D-3) + dt*Xnp1_res
end subroutine Xnp1

subroutine dfdX(X,D,dfdX_res)

	implicit none
	integer, intent(in) :: D
	real(kind=8), intent(in), dimension(D) :: X
	real(kind=8), intent(out), dimension(D-3,D-3) :: dfdX_res 	
	integer :: i,j
	
	dfdX_res = 0.d0
	do i=3,D-1
		j = i - 2	
		dfdX_res(j,j) = -1.d0
		dfdX_res(j,j-1) = -X(i-2) + X(i+1)
		dfdX_res(j,j+1) = X(i-1)
		dfdX_res(j,j-2) = -X(i-1)  					

	enddo		
!	dfdX_res(1,D) = X(2) - X(D-1)
!	dfdX_res(1,D-1) = -X(D)
!	dfdX_res(1,1) = -1.d0
!	dfdX_res(1,2) = X(D)
	
	    		
!	dfdX_res(2,1) = X(3) - X(D)
!	dfdX_res(2,D) = -X(1)
!	dfdX_res(2,2) = -1.d0
!	dfdX_res(2,3) = X(1)
	
!	dfdX_res(D,D-1) = X(1) - X(D-2)
!	dfdX_res(D,D-2) = -X(D-1)
!	dfdX_res(D,D) = -1.d0
!	dfdX_res(D,1) = X(D-1)

end subroutine dfdX  

subroutine dvdt(X,D,v,dvdt_res)

		implicit none
		integer, intent(in) :: D
		real(kind=8), intent(in), dimension(D) :: X
		real(kind=8), intent(in), dimension(D-3) :: v
		real(kind=8), intent(out), dimension(D-3,1) :: dvdt_res
		real(kind=8), dimension(D-3,D-3) :: dfdX_res
		real(kind=8), dimension(D-3,1) :: v1
		
		
		call dfdX(X,D,dfdX_res)
		v1 = reshape(v,[D-3,1])
		dvdt_res = matmul(dfdX_res,v1)	
		 	
end subroutine dvdt
subroutine rk4(X,D,v,vnp1)

	implicit none
	integer, intent(in) :: D
	real(kind=8) , intent(in), dimension(D,1) :: X
	real(kind=8) , intent(in), dimension(D-3,1) :: v
	real(kind=8) , intent(out), dimension(D-3,1) :: vnp1
	real(kind=8) , dimension(D-3,1) :: dvdt_res,k1,k2,k3,k4
	real(kind=8) :: dt
	

	dt = 0.001d0	
	call dvdt(X,D,v,dvdt_res)
	k1 = dt*dvdt_res
	call dvdt(X,D,v + 0.5d0*k1,dvdt_res)
	k2 = dt*dvdt_res
	call dvdt(X,D,v + 0.5d0*k2,dvdt_res)
	k3 = dt*dvdt_res
	call dvdt(X,D,v + k3,dvdt_res)
	k4 = dt*dvdt_res

	vnp1 = v + 1.d0/6.d0*k1 + &
		 1.d0/3.d0*k2 + 1.d0/3.d0*k3 + &
		1.d0/6.d0*k4
	

end subroutine rk4 

end module Lorenz96
