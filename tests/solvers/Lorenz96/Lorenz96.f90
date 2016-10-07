! Lorenz ' 96 system

module Lorenz96

    ! system parameters:
    integer, parameter :: D = 36
    real(kind=8), parameter :: M = 10.d0
contains

subroutine f(X,dXdt)

        
    implicit none
	real(kind=8), intent(in), dimension(D) :: X
	real(kind=8), intent(out), dimension(D) :: dXdt
	integer :: i
	do i=3,D-1
		dXdt(i) = (-X(i-2) + X(i+1))*X(i-1) - X(i) + M	
	end do
	dXdt(1) = (-X(D-1) + X(2))*X(D) - X(1) + M
  	dXdt(2) = (-X(D) + X(3))*X(1) - X(2) + M
	dXdt(D) = (-X(D-2) + X(1))*X(D-1) - X(D) + M
end subroutine f

subroutine 

end module Lorenz96
