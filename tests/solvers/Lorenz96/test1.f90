! $UWHPSC/codes/fortran/newton/test1.f90

program test1

    use Lorenz96
    

    implicit none
	real(kind=8), dimension(D,1) ::X,v
	real(kind=8), dimension(D,1) :: dXdt
	real(kind=8), dimension(D,D) :: dfdX_res
	real(kind=8), dimension(D) :: A
	integer :: itest
   	do itest=1,D
    
		
		 X(itest,1) = 1.d0 
		 v(itest,1) = 1.d0

	enddo
	
	call rk4(X,v,dfdX_res)
	print *, dfdX_res(:,1)
	!print *, A

	!print *, dfdX_res(4,5)
	
end program test1
