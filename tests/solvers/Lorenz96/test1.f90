! $UWHPSC/codes/fortran/newton/test1.f90

program test1

    use Lorenz96
    

    implicit none
	real(kind=8), dimension(D) ::X
	real(kind=8), dimension(D) :: dXdt
	real(kind=8), dimension(D,D) :: dfdX_res
	real(kind=8), dimension(3,3) :: A
	integer :: itest
   	do itest=1,D
    
		
		 X(itest) = 1.d0 

	enddo
	
	call dfdX(X,dfdX_res)
	!print *, dfdX_res
	
	A(2,3) = 5.d0
	print *, dfdX_res(4,5)
	print *, A
end program test1
