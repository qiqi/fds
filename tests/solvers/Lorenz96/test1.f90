! $UWHPSC/codes/fortran/newton/test1.f90

program test1

    use Lorenz96
    

    implicit none
	real(kind=8), dimension(D) ::X
	real(kind=8), dimension(D) :: dXdt
	integer :: itest
   	do itest=1,D
    
		
		 X(itest) = 1.d0 

	enddo
	call f(X,dXdt)
	print *, dXdt
end program test1
