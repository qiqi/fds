PROGRAM AdjointVerification

    Use Equations

    IMPLICIT NONE
    INTEGER, PARAMETER :: NDIM = d
    INTEGER :: nSteps = 100
    INTEGER :: iStep, iEps, iS, iSteps
    REAL(8), ALLOCATABLE :: x(:,:)
    REAL(8) :: dx(NDIM), ds(NPARAMS), ax(NDIM), axtemp(NDIM)
    REAL(8) :: dJtan(NPARAMS), dJadj(NPARAMS)
	REAL(8) :: Dcheb(Ncheb+1,Ncheb+1)

	Dcheb = cheb_diff_matrix()
    DO iSteps = 1, 1
        ALLOCATE(x(NDIM, nSteps+1))
        DO iS = 1, NPARAMS
            ds = 0.0
			!if(iS .lt. NPARAMS) then
            	ds(iS) = 1.0
			!end if
            x(:,1) = 1.0
            dx(:) = 1.3d0
            dJtan(iS) = 0.5d0 / nsteps * TangentdJds(x(:,1), S0, dx, ds)
            DO iStep = 1, nSteps
                if (iStep .GT. 1) then
                    dJtan(iS) = dJtan(iS) &
                              + 1.d0/nsteps * TangentdJds(x(:,iStep), S0, dx, ds)
                end if
                CALL TangentStep(x(:,iStep), S0, dx, ds, Dcheb)
                x(:,iStep+1) = x(:,iStep)
                CALL Step(x(:,iStep+1), S0, Dcheb)
            END DO
			!if(iS .eq. NPARAMS) then
			!	print *, dx
			!end if
            dJtan(iS) = dJtan(iS) &
                      + 0.5d0/nsteps * TangentdJds(x(:,nSteps+1), S0, dx, ds)
            if(iS .eq. 6) then
				print *, dx
				print *, dJtan(iS)
			end if

			dJtan(iS) = dJtan(iS) + dx(21)
        END DO
        !PRINT *, dJTan
        ax(:) = 0.0
        ax(21) = 1.0
        dJadj(:) = 0.0
       	axtemp(:) = 0.0 
    	CALL AdjointDJDS(x(:,nsteps+1), s0, axtemp, dJadj, Dcheb, 0.5_8/nSteps)
		CALL AdjointSource(x(:,nSteps+1), s0, ax, 0.5_8 / nSteps)
		
    	DO iStep = nSteps, 2, -1
        	CALL AdjointDJDS(x(:,iStep), s0, ax, dJadj, Dcheb, 1.0_8/nSteps)
        	CALL AdjointStep(x(:,iStep), s0, ax, Dcheb)
       		CALL AdjointSource(x(:,iStep), s0, ax, 1.0_8 / nSteps)
    	END DO
		CALL AdjointDJDS(x(:,1), s0, ax, dJadj, Dcheb, 0.5_8/nSteps)
    	CALL AdjointStep(x(:,1), s0, ax, Dcheb)
    	CALL AdjointSource(x(:,1), s0, ax, 0.5_8 / nSteps)
        
		!print *, ax
		!PRINT *, sum(ax) + dJadj
        nSteps = nSteps * 2
        DEALLOCATE(x)
    END DO
END PROGRAM
