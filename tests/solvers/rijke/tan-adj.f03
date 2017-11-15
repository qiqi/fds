PROGRAM AdjointVerification

    Use Equations

    IMPLICIT NONE
    INTEGER, PARAMETER :: NDIM = d
    INTEGER :: nSteps = 1000
    INTEGER :: iStep, iEps, iS, iSteps
    REAL(8), ALLOCATABLE :: x(:,:)
    REAL(8) :: dx(NDIM), ds(NPARAMS), ax(NDIM)
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
            dx(:) = 0.0
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
		
            dJtan(iS) = dJtan(iS) &
                      + 0.5d0/nsteps * TangentdJds(x(:,nSteps+1), S0, dx, ds)
        END DO
        !PRINT *, dJTan
        ax(:) = 1.0
        dJadj(:) = 0.0
        CALL AdjointSource(x(:,nSteps+1), S0, ax, 0.5d0/nsteps)
        !print *, "Product at nsteps: ", sum(dx)
		DO iStep = nSteps, 1, -1
            CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj, Dcheb)
            CALL AdjointStep(x(:,iStep), S0, ax, Dcheb)
            if (iStep .GT. 1) then
                CALL AdjointSource(x(:,iStep), S0, ax, 1.d0/nsteps)
            end if
        END DO
        CALL AdjointSource(x(:, 1), S0, ax, 0.5d0/nsteps)
		print *, "Product at step 1:", sum(ax)
        PRINT *, dJAdj
        nSteps = nSteps * 2
        DEALLOCATE(x)
    END DO
END PROGRAM
