PROGRAM AdjoingVerification

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
    DO iSteps = 1, 3
        ALLOCATE(x(NDIM, nSteps+1))
        DO iS = 1, NPARAMS
            ds = 0.0
            ds(iS) = 1.0
            x(:,1) = 1.0
            dx(:) = 0.0
            dJtan(iS) = DT / 2 * TangentdJds(x(:,1), S0, dx, ds)
            DO iStep = 1, nSteps
                if (iStep .GT. 1) then
                    dJtan(iS) = dJtan(iS) &
                              + DT * TangentdJds(x(:,iStep), S0, dx, ds)
                end if
                CALL TangentStep(x(:,iStep), S0, dx, ds, Dcheb)
                x(:,iStep+1) = x(:,iStep)
                CALL Step(x(:,iStep+1), S0, Dcheb)
            END DO
            dJtan(iS) = dJtan(iS) &
                      + DT / 2.0_8 * TangentdJds(x(:,nSteps+1), S0, dx, ds)
        END DO
        PRINT *, dJTan
        ax(:) = 0.0
        dJadj(:) = 0.0
        CALL AdjointSource(x(:,nSteps+1), S0, ax, 0.5_8 * DT)
        DO iStep = nSteps, 1, -1
            CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj, Dcheb)
            CALL AdjointStep(x(:,iStep), S0, ax, Dcheb)
            if (iStep .GT. 1) then
                CALL AdjointSource(x(:,iStep), S0, ax, 1.0_8 * DT)
            end if
        END DO
        CALL AdjointSource(x(:, 1), S0, ax, 0.5_8 * DT)
        PRINT *, dJAdj
        nSteps = nSteps * 2
        DEALLOCATE(x)
    END DO
END PROGRAM
