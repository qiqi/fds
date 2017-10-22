PROGRAM AdjoingVerification

    Use Equations

    IMPLICIT NONE

    INTEGER :: nSteps = 100000
    INTEGER :: iStep, iEps, iS, iSteps
    REAL(8), ALLOCATABLE :: x(:,:)
    REAL(8) :: dx(NDIM), ds(NPARAMS), ax(NDIM)
    REAL(8) :: dJtan(NPARAMS), dJadj(NPARAMS)

    DO iSteps = 1, 3
        ALLOCATE(x(NDIM, nSteps))
        DO iS = 1, NPARAMS
            ds = 0.0
            ds(iS) = 1.0
            x(:,1) = 1.0
            dx(:) = 0.0
            dJtan(iS) = DT / 2 * TangentObjective(x(:,iStep), S0, dx, ds)
            DO iStep = 1, nSteps
                if (iStep .GT. 1) then
                    dJtan(iS) = dJtan(iS) &
                              + DT * TangentObjective(x(:,iStep), S0, dx, ds)
                end if
                CALL TangentStep(x(:,iStep), S0, dx, ds)
                x(:,iStep+1) = x(:,iStep)
                CALL Step(x(:,iStep+1), S0)
            END DO
            dJtan(iS) = dJtan(iS) &
                      + DT / 2.0_8 * TangentObjective(x(:,nSteps+1), S0, dx, ds)
        END DO
        PRINT *, dJTan
        ax(:) = 0.0
        dJadj(:) = 0.0
        CALL AdjointSource(x(:,nSteps+1), S0, ax, 0.5_8 * DT)
        DO iStep = nSteps, 1, -1
            CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj)
            CALL AdjointStep(x(:,iStep), S0, ax)
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
