PROGRAM AdjoingVerification

    Use Equations

    IMPLICIT NONE

    INTEGER, PARAMETER :: nSteps = 1000
    INTEGER :: iStep, iEps, iS
    REAL(8) :: x(NDIM, nSteps+1), dx(NDIM), ds(NPARAMS), ax(NDIM)
    REAL(8) :: dJtan(NPARAMS), dJadj(NPARAMS)

    DO iS = 1, NPARAMS
        ds = 0.0
        ds(iS) = 1.0
        x(:,1) = 1.0
        dx(:) = 0.0
        dJtan(iS) = 0
        DO iStep = 1, nSteps
            dJtan(iS) = dJtan(iS) &
                      + DT * TangentObjective(x(:,iStep), S0, dx, ds)
            CALL TangentStep(x(:,iStep), S0, dx, ds)
            x(:,iStep+1) = x(:,iStep)
            CALL Step(x(:,iStep+1), S0)
        END DO
    END DO
    PRINT *, dJTan
    ax(:) = 0.0
    dJadj(:) = 0.0
    DO iStep = nSteps, 1, -1
        CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj)
        CALL AdjointStep(x(:,iStep), S0, ax)
    END DO
    PRINT *, dJAdj
END PROGRAM
