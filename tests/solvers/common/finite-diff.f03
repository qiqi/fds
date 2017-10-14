PROGRAM Verification

    Use Equations

    IMPLICIT NONE

    REAL(8), PARAMETER :: EPS(7) = (/1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2/)
    INTEGER, PARAMETER :: nSteps = 1000
    INTEGER :: iStep, iEps, iS
    REAL(8) :: x(NDIM), dx(NDIM), s(NPARAMS), ds(NPARAMS)
    REAL(8) :: J0, dJ0, J, dJ(SIZE(EPS))

    DO iS = 1, NPARAMS
        s = S0
        ds = 0.0
        ds(iS) = 1.0
        x(:) = 1.0
        dx(:) = 0.0
        J0 = 0.0
        dJ0 = 0
        DO iStep = 1, nSteps
            dJ0 = dJ0 + DT * TangentObjective(x, s, dx, ds)
            CALL TangentStep(x, s, dx, ds)
            J0 = J0 + DT * Objective(x, s)
            CALL Step(x, s)
        END DO
        J0 = J0 / (DT * nSteps)
        dJ0 = dJ0 / (DT * nSteps)
        PRINT *, dJ0
        DO iEps = 1, SIZE(EPS)
            s = S0
            s(iS) = s(iS) + EPS(iEps)
            x(:) = 1.0
            J = 0.0
            DO iStep = 1, nSteps
                J = J + DT * Objective(x, s)
                CALL Step(x, s)
            END DO
            J = J / (DT * nSteps)
            dJ(iEps) = (J - J0) / EPS(iEps)
            PRINT '(I1,X,E6.1,X,E11.3)', iS, EPS(iEps), dJ(iEps) - dJ0
        END DO
        PRINT *
    END DO
END PROGRAM
