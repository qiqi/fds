PROGRAM Verification

    Use Equations

    IMPLICIT NONE

    REAL(8), PARAMETER :: EPS=1E-5
    INTEGER, PARAMETER :: nSteps = 1000000
    INTEGER :: iStep
    REAL(8) :: x(NDIM), y(NDIM), s(1)
    REAL(8) :: J

    s = S0
    DO WHILE (s(1) < S0(1) + 6)
        x(:) = 1.0
        DO iStep = 1, 1000
            CALL Step(x, s)
        END DO
        DO iStep = 1, nSteps
            J = J + DT * Objective(x, s)
            CALL Step(x, s)
        END DO
        J = J / (DT * nSteps)
        PRINT *, s(1), J
        s(1) = s(1) + 0.1
    END DO
END PROGRAM
