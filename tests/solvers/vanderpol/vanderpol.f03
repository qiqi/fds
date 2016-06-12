MODULE Equations
    IMPLICIT NONE

    REAL(8), PARAMETER :: DT = 0.1, S0(1) = (/4.0/)
    INTEGER, PARAMETER :: NDIM = 2, NPARAMS = 1

    CONTAINS

SUBROUTINE Step(x, s)
    REAL(8), INTENT(inout) :: x(2)
    REAL(8), INTENT(in) :: s(1)

    REAL(8) :: dx(2)
    dx(1) = x(2)
    dx(2) = (s(1) / 4) * (1 - x(1)**2) * x(2) - x(1)
    x(:) = x(:) + DT * dx(:)
END SUBROUTINE

REAL(8) FUNCTION Objective(x, s)
    REAL(8), INTENT(in) :: x(2)
    REAL(8), INTENT(in) :: s(1)

    Objective = x(1)**2
END FUNCTION
END MODULE
