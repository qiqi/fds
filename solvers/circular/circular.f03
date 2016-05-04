MODULE Equations
    IMPLICIT NONE

    REAL(8), PARAMETER :: DT = 0.001, S0(1) = (/0.0/)
    INTEGER, PARAMETER :: NDIM = 2, NPARAMS = 1

    CONTAINS

SUBROUTINE Step(x, s)
    IMPLICIT NONE

    REAL(8), INTENT(inout) :: x(2)
    REAL(8), INTENT(in) :: s(1)

    REAL(8), PARAMETER :: dt = 0.001

    REAL(8) :: dx(2)
    dx(1) = +x(2) - (x(1)*x(1) + x(2)*x(2) - s(1) - 1) * x(1)
    dx(2) = -x(1) - (x(1)*x(1) + x(2)*x(2) - s(1) - 1) * x(2)
    x(:) = x(:) + dt * dx(:)
END SUBROUTINE

REAL(8) FUNCTION Objective(x, s)
    REAL(8), INTENT(in) :: x(2)
    REAL(8), INTENT(in) :: s(1)

    Objective = x(1)*x(1) + x(2)*x(2)
END FUNCTION
END MODULE
