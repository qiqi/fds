MODULE Equations
    IMPLICIT NONE

    REAL(8), PARAMETER :: DT = 0.001, S0(1) = (/0.0/)
    INTEGER, PARAMETER :: NDIM = 3, NPARAMS = 1

    CONTAINS

SUBROUTINE Step(x, s)
    IMPLICIT NONE

    REAL(8), INTENT(inout) :: x(3)
    REAL(8), INTENT(in) :: s(1)

    REAL(8), PARAMETER :: dt = 0.001

    REAL(8) :: dx(3)
    dx(1) = 10 * (x(2) - x(1))
    dx(2) = x(1) * (28 + s(1) - x(3)) - x(2)
    dx(3) = x(1) * x(2) - 8. / 3 * x(3)
    x(:) = x(:) + dt * dx(:)
END SUBROUTINE

END MODULE
