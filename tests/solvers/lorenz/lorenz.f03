MODULE Equations
    IMPLICIT NONE

    REAL(8), PARAMETER :: DT = 0.001, S0(3) = (/10.0, 28.0, 8./3/)
    INTEGER, PARAMETER :: NDIM = 3, NPARAMS = 3

    CONTAINS

! ---------------------------- Primal solver --------------------------------- !
SUBROUTINE Step(x, s)
    REAL(8), INTENT(inout) :: x(3)
    REAL(8), INTENT(in) :: s(3)

    REAL(8) :: dx(3)
    dx(1) = s(1) * (x(2) - x(1))
    dx(2) = x(1) * (s(2) - x(3)) - x(2)
    dx(3) = x(1) * x(2) - s(3) * x(3)
    x(:) = x(:) + DT * dx(:)
END SUBROUTINE

REAL(8) FUNCTION Objective(x, s)
    REAL(8), INTENT(in) :: x(3)
    REAL(8), INTENT(in) :: s(3)

    Objective = (x(3) - 28)**2
END FUNCTION

! ---------------------------- Tangent solver -------------------------------- !
SUBROUTINE TangentStep(x, s, dx, ds)
    REAL(8), INTENT(in) :: x(3)
    REAL(8), INTENT(in) :: s(3)
    REAL(8), INTENT(inout) :: dx(3)
    REAL(8), INTENT(in) :: ds(3)

    REAL(8) :: ddx(3)
    ddx(1) = s(1) * (dx(2) - dx(1)) + ds(1) * (x(2) - x(1))
    ddx(2) = dx(1) * (s(2) - x(3)) + x(1) * (ds(2) - dx(3)) - dx(2)
    ddx(3) = dx(1) * x(2) + x(1) * dx(2) - s(3) * dx(3) - ds(3) * x(3)
    dx(:) = dx(:) + DT * ddx(:)
END SUBROUTINE

REAL(8) FUNCTION TangentObjective(x, s, dx, ds)
    REAL(8), INTENT(in) :: x(3)
    REAL(8), INTENT(in) :: s(3)
    REAL(8), INTENT(in) :: dx(3)
    REAL(8), INTENT(in) :: ds(3)

    TangentObjective = 2 * (x(3) - 28) * dx(3)
END FUNCTION

! ---------------------------- Adjoint solver -------------------------------- !
SUBROUTINE AdjointStep(x, s, ax)
    REAL(8), INTENT(in) :: x(3)
    REAL(8), INTENT(in) :: s(3)
    REAL(8), INTENT(inout) :: ax(3)

    REAL(8) :: addx(3)
    addx = ax * DT
    ax(1) = ax(1) - addx(1) * s(1) + addx(2) * (s(2) - x(3)) + addx(3) * x(2)
    ax(2) = ax(2) + addx(1) * s(1) - addx(2) + addx(3) * x(1)
    ax(3) = ax(3) - addx(2) * x(1) - addx(3) * s(3)
    ax(3) = ax(3) + DT * 2 * (x(3) - 28)
END SUBROUTINE

SUBROUTINE AdjointDJDS(x, s, ax, dJds)
    REAL(8), INTENT(in) :: x(3)
    REAL(8), INTENT(in) :: s(3)
    REAL(8), INTENT(in) :: ax(3)
    REAL(8), INTENT(inout) :: dJds(3)

    REAL(8) :: addx(3)
    addx = ax * DT
    dJds(1) = dJds(1) + (x(2) - x(1)) * addx(1)
    dJds(2) = dJds(2) + x(1) * addx(2)
    dJds(3) = dJds(3) - x(3) * addx(3)
END SUBROUTINE

END MODULE
