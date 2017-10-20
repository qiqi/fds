MODULE Equations
    IMPLICIT NONE

    INTEGER, PARAMETER :: NDIM = 3, NPARAMS = 4
    REAL(8), PARAMETER :: DT = 0.0001_8
    REAL(8), PARAMETER :: S0(NPARAMS) = (/10.0_8, 28.0_8, 8.0_8/3, 1.0_8/)

    CONTAINS

! ---------------------------- Primal solver -------------------------------- !
SUBROUTINE Step(x, s)
    REAL(8), INTENT(inout) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)

    REAL(8) :: dx(3)
    dx(1) = s(1) * (x(2) - x(1))
    dx(2) = x(1) * (s(2) + s(4) - x(3)) - x(2)
    dx(3) = x(1) * x(2) - s(3) * (x(3) - s(4))
    x(:) = x(:) + DT * dx(:)
END SUBROUTINE

REAL(8) FUNCTION Objective(x, s)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)

    !Objective = (x(3) - 28)**2
    Objective = x(3)
END FUNCTION

! ---------------------------- Tangent solver ------------------------------- !
SUBROUTINE TangentStep(x, s, dx, ds)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)
    REAL(8), INTENT(inout) :: dx(NDIM)
    REAL(8), INTENT(in) :: ds(NPARAMS)

    REAL(8) :: ddx(3)
    ddx(1) = s(1) * (dx(2) - dx(1)) + ds(1) * (x(2) - x(1))
    ddx(2) = dx(1) * (s(2) + s(4) - x(3)) &
           + x(1) * (ds(2) + ds(4) - dx(3)) - dx(2)
    ddx(3) = dx(1) * x(2) + x(1) * dx(2) - s(3) * (dx(3) - ds(4)) &
                                         - ds(3) * (x(3) - s(4))
    dx(:) = dx(:) + DT * ddx(:)
END SUBROUTINE

REAL(8) FUNCTION TangentObjective(x, s, dx, ds)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)
    REAL(8), INTENT(in) :: dx(NDIM)
    REAL(8), INTENT(in) :: ds(NPARAMS)

    !TangentObjective = 2 * (x(3) - 28) * dx(3)
    TangentObjective = dx(3)
END FUNCTION

! ---------------------------- Adjoint solver -------------------------------- !
SUBROUTINE AdjointSource(x, s, ax, weight)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)
    REAL(8), INTENT(inout) :: ax(NDIM)
    REAL(8), INTENT(in) :: weight

    !ax(3) = ax(3) + weight * 2 * (x(3) - 28)
    ax(3) = ax(3) + weight
END SUBROUTINE

SUBROUTINE AdjointStep(x, s, ax)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)
    REAL(8), INTENT(inout) :: ax(NDIM)

    REAL(8) :: addx(3)
    addx = ax * DT
    ax(1) = ax(1) - addx(1) * s(1) + addx(2) * (s(2) + s(4) - x(3)) + addx(3) * x(2)
    ax(2) = ax(2) + addx(1) * s(1) - addx(2) + addx(3) * x(1)
    ax(3) = ax(3) - addx(2) * x(1) - addx(3) * s(3)
END SUBROUTINE

SUBROUTINE AdjointDJDS(x, s, ax, dJds)
    REAL(8), INTENT(in) :: x(NDIM)
    REAL(8), INTENT(in) :: s(NPARAMS)
    REAL(8), INTENT(inout) :: ax(NDIM)
    REAL(8), INTENT(inout) :: dJds(NPARAMS)

    REAL(8) :: addx(3)
    addx = ax * DT
    dJds(1) = dJds(1) + (x(2) - x(1)) * addx(1)
    dJds(2) = dJds(2) + x(1) * addx(2)
    dJds(3) = dJds(3) - (x(3) - s(4)) * addx(3)
    dJds(4) = dJds(4) + x(1) * addx(2) + s(3) * addx(3)
END SUBROUTINE

END MODULE
