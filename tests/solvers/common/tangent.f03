PROGRAM Flow

    USE Equations

    IMPLICIT NONE

    CHARACTER(len=128) :: arg
    INTEGER :: iStep, nSteps
    REAL(8) :: x(NDIM), s(NPARAMS), dx(NDIM), ds(NPARAMS)
    REAL(8), Allocatable :: J(:), dJ(:)

    if (command_argument_count() .ne. 1) then
        print *, "Need 1 argument"
        call exit(-1)
    end if

    CALL get_command_argument(1, arg)
    Read(arg, '(i10)') nSteps

    Allocate(J(nSteps + 1))
    Allocate(dJ(nSteps + 1))

    Open(1, file="input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) x
    Close(1)

    Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) s
    Close(1)

    Open(1, file="tan-input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) dx
    Close(1)

    Open(1, file="tan-param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) ds
    Close(1)

    DO iStep = 1, nSteps
        J(iStep) = Objective(x, s)
        dJ(iStep) = TangentObjective(x, s, dx, ds)
        CALL TangentStep(x, s, dx, ds)
        CALL Step(x, s)
    END DO
    dJ(nSteps + 1) = TangentObjective(x, s, dx, ds)
    J(nSteps + 1) = Objective(x, s)

    Open(1, file="output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) x
    Close(1)

    Open(1, file="tan-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) dx
    Close(1)

    Open(1, file="J.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) J
    Close(1)

    Open(1, file="dJ.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) dJ
    Close(1)

END PROGRAM
