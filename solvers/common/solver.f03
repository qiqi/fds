PROGRAM Flow

    USE Equations

    IMPLICIT NONE

    CHARACTER(len=128) :: arg
    INTEGER :: iStep, nSteps
    REAL(8) :: x(NDIM), s(NPARAMS)
    REAL(8), Allocatable :: J(:)

    if (command_argument_count() .ne. 1) then
        print *, "Need 1 argument"
        call exit(-1)
    end if

    CALL get_command_argument(1, arg)
    Read(arg, '(i10)') nSteps

    Allocate(J(nSteps))

    Open(1, file="input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) x
    Close(1)

    Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) s
    Close(1)

    DO iStep = 1, nSteps
        CALL Step(x, s)
        J(iStep) = Objective(x, s)
    END DO

    Open(1, file="output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) x
    Close(1)

    Open(1, file="objective.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) J
    Close(1)

END PROGRAM
