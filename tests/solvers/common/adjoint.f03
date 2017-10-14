PROGRAM Adj

    USE Equations

    IMPLICIT NONE

    CHARACTER(len=128) :: arg
    INTEGER :: iStep, nSteps
    REAL(8) :: ax(NDIM), s(NPARAMS), dJds(NPARAMS)
    REAL(8), Allocatable :: x(:,:)

    if (command_argument_count() .ne. 1) then
        print *, "Need 1 argument"
        call exit(-1)
    end if

    CALL get_command_argument(1, arg)
    Read(arg, '(i10)') nSteps

    Allocate(x(NDIM, nSteps))

    Open(1, file="input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) x(:,1)
    Close(1)

    Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) s
    Close(1)

    DO iStep = 1, nSteps-1
        x(:,iStep+1) = x(:,iStep)
        CALL Step(x(:,iStep+1), s)
    END DO

    Open(1, file="adj-input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) ax
    Close(1)

    dJds(:) = 0.0
    DO iStep = nSteps, 1, -1
        CALL AdjointDJDS(x(:,iStep), S0, ax, dJds)
        CALL AdjointStep(x(:,iStep), S0, ax)
    END DO

    Open(1, file="adj-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) ax
    Close(1)

    Open(1, file="dJds.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) dJds
    Close(1)

END PROGRAM
