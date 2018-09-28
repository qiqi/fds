PROGRAM Adj

    USE Equations

    IMPLICIT NONE

    CHARACTER(len=128) :: arg
    INTEGER :: iStep, nSteps
	INTEGER, PARAMETER :: NDIM = d
    REAL(8) :: ax(NDIM), s(NPARAMS), dJds(NPARAMS), axtemp(NDIM)
    REAL(8), Allocatable :: x(:,:)
	REAL(8) :: Dcheb(Ncheb+1,Ncheb+1)
    if (command_argument_count() .ne. 1) then
        print *, "Need 1 argument"
        call exit(-1)
    end if

    CALL get_command_argument(1, arg)
    Read(arg, '(i10)') nSteps

    Allocate(x(NDIM, nSteps + 1))

    Open(1, file="input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) x(:,1)
    Close(1)

    Open(1, file="param.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) s
    Close(1)

	Open(1, file="Dcheb.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) Dcheb
    Close(1)



    DO iStep = 1, nSteps
        x(:,iStep+1) = x(:,iStep)
        CALL Step(x(:,iStep+1), s, Dcheb)
    END DO

    Open(1, file="adj-input.bin", form="unformatted", access="stream", &
            status="old", convert='big_endian')
    Read(1) ax
    Close(1)
	axtemp(:)= 0.0
    dJds(:) = 0.0
	CALL AdjointDJDS(x(:,nSteps+1), s, axtemp, dJds, Dcheb, 0.5_8/nSteps)
    CALL AdjointSource(x(:,nSteps+1), s, ax, 0.5_8 / nSteps)
	
    DO iStep = nSteps, 2, -1
        CALL AdjointDJDS(x(:,iStep), s, ax, dJds, Dcheb, 1.0_8/nSteps)
        CALL AdjointStep(x(:,iStep), s, ax, Dcheb)
       	CALL AdjointSource(x(:,iStep), s, ax, 1.0_8 / nSteps)
    END DO
	CALL AdjointDJDS(x(:,1), s, ax, dJds, Dcheb, 0.5_8/nSteps)
    CALL AdjointStep(x(:,1), s, ax, Dcheb)

    CALL AdjointSource(x(:,1), s, ax, 0.5_8 / nSteps)

    Open(1, file="adj-output.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) ax
    Close(1)

    Open(1, file="dJds.bin", form="unformatted", access="stream", &
         status='replace', convert='big_endian')
    Write(1) dJds
    Close(1)

END PROGRAM
