PROGRAM AdjointVerification

    Use Equations

    IMPLICIT NONE
    INTEGER, PARAMETER :: NDIM = d
    INTEGER :: nSteps = 10
    INTEGER :: iStep, iEps, iS, iSteps
    REAL(8), ALLOCATABLE :: x(:,:)
    REAL(8) :: dx(NDIM), ds(NPARAMS), ax(NDIM)
    REAL(8) :: dJtan(NPARAMS), dJadj(NPARAMS), dJadj_res(NPARAMS)
	REAL(8) :: Dcheb(Ncheb+1,Ncheb+1)
	
	Dcheb = cheb_diff_matrix()	 
    DO iSteps = 1, 3
        ALLOCATE(x(NDIM, nSteps))
        DO iS = 1, NPARAMS
            ds = 0.d0
            ds(iS) = 1.d0
            x(:,1) = 1.d0
            dx(:) = 0.d0
            dJtan(iS) = DT / 2.d0 * TangentdJds(x(:,iStep), S0, dx, ds, Dcheb)
            DO iStep = 1, nSteps
                if (iStep .GT. 1) then
                    dJtan(iS) = dJtan(iS) &
                              + DT * TangentdJds(x(:,iStep), S0, dx, ds, Dcheb)
                end if
                CALL TangentStep(x(:,iStep), S0, dx, ds, Dcheb)
                x(:,iStep+1) = x(:,iStep)
                CALL Step(x(:,iStep+1), S0, Dcheb)
            END DO
            dJtan(iS) = dJtan(iS) &
                      + DT / 2.d0 * TangentdJds(x(:,nSteps+1), S0, dx, ds, Dcheb)
        END DO
        PRINT *, dJTan
        ax(:) = 0.0
        dJadj(:) = 0.0
		CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj, Dcheb)
		dJdadj = dJadj*0.5*dT
        DO iStep = nSteps, 1, -1
            CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj_res, Dcheb)
		
            CALL AdjointStep(x(:,iStep), S0, ax, Dcheb)

     		if (iStep .GT. 1) then
                	dJadj = dJadj + dT*dJadj_res
            end if

        END DO
  		dJadj = dJadj + 0.5*DT*dJadj_res       
        PRINT *, dJAdj
        nSteps = nSteps * 2
        DEALLOCATE(x)
    END DO
END PROGRAM AdjointVerification
