PROGRAM AdjointVerification

    Use Equations

    IMPLICIT NONE
    INTEGER, PARAMETER :: NDIM = d
    INTEGER :: nSteps = 500
    INTEGER :: iStep, iEps, iS, iSteps
    REAL(8), ALLOCATABLE :: x(:,:)
    REAL(8) :: dx(NDIM), ds(NPARAMS), ax(NDIM)
    REAL(8) :: dJtan(NPARAMS), dJadj(NPARAMS), dJadj_res(NPARAMS)
	REAL(8) :: Dcheb(Ncheb+1,Ncheb+1)
	REAL(8) :: err_norm(5)

	err_norm = 0.d0
	Dcheb = cheb_diff_matrix()	 
    DO iSteps = 1, 5
        ALLOCATE(x(NDIM, nSteps))
        DO iS = 1, NPARAMS, 1
            ds = 0.d0
            ds(iS) = 1.d0
            x(:,1) = 1.d0
            dx(:) = 0.d0
            dJtan(iS) = DT / 2.d0 * TangentdJds(x(:,1), S0, dx, ds)
			x(:,2) = x(:,1)	
			CALL Step(x(:,2), S0, Dcheb)
            CALL TangentStep(x(:,1), S0, dx, ds, Dcheb)
            DO iStep = 2, nSteps-1
                               
                dJtan(iS) = dJtan(iS) &
                              + DT * TangentdJds(x(:,iStep), S0, dx, ds)

				CALL TangentStep(x(:,iStep), S0, dx, ds, Dcheb)
                

				x(:,iStep+1) = x(:,iStep)
			
				CALL Step(x(:,istep+1), S0, Dcheb)	
            END DO
            dJtan(iS) = dJtan(iS) &
                      + DT / 2.d0 * TangentdJds(x(:,nSteps), S0, dx, ds)
        END DO
		
        PRINT *, dJTan
        ax(:) = 0.0
        dJadj(:) = 0.0
		CALL AdjointDJDS(x(:,nsteps), S0, ax, dJadj_res, Dcheb)
		dJadj = dJadj_res*0.5*dT
		
        DO iStep = nSteps-1, 2, -1
            CALL AdjointStep(x(:,iStep), S0, ax, Dcheb)
        	CALL AdjointDJDS(x(:,iStep), S0, ax, dJadj_res, Dcheb)
			dJadj = dJadj + dT*dJadj_res
		END DO
		dJadj = dJadj + 0.5*DT*dJadj_res       
        PRINT *, dJAdj
        nSteps = nSteps * 2
        DEALLOCATE(x)
		err_norm(iSteps) = maxval(abs(dJadj-dJtan))/maxval(abs(dJtan))*100.d0
    END DO
	print *, err_norm
END PROGRAM AdjointVerification
