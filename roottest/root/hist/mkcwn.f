*     Generator for cwn.hbook, the input of the h2root-cwn test.
*     Build against CERNLIB, which must be linked without PIE so that the
*     commons live in the low 32 bit address space:
*       gfortran -fno-pie -no-pie -fno-automatic -o mkcwn mkcwn.f \
*                -lpacklib -lmathlib -lkernlib
      PROGRAM MKCWN
      INTEGER NWPAWC
      PARAMETER (NWPAWC = 100000)
      COMMON /PAWC/ H(NWPAWC)
      INTEGER IVAR, NEL
      REAL    FVAR, ARR(3)
      COMMON /CWNBLK/ IVAR, FVAR, NEL, ARR
      INTEGER ISTAT, ICYCLE, I, J
      CALL HLIMIT(NWPAWC)
      CALL HROPEN(10, 'TEST', 'cwn.hbook', 'N', 1024, ISTAT)
      IF (ISTAT .NE. 0) STOP 'HROPEN failed'
      CALL HBOOK1(1, 'FVAR', 10, 0., 10., 0.)
      CALL HBNT(100, 'cwn test', ' ')
      CALL HBNAME(100, 'SCALARS', IVAR, 'IVAR:I*4,FVAR:R*4')
      CALL HBNAME(100, 'ARRAYS', NEL, 'NEL[0,3]:I*4,ARR(NEL):R*4')
      DO 20 I = 1, 7
         IVAR = I
         FVAR = 0.5*I
         NEL  = MOD(I, 4)
         DO 10 J = 1, NEL
            ARR(J) = I + 0.1*J
   10    CONTINUE
         CALL HFILL(1, FVAR, 0., 1.)
         CALL HFNT(100)
   20 CONTINUE
      CALL HROUT(0, ICYCLE, ' ')
      CALL HREND('TEST')
      CLOSE(10)
      END
