*-------------------------------------------------------------------------------
*
* This file contains the kernlib's package subset needed to build h2root.
* It cannot be used by any kernlib application because many kernlib
* functionalities * are missing.
*
*-------------------------------------------------------------------------------

      SUBROUTINE CLTOU (CHV)
      CHARACTER    CHV*(*)
      DO 19  JJ=1,LEN(CHV)
          J = ICHAR(CHV(JJ:JJ))
          IF (J.LT.97)       GO TO 19
          IF (J.GE.123)      GO TO 19
          CHV(JJ:JJ) = CHAR(J-32)
   19 CONTINUE
      END

*-------------------------------------------------------------------------------

      SUBROUTINE CUTOL (CHV)
      CHARACTER    CHV*(*)
      DO 19  JJ=1,LEN(CHV)
          J = ICHAR(CHV(JJ:JJ))
          IF (J.LT.65)       GO TO 19
          IF (J.GE.91)       GO TO 19
          CHV(JJ:JJ) = CHAR(J+32)
   19 CONTINUE
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UOPTC (DATA,POSS,IOPTV)
      INTEGER      IOPTV(9)
      CHARACTER    *(*)  DATA, POSS
      NP = LEN(POSS)
      DO 24 J=1,NP
      IOPTV(J) = 0
      IF (INDEX(DATA,POSS(J:J)).NE.0)  IOPTV(J)= 1
   24 CONTINUE
      END

*-------------------------------------------------------------------------------

      FUNCTION LENOCC (CHV)
      CHARACTER    CHV*(*)
      N = LEN(CHV)
      DO 17  JJ= N,1,-1
      IF (CHV(JJ:JJ).NE.' ') GO TO 99
   17 CONTINUE
      JJ = 0
   99 LENOCC = JJ
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UBUNCH (MS,MT,NCHP)
      DIMENSION MS(99), MT(99), NCHP(9)
*      data iblan1/x'20202020'/
*      data mask1/x'000000ff'/
      data iblan1/538976288/
      data mask1/255/
      NCH = NCHP(1)
      IF (NCH) 91,39,11
   11 NWT = ISHFT (NCH,-2)
      NTRAIL = IAND (NCH,3)
      JS = 0
      IF (NWT.EQ.0) GO TO 31
      DO 24 JT=1,NWT
      MT(JT) = IOR (IOR (IOR (
     + IAND(MS(JS+1),MASK1),
     + ISHFT (IAND(MS(JS+2),MASK1), 8)),
     + ISHFT (IAND(MS(JS+3),MASK1),16)),
     + ISHFT (MS(JS+4), 24) )
   24 JS = JS + 4
      IF (NTRAIL.EQ.0) RETURN
   31 MWD = IBLAN1
      JS = NCH
      DO 34 JT=1,NTRAIL
      MWD = IOR (ISHFT(MWD,8), IAND(MS(JS),MASK1))
   34 JS = JS - 1
      MT(NWT+1) = MWD
   39 RETURN
   91 PRINT*, '>>> Abnormal end'
      END

*-------------------------------------------------------------------------------

      FUNCTION LOCF (IVAR)
      DIMENSION    IVAR(9)
      PARAMETER    (NADUPW=4, LADUPW=2)
      J = LOC(IVAR)
      LOCF = ISHFT (J, -LADUPW)
      END

      FUNCTION LOCFR (VAR)
      DIMENSION    VAR(9)
      PARAMETER    (NADUPW=4, LADUPW=2)
      J = LOC(VAR)
      LOCFR = ISHFT (J, -LADUPW)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE VZERO (A,N)
      DIMENSION A(*)
      IF (N.LE.0)  RETURN
      DO 9 I= 1,N
    9 A(I)= 0.
      END

      SUBROUTINE VZEROI (J,N)
      DIMENSION J(*)
      IF (N.LE.0)  RETURN
      DO 9 I= 1,N
    9 J(I)= 0.
      END

*-------------------------------------------------------------------------------

      FUNCTION JBIT (IZW,IZP)
      JBIT = IAND (ISHFT(IZW, -(IZP-1)), 1)
      END

*-------------------------------------------------------------------------------

      FUNCTION JBYT (IZW,IZP,NZB)
      PARAMETER (NBITPW=32)
      PARAMETER (NCHAPW=4)
      JBYT = ISHFT(ISHFT(IZW,NBITPW+1-IZP-NZB), -(NBITPW-NZB))
      END

*-------------------------------------------------------------------------------

      FUNCTION LOCATI(ARRAY,LENGTH,OBJECT)
      DIMENSION ARRAY(2)
      INTEGER ARRAY,OBJECT
      NABOVE = LENGTH + 1
      NBELOW = 0
   10 IF (NABOVE-NBELOW .LE. 1)  GO TO 200
      MIDDLE = (NABOVE+NBELOW) / 2
      IF (OBJECT - ARRAY(MIDDLE))  100, 180, 140
  100 NABOVE = MIDDLE
      GO TO 10
  140 NBELOW = MIDDLE
      GO TO 10
  180 LOCATI = MIDDLE
      GO TO 300
  200 LOCATI = -NBELOW
  300 RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE VFILL (A,N,STUFF)
      INTEGER   A(*), STUFF
      IF (N.LE.0)  RETURN
      DO 9 I= 1,N
    9 A(I)= STUFF
      END

*-------------------------------------------------------------------------------

      SUBROUTINE VBLANK (A,N)
      INTEGER      A(*), BLANK
      DATA  BLANK  / 4H     /
      IF (N.LE.0)  RETURN
      DO 9 I= 1,N
    9 A(I)= BLANK
      END

*-------------------------------------------------------------------------------

      SUBROUTINE SBYT (IT,IZW,IZP,NZB)
      PARAMETER (NBITPW=32)
      PARAMETER (NCHAPW=4)
      PARAMETER (IALL11 = -1)
      MSK = ISHFT (IALL11, -(NBITPW-NZB))
      IZW = IOR ( IAND (IZW, NOT(ISHFT(MSK,IZP-1)))
     +, ISHFT(IAND(IT,MSK),IZP-1))
      END

*-------------------------------------------------------------------------------

      SUBROUTINE SBIT1 (IZW,IZP)
      IZW = IOR (IZW, ISHFT(1,IZP-1))
      END

*-------------------------------------------------------------------------------

      SUBROUTINE SBIT0 (IZW,IZP)
      IZW = IAND (IZW, NOT(ISHFT(1,IZP-1)) )
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UCTOH (MS,MT,NPW,NCH)
      CHARACTER    MS*99
      DIMENSION    MT(99)
      PARAMETER   (NBITPW=32)
      PARAMETER   (NCHAPW=4)
      CHARACTER    CHWORD*(NCHAPW)
      CHARACTER    BLANK *(NCHAPW)
      PARAMETER   (BLANK = ' ')
      INTEGER      IWORD
      EQUIVALENCE (IWORD,CHWORD)
      IF   (NCH)             91, 29, 11
   11 IF (NPW.LE.0)          GO TO 91
      IF (NPW.EQ.1)          GO TO 21
      IF (NPW.LT.NCHAPW)     GO TO 31
      JS     = 0
      NWT    = NCH / NCHAPW
      NTRAIL = NCH - NWT*NCHAPW
      DO 14  JT=1,NWT
      CHWORD = MS(JS+1:JS+NCHAPW)
      MT(JT) = IWORD
   14 JS     = JS + NCHAPW
      IF (NTRAIL.EQ.0)       RETURN
      CHWORD    = MS(JS+1:JS+NTRAIL)
      MT(NWT+1) = IWORD
      RETURN
   21 CHWORD = BLANK
      DO 24 JS=1,NCH
      CHWORD(1:1) = MS(JS:JS)
      MT(JS)      = IWORD
   24 CONTINUE
   29 RETURN
   31 CHWORD = BLANK
      JS     = 0
      NWT    = NCH / NPW
      NTRAIL = NCH - NWT*NPW
      DO 34  JT=1,NWT
      CHWORD(1:NPW) = MS(JS+1:JS+NPW)
      MT(JT) = IWORD
   34 JS     = JS + NPW
      IF (NTRAIL.EQ.0)       RETURN
      CHWORD    = MS(JS+1:JS+NTRAIL)
      MT(NWT+1) = IWORD
      RETURN
   91 PRINT*, '>>> Abnormal end'
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UHTOC (MS,NPW,MT,NCH)
      DIMENSION    MS(99)
      CHARACTER    MT*99
      PARAMETER   (NCHAPW=4)
      CHARACTER    CHWORD*(NCHAPW)
      INTEGER      IWORD
      EQUIVALENCE (IWORD,CHWORD)
      IF   (NCH)             91, 29, 11
   11 IF (NPW.LE.0)          GO TO 91
      IF (NPW.EQ.1)          GO TO 21
      IF (NPW.LT.NCHAPW)     GO TO 31
      JT     = 0
      NWS    = NCH / NCHAPW
      NTRAIL = NCH - NWS*NCHAPW
      DO 14  JS=1,NWS
      IWORD  = MS(JS)
      MT(JT+1:JT+NCHAPW) = CHWORD
   14 JT  = JT + NCHAPW
      IF (NTRAIL.EQ.0)       RETURN
      IWORD = MS(NWS+1)
      MT(JT+1:JT+NTRAIL) = CHWORD(1:NTRAIL)
      RETURN
   21 DO 24  JS=1,NCH
      IWORD  = MS(JS)
      MT(JS:JS) = CHWORD(1:1)
   24 CONTINUE
   29 RETURN
   31 JT     = 0
      NWS    = NCH / NPW
      NTRAIL = NCH - NWS*NPW
      DO 34  JS=1,NWS
      IWORD  = MS(JS)
      MT(JT+1:JT+NPW) = CHWORD(1:NPW)
   34 JT  = JT + NPW
      IF (NTRAIL.EQ.0)       RETURN
      IWORD = MS(NWS+1)
      MT(JT+1:JT+NTRAIL) = CHWORD(1:NTRAIL)
      RETURN
   91 print *,' UHTOC: wrong args.'
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UCTOH1 (MS,MT,NCH)
      CHARACTER    MS*99
      DIMENSION    MT(99)
      PARAMETER   (NBITPW=32)
      PARAMETER   (NCHAPW=4)
      CHARACTER    CHWORD*(NCHAPW)
      CHARACTER    BLANK *(NCHAPW)
      PARAMETER   (BLANK = ' ')
      INTEGER      IWORD
      EQUIVALENCE (IWORD,CHWORD)
      IF   (NCH)             91, 29, 11
   11 CHWORD = BLANK
      DO 24 JS=1,NCH
      CHWORD(1:1) = MS(JS:JS)
      MT(JS)      = IWORD
   24 CONTINUE
   29 RETURN
   91 PRINT*, '>>> Abnormal end'
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UCOPYI (IA,IB,N)
      DIMENSION IA(*),IB(*)
      IF (N.EQ.0) RETURN
         DO 21 I=1,N
   21 IB(I)=IA(I)
      END
      SUBROUTINE UCOPY (A,B,N)
      DIMENSION A(*),B(*)
      IF (N.EQ.0) RETURN
         DO 21 I=1,N
   21 B(I)=A(I)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UPKBYT (MBV,JTHP,MIV,NINTP,NBITS)
      DIMENSION MIV(99), MBV(99), JTHP(9), NINTP(9), NBITS(2)
      PARAMETER (NBITPW=32)
      PARAMETER (NCHAPW=4)
      PARAMETER (IALL11 = -1)

      JTH = JTHP(1)
      NINT = NINTP(1)
      IF (NINT.LE.0) RETURN
      NZB = NBITS(1)
      IF (NZB.GT.0) GO TO 11
      NZB = 1
      NPWD = NBITPW
      MSKA = 1
      GO TO 12
   11 NPWD = NBITS(2)
      MSKA = ISHFT (IALL11, -(NBITPW-NZB))
   12 JBV = 1
      JIV = 0
      IF (JTH.LT.2) GO TO 21
      JBV = (JTH-1)/NPWD + 1
      JPOS = JTH - (JBV-1)*NPWD - 1
      IF (JPOS.EQ.0) GO TO 21
      NR = JPOS*NZB
      JIVE = NPWD - JPOS
      IZW = ISHFT (MBV(JBV), -NR)
      GO TO 22
   21 JIVE = JIV + NPWD
      IZW = MBV(JBV)
   22 JIVE = MIN (NINT,JIVE)
   24 JIV = JIV + 1
      MIV(JIV) = IAND (MSKA,IZW)
      IF (JIV.EQ.JIVE) GO TO 27
      IZW = ISHFT (IZW, -NZB)
      GO TO 24
   27 IF (JIV.EQ.NINT) RETURN
      JBV = JBV + 1
      GO TO 21
      END

*-------------------------------------------------------------------------------

      SUBROUTINE CFOPEN (LUNDES, MEDIUM, NWREC, MODE, NBUF, NAME, ISTAT)
      DIMENSION    LUNDES(9), ISTAT(9)
      CHARACTER    MODE*(*), NAME*(*)
      DIMENSION    NVMODE(4)
      CHARACTER    CHUSE*4
      LGN = LNBLNK (NAME)
      CHUSE = MODE
      CALL CUTOL (CHUSE)
      CALL VZEROI (NVMODE,4)
      IF (INDEX(CHUSE,'a').NE.0)  NVMODE(1) = 2
      IF (INDEX(CHUSE,'w').NE.0)  THEN
          IF (NVMODE(1).NE.0)      GO TO 91
          NVMODE(1) = 1
        ENDIF
      IF (INDEX(CHUSE,'r').NE.0)  THEN
          IF (NVMODE(1).NE.0)      GO TO 91
        ENDIF
      IF (INDEX(CHUSE,'+').NE.0)  NVMODE(2) = 1
      IF (INDEX(CHUSE,'l').NE.0)  NVMODE(3) = 1
      CALL CFOPEI (LUNDES,MEDIUM,NWREC,NVMODE,NBUF,NAME,ISTAT,LGN)
      RETURN
   91 LUNDES(1) = 0
      ISTAT(1)  = -1
      END

*-------------------------------------------------------------------------------

      INTEGER FUNCTION CFSTAT (NAME, INFO)
      CHARACTER*(*)  NAME
      INTEGER        INFO(12), CFSTATI
      LGN   = LNBLNK (NAME)
      CFSTAT = CFSTATI (NAME, INFO, LGN)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE UCOPY2 (A,B,N)
      DIMENSION A(*),B(*)
      IF (N.LT.2) GO TO 41
      IA = LOCFR (A)
      IB = LOCFR (B)
      IF (IA-IB) 20,99,10
   10 DO 15 I=1,N
   15 B(I) = A(I)
      RETURN
   20 DO 25 I=N,1,-1
   25 B(I) = A(I)
      RETURN
   41 IF (N.LE.0) RETURN
      B(1) = A(1)
   99 RETURN
      END

*-------------------------------------------------------------------------------
