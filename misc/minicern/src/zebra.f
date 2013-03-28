*-------------------------------------------------------------------------------
*
* This file contains the zebra's package subset needed to build h2root.
* It cannot be used by any zebra application because many zebra functionalities
* are missing.
*
*-------------------------------------------------------------------------------

      SUBROUTINE MZEBRA (LIST)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      CHARACTER  CQALLC*96
      COMMON /ZBCDCH/ CQALLC
                      CHARACTER*1  CQLETT(96), CQNUM(10)
                      EQUIVALENCE (CQLETT(1),CQALLC(1:1))
                      EQUIVALENCE (CQNUM(1), CQALLC(27:27))
      PARAMETER     (NQTCET=256)
      COMMON /ZCETA/ IQCETA(256),IQTCET(256)
      COMMON /ZHEADP/IQHEAD(20),IQDATE,IQTIME,IQPAGE,NQPAGE(4)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZNATUR/QPI2,QPI,QPIBY2,QPBYHR
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCL/  NQLN,NQLS,NQNIO,NQID,NQNL,NQNS,NQND,NQIOCH(16)
     +,              LQSUP,NQBIA, NQIOSV(3)
      COMMON /JZUC/  LQJZ,LQUP,LQDW,LQSV,LQAN, JQLEV,JQFLAG(10)
      COMMON/RZCOUNT/RZXIO(2)
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
      PARAMETER     (MAXFILES=128, MAXSTRIP=21)
      CHARACTER*128  RZNAMES(MAXFILES),RZSFILE(MAXSTRIP)
      COMMON/RZCSTRC/RZNAMES,RZSFILE
      COMMON/RZCSTRI/ISLAST,ISTRIP(MAXFILES),NSTRIP(MAXFILES),
     +                      NRSTRIP(MAXFILES)
      CHARACTER*4 CVERSN
      DIMENSION    LIST(9), INKEYS(3)
      DATA  INKEYS / 4HEBRA, 4HINIT, 4HDONE /
   12 NQSTOR = -1
      JQSTOR = -99
      CALL VZEROI (NQOFFT,66)
      CALL MZINCO (LIST)
      NQDCUT = 201
      NQWCUT = 500
      CALL UCOPYI (INKEYS,MQKEYS,3)
      CALL VZEROI (NQLN, 28)
      CALL VZEROI (LQJZ, 16)
      CALL VZEROI (NSTRIP, MAXFILES)
      JQLEV = -1
      RZXIO(1) = 0.
      RZXIO(2) = 0.
      IMODEH   = 0
      CALL VFILL (IQFENC,4,IQNIL)
      NQINIT = -1
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZINCO (LIST)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      CHARACTER  CQALLC*96
      COMMON /ZBCDCH/ CQALLC
                      CHARACTER*1  CQLETT(96), CQNUM(10)
                      EQUIVALENCE (CQLETT(1),CQALLC(1:1))
                      EQUIVALENCE (CQNUM(1), CQALLC(27:27))
      PARAMETER     (NQTCET=256)
      COMMON /ZCETA/ IQCETA(256),IQTCET(256)
      COMMON /ZHEADP/IQHEAD(20),IQDATE,IQTIME,IQPAGE,NQPAGE(4)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZNATUR/QPI2,QPI,QPIBY2,QPBYHR
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      COMMON /QUEST/ IQUEST(100)
      DIMENSION LIST(9)
      JBIT(IZW,IZP)     = IAND(ISHFT(IZW,-(IZP-1)),1)
      CALL VZEROI (IQUEST,100)
      CALL VZEROI (IQVID,18)
      CALL VZEROI (NQPHAS,15)
      NQBITW = IQBITW
      NQBITC = IQBITC
      NQCHAW = IQCHAW
      NQLNOR = 58
      NQLMAX = 58
      NQLPTH =  0
      NQRMAX = 132
      IQLPCT = IQBLAN
      IQNIL  = 16744448
      CQALLC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()$= ,.'
      CQALLC(65:90) = 'abcdefghijklmnopqrstuvwxyz'
      CQALLC(48:64) = '#''!:"_]&@?[>< ^;%'
      CQALLC(91:96) = '{|}~`?'
      CQALLC(61:61) = CHAR(92)
      CALL UCTOH1 (CQALLC, IQLETT, 96)
      CALL UCTOH1 (' 1234567890', IQNUM2, 11)
      CALL IZHNUM (IQLETT,NQHOLL,95)
      NQHOL0 = NQHOLL(45)
      CALL VFILL (IQCETA,NQTCET,96)
      DO 24  JC=95,1,-1
      JH = NQHOLL(JC)
   24 IQCETA(JH+1) = JC
      DO 26  JL=1,NQTCET
      J = IQCETA(JL)
      IF (J.GE.64)  THEN
        IF (J.GE.94)  THEN
          J = 57
        ELSEIF (J.EQ.93)  THEN
          J = 42
        ELSEIF (J.EQ.92)  THEN
          J = 40
        ELSEIF (J.EQ.91)  THEN
          J = 41
        ELSEIF (J.EQ.64)  THEN
          J = 51
        ELSE
          J = J - 64
        ENDIF
       ENDIF
   26 IQTCET(JL) = J
      QPI    = 4.*ATAN(1.)
      QPI2   = 2.*QPI
      QPIBY2 = QPI/2.
      QPBYHR = .0002998
      IQREAD = 2
      IQPRNT = 3
      IQPNCH = 4
      IQLOG  = IQPRNT
      IQTTIN = 5
      IQTYPE = 6
      ITYPE = IQTYPE
      IF (ITYPE.EQ.0) ITYPE = IQLOG
      NLIST = LIST(1)
      IF (NLIST) 32, 38, 33
   32 NLIST = -NLIST
      IF (JBIT(NLIST,2).NE.0)  NQLOGD = -2
      IF (JBIT(NLIST,1).NE.0)  IQLOG = ITYPE
      IQPRNT = IQLOG
      GO TO 38
   33 NQLOGD = LIST(2)
      IF (NLIST.EQ.1)              GO TO 38
      IF (LIST(3).NE.0) THEN
          IF (LIST(3).LT.0) THEN
              IQLOG = ITYPE
            ELSE
              IQLOG = LIST(3)
            ENDIF
        ENDIF
      IQPRNT = IQLOG
      IF (NLIST.EQ.2)              GO TO 38
      IF (LIST(4).NE.0) THEN
          IF (LIST(4).LT.0) THEN
              IQPRNT = ITYPE
            ELSE
              IQPRNT = LIST(4)
            ENDIF
        ENDIF
   38 IQPR2  = IQPRNT
      NQLOGM = NQLOGD
      IQDLUN = IQPRNT
      IQFLUN = IQPRNT
      IQHLUN = IQPRNT
      NQUSED = 0
      CALL VBLANK (IQHEAD,20)
      CALL VZEROI (IQDATE,7)
******CALL DATIME (IQDATE,IQTIME)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZPAW (NWORDS,CHOPT)
      COMMON /PAWC/  NWPAW,IXPAWC,IHBOOK,IXHIGZ,IXKU,IFENCE(5)
     +,              LMAIN, IPAW(4000000-11)
      CHARACTER    *(*) CHOPT
      CALL UOPTC (CHOPT,'M',IPAW)
      IF (IPAW(1).NE.0)   CALL MZEBRA(-1)
      NW = MAX (NWORDS,10000)
      CALL MZSTOR (IXPAWC,'/PAWC/',' ',IFENCE,LMAIN,IPAW(1),IPAW(1),
     +            IPAW(5000),IPAW(NW-11))
      NWPAW  = NW
      IHBOOK = 0
      IXHIGZ = 0
      IXKU   = 0
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZSTOR (IXSTOR,CHNAME,CHOPT
     +,                  IFENCE,LV,LLR,LLD,LIMIT,LAST)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      PARAMETER            (NQWKTT=2560)
      COMMON /MZCWK/ IQWKTB(NQWKTT), IQWKFZ(NQWKTT)
*
      DIMENSION    IXSTOR(9),IFENCE(9)
      DIMENSION    LV(9),LLR(9),LLD(9),LIMIT(9),LAST(9)
      DIMENSION    MMSYSL(5), NAMELA(2), NAMESY(2)
      CHARACTER    *(*) CHNAME,CHOPT
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HMZST, 4HOR   /
      DATA  MMSYSL / 4HSYSL,0,0,101,2/
      DATA  NAMELA / 4Hsyst, 4Hem   /
      DATA  NAMESY / 4Hsyst, 4Hem   /
      DATA  NAMWSP / 4Hqwsp /
      DATA  NAMEDV / 4HQDIV /
      MSBIT1 (IZW,IZP)   = IOR  (IZW,     ISHFT(1,IZP-1))
      IF (NQSTOR.NE.-1)            GO TO 13
      CALL VZEROI (NQOFFT,32)
      LQATAB = LOCF (IQTABV(1)) - 1
      LQASTO = LOCF (LQ(1)) - 1
      LQBTIS = LQATAB - LQASTO
      LQWKTB = LOCF(IQWKTB(1)) - LQASTO
      LQWKFZ = LOCF(IQWKFZ(1)) - LQASTO
      NQTSYS = LOCF(IQDN2(20)) - LQATAB
      NQWKTB = NQWKTT
      KQFT = 342
      IF (NQLOGD.GE.-1)
     +WRITE (IQLOG,9011) LQATAB,LQATAB
 9011 FORMAT (1X/' MZSTOR.  ZEBRA table base TAB(0) in /MZCC/ at adr'
     F,I12,1X,Z11,' HEX')
   13 CONTINUE
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      CALL UOPTC (CHOPT,'Q:',IQUEST)
      LOGQ   = IQUEST(1)
      IFLSPL = IQUEST(2)
      JQSTOR = NQSTOR + 1
      CALL VZEROI (KQT,27)
      LQSTOR = LOCF(LV(1)) - 1
      KQS    = LQSTOR - LQASTO
      NFEND  = (LQSTOR+1) - LOCF(IFENCE(1))
      NQFEND = NFEND
      NQSNAM(1) = IQBLAN
      NQSNAM(2) = IQBLAN
      N = MIN (8, LEN(CHNAME))
      IF (N.NE.0)  CALL UCTOH (CHNAME,NQSNAM,4,N)
      NQLOGL = NQLOGD
      IF (LOGQ.NE.0)  NQLOGL=-2
      NQSTRU = LOCF(LLR(1)) - (LQSTOR+1)
      NQREF  = LOCF(LLD(1)) - (LQSTOR+1)
      NQLINK = NQREF
      LQ2END = LOCF(LIMIT(1)) - LQSTOR
      NDATAT = LOCF(LAST(1))  - LQSTOR
      NDATA = NDATAT
      LOCT  = LQATAB
      IF (JQSTOR.NE.0)  THEN
          NDATA = NDATA  - NQTSYS
          NQSNAM(6) = NDATA
          LOCT  = LQSTOR + NDATA
          KQT   = LOCT   - LQATAB
          NDATA = NDATA - 4
          CALL VFILL (LQ(KQS+NDATA),10,IQNIL)
        ENDIF
      IF (NQLOGL.GE.-1)
     +WRITE (IQLOG,9021) JQSTOR,NQSNAM(1),NQSNAM(2)
     +,                  LQSTOR,LOCT,LQSTOR,LOCT,KQS,KQT,KQS,KQT
     +,                  NQSTRU,NQREF,LQ2END,NDATAT,NFEND
 9021 FORMAT (1X/' MZSTOR.  Initialize Store',I3,'  in ',2A4,
     F/10X,'with Store/Table at absolute adrs',2I12
     F/40X,'HEX',2(1X,Z11)/40X,'HEX',2(1X,Z11)
     F/30X,'relative adrs',2I12
     F/10X,'with',I6,' Str. in',I6,' Links in',I7,' Low words in'
     F,I8,' words.'
     F/10X,'This store has a fence of',I5,' words.')
      NSYS   =  400
      NQMINR =   40
      NWF    = 2000
      IF (JQSTOR.EQ.0)  NQMINR=164
      IF (NQSTRU.LT.0)               GO TO 91
      IF (NQREF .LT.NQSTRU)          GO TO 91
      IF (NDATAT.LT.NQLINK+NWF)      GO TO 91
      IF (LQ2END.LT.NQLINK+NQMINR)   GO TO 91
      IF (NFEND .LT.1)               GO TO 92
      IF (NFEND .GE.1001)            GO TO 92
      IF (IFLSPL.EQ.1)  THEN
          IF (JQSTOR.EQ.0)           GO TO 96
          GO TO 39
        ENDIF
      IF (JQSTOR.EQ.0)             GO TO 41
      KSA = KQS - NQFEND
      KSE = KQS + NDATAT
      DO 36  JSTO=1,JQSTOR
      JT  = NQOFFT(JSTO)
      JS  = NQOFFS(JSTO)
      JSA = JS  - IQTABV(JT+2)
      JSE = JS  + LQSTA(JT+21)
      JTA = JT  + LQBTIS
      JTE = JTA + NQTSYS
      IF (KSE.GT.JTA .AND. KSA.LT.JTE)    GO TO 94
      IF (KSE.GT.JSA .AND. KSA.LT.JSE)    GO TO 95
   36 CONTINUE
   39 IF (JQSTOR.GE.16)            GO TO 93
   41 NQOFFT(JQSTOR+1) = KQT
      NQOFFS(JQSTOR+1) = KQS
      NQALLO(JQSTOR+1) = IFLSPL
      CALL VZEROI (IQTABV(KQT+1),NQTSYS)
      CALL VBLANK (IQDN1(KQT+1), 40)
      NQSTOR = NQSTOR + 1
      LQ(KQS+NDATA-1) = IQNIL
      LQ(KQS+NDATA)   = IQNIL
      NDATA = NDATA - 2
      LQSTA(KQT+21) = NDATA
      JQDVLL = 2
      JQDVSY = 20
      LQSTA(KQT+20)  = NDATA
      LQEND(KQT+20)  = NDATA
      NQDMAX(KQT+20) = NDATA
      IQMODE(KQT+20) = 1
      IQKIND(KQT+20) = ISHFT (1, 23)
      IQRNO(KQT+20)  = 9437183
      IQDN1(KQT+20)  = NAMESY(1)
      IQDN2(KQT+20)  = NAMESY(2)
      LQSTA(KQT+2)  = NDATA - NSYS
      LQEND(KQT+2)  = LQSTA(KQT+2)
      NQDMAX(KQT+2) = NDATA
      IQMODE(KQT+2) = 1
      IQKIND(KQT+2) = MSBIT1 (2, 21)
      IQRCU(KQT+2)  = 3
      IQRTO(KQT+2)  = ISHFT (3,20)
      IQRNO(KQT+2)  = 9437183
      IQDN1(KQT+2)  = NAMEDV
      IQDN2(KQT+2)  = IQNUM(3)
      LQSTA(KQT+1)  = NQLINK + 1
      LQEND(KQT+1)  = LQSTA(KQT+1)
      NQDMAX(KQT+1) = NDATA
      IQKIND(KQT+1) = MSBIT1 (1, 21)
      IQRCU(KQT+1)  = 3
      IQRTO(KQT+1)  = ISHFT (3,20)
      IQRNO(KQT+1)  = 9437183
      IQDN1(KQT+1)  = NAMEDV
      IQDN2(KQT+1)  = IQNUM(2)
      CALL UCOPYI (IQCUR,IQTABV(KQT+1),16)
      CALL VFILL (IFENCE,NFEND,IQNIL)
      IF (NQLINK.NE.0)  CALL VZEROI (LV,NQLINK)
      IF (JQSTOR.EQ.0)  THEN
          IF (IXSTOR(1).EQ.0)      GO TO 71
        ENDIF
      IDN = ISHFT (JQSTOR,26)
      IXSTOR(1) = IDN
   71 JQDIVI = JQDVSY
      CALL MZLIFT (-7,LSYS,0,2,MMSYSL,0)
      LQSYSS(KQT+1) = LSYS
      NALL   = LOCF(IQTDUM(1)) - LOCF(LQSYSS(1))
      NSTR   = LOCF(LQSYSR(1)) - LOCF(LQSYSS(1))
      LOCAR  = LOCF (LQSYSS(KQT+1)) - LQSTOR
      LOCARE = LOCAR + NALL
      IQ(KQS+LSYS+1) = 11
      IQ(KQS+LSYS+2) = 1
      IQ(KQS+LSYS+3) = 1 + NQLINK
      IQ(KQS+LSYS+4) = NQSTRU
      IQ(KQS+LSYS+5) = NAMWSP
      IQ(KQS+LSYS+6) = IQBLAN
      IQ(KQS+LSYS+7) = LOCAR
      IQ(KQS+LSYS+8) = LOCARE
      IQ(KQS+LSYS+9) = NSTR
      IQ(KQS+LSYS+10)= NAMELA(1)
      IQ(KQS+LSYS+11)= NAMELA(2)
      IQTABV(KQT+13) = MIN (1, LOCAR)
      IQTABV(KQT+14) = MAX (LQSTA(KQT+21), LOCARE)
  999 NQTRAC = NQTRAC - 2
      RETURN
   95 NQCASE = 1
   94 NQCASE = NQCASE - 2
      NQFATA = 3
      IQUEST(20) = JSTO - 1
      IQUEST(21) = NQPNAM(JT+1)
      IQUEST(22) = NQPNAM(JT+2)
   96 NQCASE = NQCASE + 3
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 9
      IQUEST(11) = NQSNAM(1)
      IQUEST(12) = NQSNAM(2)
      IQUEST(13) = NFEND
      IQUEST(14) = NQSTRU
      IQUEST(15) = NQLINK
      IQUEST(16) = LQ2END
      IQUEST(17) = NDATAT
      IQUEST(18) = NQMINR
      IQUEST(19) = NWF
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZOPEN(LUNIN,CHDIR,CFNAME,CHOPTT,LRECL,ISTAT)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      COMMON/RZCKEY/IHEAD(3),KEY(100),KEY2(100),KEYDUM(50)
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
      COMMON /QUEST/ IQUEST(100)
      COMMON /RZBUFF/ ITEST(8704)
      PARAMETER     (MAXFILES=128, MAXSTRIP=21)
      CHARACTER*128  RZNAMES(MAXFILES),RZSFILE(MAXSTRIP)
      COMMON/RZCSTRC/RZNAMES,RZSFILE
      COMMON/RZCSTRI/ISLAST,ISTRIP(MAXFILES),NSTRIP(MAXFILES),
     +                      NRSTRIP(MAXFILES)
      integer cfstat,statf,info(12)
      CHARACTER*(*) CFNAME,CHDIR,CHOPTT
      CHARACTER*9   SPACES
      CHARACTER*8   STAT
      CHARACTER*36  CHOPT
      CHARACTER*255 CHFILE
      LOGICAL       IEXIST
      CHARACTER*4   CHOPE
      PARAMETER (NWORD = 8704)
      PARAMETER (IBYTES=4)
      CHOPT=CHOPTT
      CALL CLTOU(CHOPT)
      IOPT1=INDEX(CHOPT,'1')
      IOPTC=INDEX(CHOPT,'C')
      IOPTL=INDEX(CHOPT,'L')
      IOPTN=INDEX(CHOPT,'N')
      IOPTS=INDEX(CHOPT,'S')
      IOPTP=INDEX(CHOPT,'P')
      IOPTU=INDEX(CHOPT,'U')
      IOPTV=INDEX(CHOPT,'V')
      IOPTW=INDEX(CHOPT,'W')
      IOPTX=INDEX(CHOPT,'X')
      IOPTY=INDEX(CHOPT,'Y')
      LUNIT=LUNIN
      IQUEST(10) = 0
      IQUEST(11) = 0
      IQUEST(12) = 0
      IMODEC = IOPTC
      IMODEX = IOPTX
      LRECI = LRECL
      LRECL2 = 0
      IMODEH = 0
      CHFILE=CFNAME
      IF(IOPTP.EQ.0)CALL CUTOL(CHFILE)
      IPASS=0
   10 CONTINUE
      IF(IOPTN.NE.0)THEN
         STAT='UNKNOWN'
         IF(LRECI.LE.0) THEN
            WRITE(IQPRNT,10000) LRECI
10000 FORMAT(' RZOPEN. - invalid record length: ',I6)
            ISTAT = 1
            GOTO 70
         ELSEIF(LRECI.GE.8191) THEN
            WRITE(IQPRNT,10100) LRECI
10100 FORMAT(' RZOPEN. record length:',I6,
     + ' > maximum safe value (8191 words).')
            IF(LRECI.GT.8192) WRITE(IQPRNT,10200)
10200 FORMAT(' RZOPEN. Automatic record length determination will not',
     + ' work with this file.')
            WRITE(IQPRNT,10300)
10300 FORMAT(' RZOPEN. You may have problems transferring your',
     + ' file to other systems ',/,
     + '         or writing it to tape.')
         ENDIF
      ELSE
         STAT='OLD'
         LENF = LENOCC(CHFILE)
         IF(IOPTC.EQ.0) THEN
            INQUIRE(FILE=CHFILE,EXIST=IEXIST)
            ISTATF = 0
         ELSE
            IEXIST = CFSTAT(CHFILE(1:LENF),INFO).EQ.0
            ISTATF = 1
         ENDIF
         IF(.NOT.IEXIST) THEN
            WRITE(IQPRNT,*) 'RZOPEN. Error - input file ',
     + CHFILE(1:LENF),' does not exist'
            ISTAT = 2
            GOTO 70
         ENDIF
         IF(LRECL.EQ.0) THEN
            ICOUNT = NWORD
            IF(IOPTC.EQ.0) THEN
            ENDIF
   20 NREAD = ICOUNT
            IF(IOPTC.EQ.0) THEN
               OPEN(LUNIT,FILE=CHFILE,FORM='UNFORMATTED',STATUS='OLD',
     +         RECL=IBYTES*NREAD,ACCESS='DIRECT',IOSTAT=ISTAT)
               IF(ISTAT.NE.0)GOTO 60
               READ(LUNIT,REC=1,IOSTAT=IOS) (ITEST(JW),JW=1,NREAD)
               IF(IOS.NE.0) THEN
                  CLOSE(LUNIT)
                  ICOUNT = ICOUNT * .75
                  GOTO 20
               ENDIF
            ELSE
               CALL CFOPEN(LUNPTR,0,NREAD,'r',0,CHFILE,IOS)
               IF (IOS .NE. 0) THEN
                  ISTAT = -1
                  GOTO 70
               ENDIF
               NWTAK = NREAD
               CALL CFGET(LUNPTR,0,NREAD,NWTAK,ITEST,IOS)
               IF(IOS.NE.0) THEN
                  CALL CFCLOS(LUNPTR,0)
                  ICOUNT = ICOUNT * .75
                  GOTO 20
               ENDIF
            ENDIF
            IF(IOPTX.NE.0) CALL VXINVB(ITEST(1),NREAD)
            DO 30 J=1, NWORD
               IF(ITEST(J+25).GT.0.AND.ITEST(J+25).LE.J) THEN
                  IF((J+ITEST(J+25)).GT.8703) GOTO 30
                  LRC = ITEST(J+ITEST(J+25)+1)
                  IF(LRC.EQ.J) THEN
                     LE = ITEST(J+30)
                     LD = ITEST(J+24)
                     NRD = ITEST(J+LD)
                     IF(NRD*LRC.NE.LE) GOTO 30
                     LRECL = J
                     IF(IOPTC.EQ.0) THEN
                        CLOSE(LUNIT)
                     ELSE
                        CALL CFCLOS(LUNPTR,0)
                     ENDIF
                     GOTO 40
                  ENDIF
               ENDIF
   30 CONTINUE
            IF(IOPTC.EQ.0) THEN
               CLOSE(LUNIT)
            ELSE
               CALL CFCLOS(LUNPTR,0)
            ENDIF
            IF(IOPTX.EQ.0.AND.IPASS.EQ.0) THEN
               WRITE(IQPRNT,10400)
10400 FORMAT(' RZOPEN. Cannot determine record length - ',
     + ' EXCHANGE mode is used.')
               IOPTX = 1
               IMODEX = 1
               IPASS = 1
               GOTO 10
            ENDIF
            WRITE(IQPRNT,*) ' RZOPEN. Error in the input file'
            ISTAT = 3
            GOTO 70
         ENDIF
      ENDIF
   40 CONTINUE
      IF(IOPTC.EQ.0) THEN
      NBYTES = IBYTES
      OPEN(UNIT=LUNIT,FILE=CHFILE,FORM='UNFORMATTED',
     + RECL=NBYTES*LRECL,ACCESS='DIRECT',STATUS=STAT,IOSTAT=ISTAT)
      ELSE
         CHOPE = 'r'
         IF(IOPTU.NE.0.OR.IOPT1.NE.0) CHOPE = 'r+'
         IF(IOPTN.NE.0) CHOPE = 'w+'
         JRECL = LRECL
         CALL CFOPEN(LUNPTR,0,JRECL,CHOPE,0,CHFILE,ISTAT)
         LUNIT = 1000 + LUNPTR
      ENDIF
      IF(ISTAT.NE.0)GOTO 60
      IF(IOPTY.NE.0)GOTO 50
      IF(IOPTN.EQ.0.AND.IPASS.EQ.0.AND.ISTAT.EQ.0)THEN
         IMODEX=IOPTX
         IZRECL=LRECL
         CALL RZIODO(LUNIT,50,2,ITEST,1)
         CALL VXINVB(ITEST(9),1)
         IF(JBIT(ITEST(9),12).NE.0)THEN
            IMODEX=1
            CALL RZIODO(LUNIT,50,2,ITEST,1)
         ELSE
            CALL VXINVB(ITEST(9),1)
         ENDIF
         LB=ITEST(25)
         IF(LB.GT.8187) THEN
            WRITE(IQPRNT,10500) CHFILE(1:LENOCC(CHFILE))
10500 FORMAT(' RZOPEN: cannot determine record length.',
     + ' File ',A,' probably not in RZ format')
            LRECP=-1
            ISTAT=2
            IF(IOPTC.EQ.0) THEN
               CLOSE(LUNIT)
            ELSE
               CALL CFCLOS(LUNIT-1000,0)
            ENDIF
            GOTO 70
         ENDIF
         IF(LB.GT.48) CALL RZIODO(LUNIT,LB+6,2,ITEST,1)
         LRECP=ITEST(LB+1)
         IQUEST(1)=0
         IF(LRECP.NE.LRECL)THEN
            LRECL2=LRECL
            LRECL=0
            IF(IOPTC.EQ.0) THEN
               CLOSE(LUNIT)
            ELSE
               CALL CFCLOS(LUNIT-1000,0)
            ENDIF
            IF(IPASS.EQ.0) THEN
               IPASS=1
               GOTO 10
            ELSE
               WRITE(IQPRNT,*) 'Cannot determine record length'
               ISTAT = 1
               GOTO 70
            ENDIF
         ENDIF
      ENDIF
      IF (IPASS.NE.0 .AND. LRECL2.NE.0) THEN
         WRITE(IQPRNT,10600) LRECL2,LRECL
10600 FORMAT(' RZOPEN:  LRECL inconsistent - ',
     + ' file was opened with LRECL = ',I6,
     + ' should be LRECL = ',I6)
      ENDIF
   50 IF(IOPTW.NE.0)THEN
         IF (IOPTC .EQ. 0) THEN
            LUN = LUNIT
         ELSE
            LUN = LUNIT - 1000
         ENDIF
         IF(LUN.LT.10)WRITE(CHDIR,10700)LUN
         IF(LUN.GE.10)WRITE(CHDIR,10800)LUN
10700 FORMAT('LUN',I1,'    ')
10800 FORMAT('LUN',I2,'   ')
      ENDIF
   60 CONTINUE
      IQUEST(10) = LRECL
      IQUEST(11) = LUNIT
      IQUEST(12) = IMODEX
   70 CONTINUE
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZIODO(LUNRZ,JREC,IREC1,IBUF,IRW)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      COMMON /RZBUFF/ ITEST(8704)
      COMMON/RZCOUNT/RZXIO(2)
      PARAMETER (MAXFILES=128, MAXSTRIP=21)
      CHARACTER*128 RZNAMES(MAXFILES),RZSFILE(MAXSTRIP)
      COMMON/RZCSTRC/RZNAMES,RZSFILE
      COMMON/RZCSTRI/ISLAST,ISTRIP(MAXFILES),NSTRIP(MAXFILES),
     + NRSTRIP(MAXFILES)
      DIMENSION IBUF(JREC)
      PARAMETER (MEDIUM=0)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      RZXIO(IRW) = RZXIO(IRW) + JREC
      IREC=IREC1
      IF(LUNRZ.GT.0)THEN
         NERR=0
         IF(IMODEH.NE.0) THEN
            IQUEST(1) = JBYT(IQ(KQSP+LTOP),7,7)
            IQUEST(2) = JREC
            IQUEST(4) = IREC
            IOWAY = IRW - 1
            IF(IRW.EQ.2.AND.IMODEX.NE.0) CALL VXINVB(IBUF,JREC)
            print*,'>>>>>> CALL JUMPST(LUNRZ)'
******      CALL JUMPST(LUNRZ)
            print*,'>>>>>> CALL JUMPX2(IBUF,IOWAY)'
******      CALL JUMPX2(IBUF,IOWAY)
            IF(IQUEST(1).NE.0) IQUEST(1) = 100 + IRW
            IF(IMODEX.NE.0) CALL VXINVB(IBUF,JREC)
         ELSE
   10 IF (IRW.EQ.1)THEN
               IF(IMODEC.EQ.0) THEN
                  READ (UNIT=LUNRZ,REC=IREC,ERR=20,IOSTAT=ISTAT)IBUF
               ELSE
                     CALL CFSEEK(LUNRZ-1000,MEDIUM,IZRECL,IREC-1,ISTAT)
                     NWTAK = JREC
                     CALL CFGET(LUNRZ-1000,MEDIUM,JREC,NWTAK,IBUF,
     + ISTAT)
                     IF(ISTAT.NE.0) GOTO 20
               ENDIF
               IF(IMODEX.NE.0) CALL VXINVB(IBUF,JREC)
            ELSE
               IF(IMODEX.NE.0) CALL VXINVB(IBUF,JREC)
               IF(IMODEC.EQ.0) THEN
                  WRITE(UNIT=LUNRZ,REC=IREC,ERR=20,IOSTAT=ISTAT)IBUF
               ELSE
                  CALL CFSEEK(LUNRZ-1000,MEDIUM,IZRECL,IREC-1,ISTAT)
                  IF(ISTAT.NE.0) GOTO 20
                  print*,'>>>>>> CALL CFPUT()'
******            CALL CFPUT(LUNRZ-1000,MEDIUM,JREC,IBUF,ISTAT)
                  IF(ISTAT.NE.0) GOTO 20
               ENDIF
               IF(IMODEX.NE.0) CALL VXINVB(IBUF,JREC)
            ENDIF
            RETURN
   20 NERR=NERR+1
            IF(NERR.LT.100)GO TO 10
            IQUEST(1)=100+IRW
            WRITE(IQLOG,1000)IREC,LUNRZ,ISTAT
 1000 FORMAT(' RZIODO. Error at record =',I5,' LUN =',I6,
     + ' IOSTAT =',I6)
         ENDIF
      ELSE
         KOF=IQ(KQSP+LRZ0-2*LUNRZ-1)+IQ(KQSP+LRZ0-2*LUNRZ)*(IREC-1)
         IF (IRW.EQ.1)THEN
            CALL UCOPYI(IQ(KOF),IBUF,JREC)
         ELSE
            CALL UCOPYI(IBUF,IQ(KOF),JREC)
         ENDIF
      ENDIF
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZVCYC(LTAD)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     +           KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     +           KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     +           KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     +           KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      INTEGER        KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     +               KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON/RZCYCLE/KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     +               KCNCYC, KNWCYC, KKYCYC, KVSCYC
      IF (LTAD.EQ.0) GO TO 99
      IF (IQ(KQSP+LTAD+KRZVER).EQ.0) THEN
         KLCYCL = 4
         KPPCYC = 0
         KFRCYC = 2
         KSRCYC = 0
         KFLCYC = 1
         KORCYC = 2
         KCNCYC = 3
         KNWCYC = 3
         KKYCYC =-1
         KVSCYC = 0
      ELSE
         KLCYCL = 7
         KPPCYC = 0
         KFRCYC = 2
         KSRCYC = 5
         KFLCYC = 1
         KORCYC = 3
         KCNCYC = 3
         KNWCYC = 4
         KKYCYC = 6
         KVSCYC = 1
      ENDIF
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZIN(IXDIV,LSUP,JBIAS,KEYU,ICYCLE,CHOPT)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     +           KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     +           KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     +           KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     +           KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      CHARACTER*(*) CHOPT
      DIMENSION KEYU(*)
      DIMENSION LSUP(1),JBIAS(1),IQK(10),IQKS(10)
      EQUIVALENCE (IOPTA,IQUEST(91)), (IOPTC,IQUEST(92))
     +,      (IOPTD,IQUEST(93)), (IOPTN,IQUEST(94)), (IOPTR,IQUEST(95))
     +,      (IOPTS,IQUEST(96))
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      LRIN=LQ(KQSP+LTOP-7)
      IF(LRIN.EQ.0)THEN
         CALL MZBOOK(JQPDVS,LRIN,LTOP,-7,'RZIN',0,0,LREC+1,2,-1)
         IQ(KQSP+LRIN-5)=IQ(KQSP+LTOP-5)
         IQ(KQSP+LTOP+KIRIN)=0
      ENDIF
      CALL RZINK(KEYU,ICYCLE,CHOPT)
      IF(IQUEST(1).NE.0)GO TO 99
      IF(IOPTC.NE.0.AND.IOPTD.EQ.0)GO TO 99
      IDTIME=IQUEST(14)
      IDNW  =IQUEST(12)
      IF(IOPTS.NE.0)CALL UCOPYI(IQUEST(20),IQKS,10)
      IF(IOPTN.NE.0)THEN
         IF(IOPTD.EQ.0)GO TO 99
         CALL UCOPYI(IQUEST(41),IQK,10)
      ENDIF
      LBANK=0
      IF(LSUP(1).NE.0)THEN
         CALL MZSDIV(IXDIV,1)
         IF(JBIAS(1).LE.0)LBANK=LQ(KQS+LSUP(1)+JBIAS(1))
         IF(JBIAS(1).GT.0)LBANK=LSUP(1)
      ENDIF
      IFORM=JBYT(IQUEST(14),1,3)
      IF(IFORM.EQ.0)THEN
         CALL RZINS(IXDIV,LSUP,JBIAS,LBANK)
      ELSE
         NDATA=IQUEST(12)
         IF(LBANK.NE.0)THEN
            IF(NDATA.LE.IQ(KQS+LBANK-1))THEN
               CALL RZREAD(IQ(KQS+LBANK+1),NDATA,1,IFORM)
               IQUEST(11) = LBANK
            ELSE
               IQUEST(1)=3
            ENDIF
         ELSE
            CALL MZBOOK(IXDIV,LFROM,LSUP,JBIAS,'RZIN',0,0,NDATA,
     +                  IFORM,-1)
            CALL RZREAD(IQ(KQS+LFROM+1),NDATA,1,IFORM)
            IQUEST(11) = LFROM
         ENDIF
      ENDIF
      IQUEST(14)=IDTIME
      IQUEST(12)=IDNW
      IF(IOPTN.NE.0)CALL UCOPYI(IQK ,IQUEST(41),10)
      IF(IOPTS.NE.0)CALL UCOPYI(IQKS,IQUEST(20),10)
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZINS(IXDIVP,LSUPP,JBIASP,LBANK)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/  MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +,              MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +,              IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +,              LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +,                         LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +,              LQMTC1,LQMTC2, NQFRTC,NQLIVE
      COMMON /MZIOC/ NWFOAV,NWFOTT,NWFODN,NWFORE,IFOCON(3)
     +,              MFOSAV(2),  JFOEND,JFOREP,JFOCUR,MFO(200)
      COMMON /MZCN/  IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /FZCI/  LUNI,LUNNI,IXDIVI,LTEMPI,IEVFLI
     +,              MSTATI,MEDIUI,IFIFOI,IDAFOI,IACMOI,IUPAKI
     +,              IADOPI,IACTVI,INCBPI,LOGLVI,MAXREI,  ISTENI
     +,              LBPARI, L4STOI,L4STAI,L4CURI,L4ENDI
     +,              IFLAGI,NFASTI,N4SKII,N4RESI,N4DONI,N4ENDI
     +,              IOPTIE,IOPTIR,IOPTIS,IOPTIA,IOPTIT,IOPTID
     +,                     IOPTIF,IOPTIG,IOPTIH,IOPTI2(4)
     +,              IDI(2),IPILI(4),NWTXI,NWSEGI,NWTABI,NWBKI,LENTRI
     +,              NWUHCI,IOCHI(16),NWUMXI,NWUHI,NWIOI
     +,              NWRDAI,NRECAI,LUHEAI,JRETCD,JERROR,NWERR
      PARAMETER      (JAUIOC=50, JAUSEG=68, JAUEAR=130)
      COMMON /FZCSEG/NQSEG,IQSEGH(2,20),IQSEGD(20),IQSGLU,IQSGWK
      COMMON /FZCOCC/NQOCC,IQOCDV(20),IQOCSP(20)
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
      DIMENSION    IXDIVP(9),LSUPP(9),JBIASP(9),IDUM(3)
      EQUIVALENCE (IOPTR,IQUEST(95))
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IXDIVI = IXDIVP(1)
      JRETCD = 0
      JERROR = 0
      NQOCC  = 0
      NQSEG  = 0
      CALL RZREAD(NWTABI,3,1,1)
      IF(IQUEST(1).NE.0) GO TO 99
      NWIOI  = 0
      NWUHI  = 0
      NWUHCI = 0
      NWUMXI = 0
      NWTXI  = 0
      NWSEGI = 0
      CALL MZSDIV(IXDIVI,7)
      IF(JQDIVI.EQ.0) JQDIVI=2
      IF(LBANK.NE.0.AND.IOPTR.NE.0)THEN
         NLINK=IQ(KQS+LBANK-3)
         NWK  =NWBKI-10-NLINK
         IF(IQ(KQS+LBANK-1).LT.NWK)GO TO 91
         IF(IMODEX.GT.0)THEN
            CALL MZIOCR(LQ(KQS+LBANK-NLINK-1))
            IF(IQUEST(1).LT.0)GO TO 99
            IQUEST(1)=0
         ENDIF
         CALL RZREAD(IQ(KQS+LBANK+1),NWK,NWTABI+14+NLINK,0)
         GO TO 99
      ENDIF
      LQSYSR(KQT+1)=LSUPP(1)
      CALL FZIMTB
      IF(JRETCD.EQ.3)GO TO 91
      CALL RZREAD(LQ(LQTA+NWTABI),NWTABI,4,1)
      IF(IQUEST(1).NE.0) GO TO 99
      LSTA = LQ(LQMTA+3)
      LEND = LQ(LQMTA+4)
      IF(IMODEX.LE.0)GO TO 30
      LIN = LSTA
      NWR = NWTABI+4
   10 CONTINUE
      CALL RZREAD(LQ(KQS+LIN),1,NWR,1)
      IF(IQUEST(1).NE.0)GO TO 99
      NWR   = NWR+1
      IWD   = LQ(KQS+LIN)
      NST   = JBYT(IWD,1,16)-12
      IF(NST.LT.0)GO TO 20
      IQLN = LIN
      IQLS = LIN + NST + 1
      IF(IQLS+8.GE.LEND)GO TO 92
      MFO(1) =  1
      MFO(2) =  NST + 2
      MFO(3) =  2
      MFO(4) =  2
      MFO(5) =  5
      MFO(6) =  1
      MFO(7) =  1
      MFO(8) = -1
      JFOEND =  8
      NWFOTT = NST+9
      NWFODN = 0
      CALL RZREAD(LQ(KQS+LIN+1),NST+9,NWR,0)
      IF(IQUEST(1).NE.0)GO TO 99
      NWR    = NWR+NST+9
      IQNIO  = JBYT(IQ(KQS+IQLS),19,4)
      IQNL   = IQ(KQS+IQLS-3)
      IQND   = IQ(KQS+IQLS-1)
      IF(IQNIO+IQNL.NE.NST)GO TO 92
      LIN    = IQLS + IQND + 9
      IF(IQND.GT.0)THEN
         IF(LIN.GT.LEND)GO TO 92
         CALL MZIOCR(LQ(KQS+IQLN))
         IF(IQUEST(1).LT.0)GO TO 99
         IQUEST(1)=0
         CALL RZREAD(IQ(KQS+IQLS+1),IQND,NWR,0)
         IF(IQUEST(1).NE.0)GO TO 99
         NWR = NWR+IQND
      ENDIF
      IF(LIN.LT.LEND)GO TO 10
      GO TO 40
   20 NWD = JBYT(IWD,17,IQDROP-17)
      IF(NWD.EQ.0.OR.NWD.NE.NST+12)GO TO 92
      IF(JBYT(IWD,IQDROP,IQBITW-IQDROP).NE.1)GO TO 92
      LIN = LIN + NWD
      IF(LIN.LT.LEND)GO TO 10
      GO TO 40
   30 NWR = LEND - LSTA
      CALL RZREAD(LQ(KQS+LSTA),NWR,NWTABI+4,0)
      IF(IQUEST(1).NE.0) GO TO 99
   40 CONTINUE
      CALL FZIREL
      IF(JRETCD.NE.0)GO TO 93
      JB=JBIASP(1)
      IF(JB.GE.2)THEN
         LSUPP(1)=LENTRI
      ELSE
         LSUPP(1)=LQSYSR(KQT+1)
         CALL ZSHUNT(IXDIVI,LENTRI,LSUPP,JB,1)
      ENDIF
      IQUEST(1)  = 0
      IQUEST(11) = IEVFLI
      IQUEST(12) = 0
      IQUEST(13) = LENTRI
      IQUEST(14) = NWBKI
      GO TO 99
   91 IQUEST(11)= -2
      IQUEST(1) =  1
      GO TO 99
   92 IQUEST(11)= -3
      IQUEST(1) =  1
      GO TO 99
   93 IQUEST(11)= -3
      IQUEST(1) =  1
   99 RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE ZITOH (INTV,IHOLL,NP)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      COMMON /SLATE/ DUMMY(8), MM(4), DUMB(28)
      DIMENSION    INTV(99), IHOLL(99), NP(9)
      DIMENSION    MPAK(2)
      DATA  MPAK   /6,4/
      N = NP(1)
      DO 39  JW=1,N
      CALL UPKBYT (INTV(JW),1,MM(1),4,MPAK(1))
      DO 16  J=1,4
      JV = MM(J)
      IF (JV.EQ.0)  JV=45
   16 MM(J) = IQLETT(JV)
      CALL UBUNCH (MM(1),IHOLL(JW),4)
   39 CONTINUE
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZRESV
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      JBIT(IZW,IZP)     = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      JQDIVR = JQDIVI
      IF (JQDIVR.LT.3)             GO TO 41
      JQMODE = JBIT (IQMODE(KQT+JQDIVR), 1)
      JQKIND = JBYT (IQKIND(KQT+JQDIVR),21,4)
      IF (JQMODE.NE.0)             GO TO 31
      JQDIVN = JQDIVR + 1
      IF (JQDIVR.EQ.JQDVLL)        GO TO 24
      IF (JQDIVR.EQ.20)            GO TO 25
      IF (JBYT(IQKIND(KQT+JQDIVN),21,4).NE.JQKIND)  GO TO 25
      IF (JBIT(IQMODE(KQT+JQDIVN),1)   .EQ.JQMODE)  GO TO 25
      JQSHAR = JQDIVN
      JQSHR1 = JQDIVR
      JQSHR2 = JQDIVN
      NQRESV = LQSTA(KQT+JQDIVN) - LQEND(KQT+JQDIVR)
      RETURN
   24 JQDIVN = JQDVSY
   25 L      = MIN (LQSTA(KQT+JQDIVR)+NQDMAX(KQT+JQDIVR),
     +               LQSTA(KQT+JQDIVN) )
      NQRESV = L - LQEND(KQT+JQDIVR)
      JQSHAR = 0
      RETURN
   31 JQDIVN = JQDIVR - 1
      IF (JQDIVR.EQ.JQDVSY)        GO TO 34
      IF (JBYT(IQKIND(KQT+JQDIVN),21,4).NE.JQKIND)  GO TO 35
      IF (JBIT(IQMODE(KQT+JQDIVN),1)   .EQ.JQMODE)  GO TO 35
      JQSHAR = JQDIVN
      JQSHR1 = JQDIVN
      JQSHR2 = JQDIVR
      NQRESV = LQSTA(KQT+JQDIVR) - LQEND(KQT+JQDIVN)
      RETURN
   34 JQDIVN = JQDVLL
   35 L      = MAX (LQEND(KQT+JQDIVR)-NQDMAX(KQT+JQDIVR),
     +               LQEND(KQT+JQDIVN) )
      NQRESV = LQSTA(KQT+JQDIVR) - L
      JQSHAR = 0
      RETURN
   41 JQKIND = 1
      JQSHR1 = 1
      JQSHR2 = 2
      NQRESV = LQSTA(KQT+2) - LQEND(KQT+1) - NQMINR
      IF (JQDIVR.EQ.1)             GO TO 44
      JQMODE = 1
      JQDIVN = 1
      JQSHAR = 1
      RETURN
   44 JQMODE = 0
      JQDIVN = 2
      JQSHAR = 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZSAVE
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     +           KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     +           KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     +           KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     +           KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      JBIT(IZW,IZP)     = IAND(ISHFT(IZW,-(IZP-1)),1)
      IF(LQRS.EQ.0)GO TO 99
      IF(LTOP.EQ.0)GO TO 99
      IF(JBIT(IQ(KQSP+LTOP),2).NE.0)THEN
         IF(ISAVE.NE.2)THEN
            IDTIME=0
            CALL RZDATE(IDTIME,IDATE,ITIME,2)
            IQ(KQSP+LTOP+KDATEM)=IDTIME
         ENDIF
         LUNC= IQ(KQSP+LTOP-5)
         LB  = IQ(KQSP+LTOP+KLB)
         LREK= IQ(KQSP+LTOP+LB+1)
         LUS = LQ(KQSP+LTOP-3)
         IF(LUS.NE.0)THEN
            NUSED=IQ(KQSP+LUS+1)
            IF(NUSED.GT.0)THEN
               DO 40 I=1,NUSED
                  IR1=IQ(KQSP+LUS+2*(I-1)+2)
                  IRL=IQ(KQSP+LUS+2*(I-1)+3)
                  DO 30 J=IR1,IRL
                     IWORD = (J-1)/32 + 1
                     IBIT  = J-32*(IWORD-1)
                     CALL SBIT1(IQ(KQSP+LTOP+LB+2+IWORD),IBIT)
  30              CONTINUE
  40           CONTINUE
               IQ(KQSP+LUS+1)=0
            ENDIF
         ENDIF
         LPU = LQ(KQSP+LTOP-5)
         IF(LPU.NE.0)THEN
            NPURG=IQ(KQSP+LPU+1)
            IF(NPURG.GT.0)THEN
               DO 60 I=1,NPURG
                  IR1=IQ(KQSP+LPU+2*(I-1)+2)
                  IRL=IQ(KQSP+LPU+2*(I-1)+3)
                  DO 50 J=IR1,IRL
                     IWORD = (J-1)/32 + 1
                     IBIT  = J-32*(IWORD-1)
                     CALL SBIT0(IQ(KQSP+LTOP+LB+2+IWORD),IBIT)
  50              CONTINUE
  60           CONTINUE
               IQ(KQSP+LPU+1)=0
            ENDIF
         ENDIF
         LROUT=LQ(KQSP+LTOP-6)
         IF(LROUT.NE.0)THEN
            IROUT=IQ(KQSP+LTOP+KIROUT)
            IF(IROUT.NE.0)THEN
               CALL RZIODO(LUNC,LREK,IROUT,IQ(KQSP+LROUT+1),2)
               IF(IQUEST(1).NE.0)GO TO 99
            ENDIF
         ENDIF
         LDS =IQ(KQSP+LTOP+KLD)
         NRD =IQ(KQSP+LTOP+LDS)
         IF(ISAVE.NE.2)THEN
            IF(LTOP.EQ.LCDIR)IQ(KQSP+LTOP+KDATEM)=IDTIME
         ENDIF
         CALL SBIT0(IQ(KQSP+LTOP),2)
         DO 70 J=NRD,1,-1
            IREC=IQ(KQSP+LTOP+LDS+J)
            L=(J-1)*LREK+1
            CALL RZIODO(LUNC,LREK,IREC,IQ(KQSP+LTOP+L),2)
            IF(IQUEST(1).NE.0)THEN
               CALL SBIT1(IQ(KQSP+LTOP),2)
               GO TO 99
            ENDIF
  70     CONTINUE
         IF(LCDIR.EQ.0.OR.LTOP.EQ.LCDIR)GO TO 99
         IF(JBIT(IQ(KQSP+LCDIR),2).NE.0)THEN
            LDS =IQ(KQSP+LCDIR+KLD)
            NRD =IQ(KQSP+LCDIR+LDS)
            IF(ISAVE.NE.2)THEN
               IQ(KQSP+LCDIR+KDATEM)=IDTIME
            ENDIF
            CALL SBIT0(IQ(KQSP+LCDIR),2)
            DO 80 J=NRD,1,-1
               IREC=IQ(KQSP+LCDIR+LDS+J)
               L=(J-1)*LREK+1
               CALL RZIODO(LUNC,LREK,IREC,IQ(KQSP+LCDIR+L),2)
               IF(IQUEST(1).NE.0)THEN
                  CALL SBIT1(IQ(KQSP+LCDIR),2)
                  GO TO 99
               ENDIF
  80        CONTINUE
         ENDIF
      ENDIF
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE FZICV (MS,IRMT)
      COMMON /QUEST/ IQUEST(100)
      COMMON /MZIOC/ NWFOAV,NWFOTT,NWFODN,NWFORE,IFOCON(3)
     +,              MFOSAV(2),  JFOEND,JFOREP,JFOCUR,MFO(200)
      DIMENSION    MS(99), IRMT(99)
      DOUBLE PRECISION   THDB
      DIMENSION    THIS(2)
      EQUIVALENCE  (THDB,THIS)
      EQUIVALENCE  (ITHA,THA,THIS(1)), (ITHB,THB,THIS(2))

      JMS = 0
      IF (NWFODN.NE.0) GO TO 30
      NWFORE = NWFOTT
      JMSEX = MIN (NWFORE,NWFOAV)
      JMT = 0
      JFOCUR = 0
      IFOCON(1) = 0
   21 ITYPE = MFO(JFOCUR+1)
      IF (ITYPE.EQ.7) GO TO 24
      NWSEC = MFO(JFOCUR+2)
      IF (NWSEC) 22, 23, 31
   22 NWSEC = NWFORE
      GO TO 31
   23 IWORD = MS(JMS+1)
      NWSEC = IWORD
      GO TO 25
   24 IWORD = MS(JMS+1)
      ITYPE = MOD (IWORD,16)
      NWSEC = IWORD/16
   25 IRMT(JMT+1) = IWORD
      JMS = JMS + 1
      JMT = JMT + 1
      NWFORE = NWFORE - 1
      IF (ITYPE.GE.8) GO TO 27
      IF (NWSEC.EQ.0) GO TO 29
      IF (NWSEC.GT.0) GO TO 31
   27 IFOCON(1) = -1
      IFOCON(2) = JMT
      IFOCON(3) = IWORD
   29 ITYPE = 0
      NWSEC = NWFORE
      GO TO 31
   30 JMSEX = MIN (NWFORE,NWFOAV)
      JMT = NWFODN
      ITYPE = MFOSAV(1)
      NWSEC = MFOSAV(2)
   31 NWDO = MIN (NWSEC,JMSEX-JMS)
      IF (NWDO.EQ.0) GO TO 801
      IF (ITYPE.LE.0) GO TO 91
      GO TO (101,201,301,401,501,101,101), ITYPE
   91 CALL VZEROI (IRMT(JMT+1),NWDO)
      JMT = JMT + NWDO
      JMS = JMS + NWDO
      GO TO 801
  401 NDPN = NWDO / 2
      NWDODB = NDPN * 2
      IF (NWDODB.EQ.0) GO TO 451
      DO 449 JL=1,NDPN
      IRMT(JMT+1) = MS(JMS+2)
      IRMT(JMT+2) = MS(JMS+1)
      JMT = JMT + 2
  449 JMS = JMS + 2
  451 IF (NWDODB .EQ.NWDO) GO TO 801
      IF (NWDODB+1.EQ.NWSEC) GO TO 471
      IF (NWDODB+1.EQ.NWFORE) GO TO 471
      NWDO = NWDODB
      JMS = JMS + 2
      GO TO 801
  471 JMT = JMT + 1
      JMS = JMS + 1
      IFOCON(1) = -2
      IFOCON(2) = JMT
      IFOCON(3) = NWDO
      IRMT(JMT) = 0
      GO TO 801
  501 CONTINUE
      CALL VXINVC (MS(JMS+1),IRMT(JMT+1),NWDO)
      JMT = JMT + NWDO
      JMS = JMS + NWDO
      GO TO 801
  301 CONTINUE
  201 CONTINUE
  101 CONTINUE
      CALL UCOPYI (MS(JMS+1),IRMT(JMT+1),NWDO)
      JMT = JMT + NWDO
      JMS = JMS + NWDO
  801 NWFORE = NWFOTT - JMT
      IF (JMS.GE.JMSEX) GO TO 804
      JFOCUR = JFOCUR + 2
      IF (JFOCUR.LT.JFOEND) GO TO 21
      JFOCUR = JFOREP
      GO TO 21
  804 IQUEST(1) = JMS
      NWFOAV = NWFOAV - JMS
      IF (NWFORE.EQ.0) RETURN
      NWFODN = JMT
      MFOSAV(1) = ITYPE
      MFOSAV(2) = NWSEC - NWDO
      END

*-------------------------------------------------------------------------------

      SUBROUTINE FZIREL
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
                   EQUIVALENCE (LQFS,LQSYSS(4)), (LQFF,LQSYSR(4))
     +,                        (LQFI,LQSYSR(5)), (LQFX,LQSYSR(6))
      COMMON /MZCN/  IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/  MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +,              MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +,              IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +,              LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +,                         LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +,              LQMTC1,LQMTC2, NQFRTC,NQLIVE
      COMMON /FZCI/  LUNI,LUNNI,IXDIVI,LTEMPI,IEVFLI
     +,              MSTATI,MEDIUI,IFIFOI,IDAFOI,IACMOI,IUPAKI
     +,              IADOPI,IACTVI,INCBPI,LOGLVI,MAXREI,  ISTENI
     +,              LBPARI, L4STOI,L4STAI,L4CURI,L4ENDI
     +,              IFLAGI,NFASTI,N4SKII,N4RESI,N4DONI,N4ENDI
     +,              IOPTIE,IOPTIR,IOPTIS,IOPTIA,IOPTIT,IOPTID
     +,                     IOPTIF,IOPTIG,IOPTIH,IOPTI2(4)
     +,              IDI(2),IPILI(4),NWTXI,NWSEGI,NWTABI,NWBKI,LENTRI
     +,              NWUHCI,IOCHI(16),NWUMXI,NWUHI,NWIOI
     +,              NWRDAI,NRECAI,LUHEAI,JRETCD,JERROR,NWERR
      PARAMETER      (JAUIOC=50, JAUSEG=68, JAUEAR=130)
      DIMENSION    LADESV(6)
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HFZIR, 4HEL   /
      DATA  LADESV / 6, 5*0 /
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (NWTABI.EQ.0)             GO TO 61
      LPUT  = LQTA
      LTAKE = LQTA + NWTABI
      LMT = LQMTA
   22 IF (LQ(LMT+1).NE.0)          GO TO 24
      NWSG  = LQ(LMT+3)
   23 IF (NWSG.GE.0)               GO TO 29
      IF (LTAKE.GE.LQTE)           GO TO 731
      NWSG  = NWSG + (LQ(LTAKE+1)-LQ(LTAKE))
      LTAKE = LTAKE + 2
      GO TO 23
   24 LSTA  = LQ(LMT+3)
      LEND  = LQ(LMT+4)
      NWSG  = LSTA - LEND
      NREL  = 0
      LE    = LSTA
   25 IF (LTAKE.GE.LQTE)           GO TO 731
      LA    = LQ(LTAKE)
      NREL  = NREL - (LA-LE)
      LE    = LQ(LTAKE+1)
      LQ(LPUT)   = LA
      LQ(LPUT+1) = LE
      LQ(LPUT+2) = NREL
      LQ(LPUT+3) = 0
      LTAKE = LTAKE + 2
      LPUT  = LPUT  + 4
      NWSG  = NWSG + (LE-LA)
      IF (NWSG.LT.0)               GO TO 25
   29 IF (NWSG.NE.0)               GO TO 732
      LMT = LMT + 8
      IF (LMT.LT.LQMTE)            GO TO 22
      IF (LTAKE.NE.LQTE)           GO TO 733
      LQTE = LPUT
      LQ(LQTE)   =  LQ(LQTE-3)
      LQ(LQTA-1) =  LQ(LQTA)
      IF (LOGLVI.GE.4)
     +  WRITE (IQLOG,9167) LENTRI,(LQ(J),J=LQTA,LQTE-1)
 9167 FORMAT (' FZIREL-  Relocation Table, LENTRY before=',I10/
     F (15X,3I9,I4))
      IQFLIO = 7
      CALL MZRELB
      IF (IQFLIO.LT.0)             GO TO 734
      LADESV(2) = LOCF(LENTRI) - LQSTOR
      LADESV(3) = LADESV(2) + 1
      LADESV(5) = IQLETT(9)
      LADESV(6) = IQLETT(15)
      CALL MZRELL (LADESV)
      IF (LOGLVI.GE.4)  WRITE (IQLOG,9037) LENTRI
 9037 FORMAT (10X,'LENTRY after=',I10)
      LQ(KQS+LENTRI+1) = 0
      LQ(KQS+LENTRI+2) = 0
      GO TO 999
   61 CALL FZILIN
      IF (IQFOUL.NE.0)             GO TO 734
      LENTRI = IQUEST(1)
  999 NQTRAC = NQTRAC - 2
      RETURN
  734 JERROR = 34
      IQUEST(14)= IQLN
      NWERR  = 1
      GO TO 739
  733 JERROR = 33
      IQUEST(14)= LTAKE
      IQUEST(15)= LQTE
      NWERR  = 2
      GO TO 739
  732 JERROR = 32
      IQUEST(14)= NWSG
      NWERR  = 1
      GO TO 739
  731 JERROR = 31
      IQUEST(14)= NWSG
      NWERR  = 1
  739 JRETCD = 5
      GO TO 999
      END

*-------------------------------------------------------------------------------

      SUBROUTINE FZILIN
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/  IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/  MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +,              MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +,              IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +,              LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +,                         LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +,              LQMTC1,LQMTC2, NQFRTC,NQLIVE
      IQFOUL = 0
      LENTRI = 0
      K   = 0
      LMT = LQMTA
   22 IF (LQ(LMT+1).LE.0)          GO TO 29
      IQNX  = LQ(LMT+3)
      LEND  = LQ(LMT+4)
   24 CALL MZCHLN (-7,IQNX)
      IF (IQFOUL.NE.0)             GO TO 91
      IF (IQND.LT.0)               GO TO 27
      IF (K.EQ.0)  THEN
          LENTRI = IQLS
        ELSE
          LQ(KQS+K) = IQLS
        ENDIF
      L = IQLS - IQNL - 1
      DO 26  J=1, IQNL+2
   26 LQ(KQS+L+J) = 0
      LQ(KQS+IQLS+2) = K
      K = IQLS
   27 IF (IQNX.LT.LEND)            GO TO 24
      IF (IQNX.NE.LEND)            GO TO 91
   29 LMT = LMT + 8
      IF (LMT.LT.LQMTE)            GO TO 22
      IQUEST(1) = LENTRI
      RETURN
   91 IQFOUL = 7
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZCHLS (IXST,LP)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/  IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      DIMENSION    IXST(9), LP(9)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IXSTOR = IXST(1)
      IQLS   = LP(1)
      IF (IXSTOR.EQ.-7)                 GO TO 21
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR)  CALL MZSDIV (IXSTOR,-7)
   21 IF (IQLS.LT.LQSTA(KQT+1))       GO TO 98
      IF (IQLS.GE.LQSTA(KQT+21))      GO TO 98
      IQNIO = JBYT (IQ(KQS+IQLS),19,4)
      IQID  = IQ(KQS+IQLS-4)
      IQNL  = IQ(KQS+IQLS-3)
      IQNS  = IQ(KQS+IQLS-2)
      IQND  = IQ(KQS+IQLS-1)
      IF (  JBYT(IQNL,IQBITW-3,4)
     +    + JBYT(IQNS,IQBITW-3,4)
     +    + JBYT(IQND,IQBITW-3,4) .NE.0)    GO TO 91
      IQNX  = IQLS + IQND + 9
      IF (IQNX.GT.LQSTA(KQT+21))      GO TO 91
      IQLN  = IQLS - IQNL - IQNIO - 1
      IF (IQLN.LT.LQSTA(KQT+1))       GO TO 91
      NST = JBYT (LQ(KQS+IQLN),1,16) - 12
      IF (NST.NE.IQNIO+IQNL)       GO TO 91
      IF (IQNS.GT.IQNL)            GO TO 91
      IQFOUL = 0
      RETURN
   91 IQFOUL = 7
      RETURN
   98 IQFOUL = -7
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZBOOK (IXP,LP,LSUPP,JBP, CHIDH,NL,NS,ND,NIOP,NZP)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCL/  NQLN,NQLS,NQNIO,NQID,NQNL,NQNS,NQND,NQIOCH(16)
     +,              LQSUP,NQBIA, NQIOSV(3)
      DIMENSION    IXP(9),LP(9),LSUPP(9),JBP(9),NIOP(9),NZP(9)
      CHARACTER    CHIDH*(*)
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HMZBO, 4HOK   /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      NQID = IQQUES
      NIO  = MIN (4, LEN(CHIDH))
      IF  (NIO.NE.0)  CALL UCTOH (CHIDH,NQID,4,NIO)
      NQNL  = NL
      NQNS  = NS
      NQND  = ND
      NQBIA = JBP(1)
      IODORG = NIOP(1)
      NIO = JBYT (IODORG,12,4)
      IF (NIO.EQ.0)  THEN
          NQIOCH(1) = IODORG
        ELSE
          CALL UCOPYI (NIOP,NQIOCH,NIO+1)
          NQIOSV(1) = 0
        ENDIF
        CALL MZLIFT (IXP,LP,LSUPP,63, NQID, NZP)
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZLIFT (IXDIV,LP,LSUPP,JBIAS,NAME,NZERO)
      PARAMETER      (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +,              NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION    LQMST(9)
      EQUIVALENCE (LQMST(1),LQSYSS(2))
      EQUIVALENCE (LQFORM,LQSYSS(5))
      COMMON /MZCL/  NQLN,NQLS,NQNIO,NQID,NQNL,NQNS,NQND,NQIOCH(16)
     +,              LQSUP,NQBIA, NQIOSV(3)
      COMMON /MZCN/  IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/  MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +,              MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +,              IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +,              LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +,                         LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +,              LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION    IXDIV(9), LP(9), LSUPP(9), NAME(9)
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HMZLI, 4HFT   /
      JBIT(IZW,IZP)     = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     +       IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     +      ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
******IF (IQVSTA.NE.0)       CALL ZVAUTX
      IF (JBIAS.NE.63)  THEN
          NQBIA = JBIAS
          NIO   = JBYT (NAME(5),12,4)
          CALL UCOPYI (NAME,NQID,NIO+5)
          IF (NIO.NE.0)  NQIOSV(1)=0
        ENDIF
      JDV   = IXDIV(1)
      LQSUP = LSUPP(1)
      IF (NQBIA.GE.2)  LQSUP = 0
      ICHORG = NQIOCH(1)
      NTOT   = NQNL + NQND + 10
      IF (JDV.EQ.-7)                 GO TO 24
      IF (JBYT(JDV,27,6).NE.JQSTOR)  GO TO 22
      JQDIVI = JBYT (JDV,1,26)
      IF (JQDIVI.LT.21)              GO TO 23
   22 CALL MZSDIV (JDV,0)
   23 CALL MZCHNB (LP)
   24 CONTINUE
      J = JBYT (NQID,IQBITW-7,8)
      IF (J.EQ.0)                  GO TO 91
      IF (J.EQ.255)                GO TO 91
      IF (NTOT.GE.LQSTA(KQT+21))      GO TO 91
      IF (NQNS.GT.NQNL)            GO TO 91
      IF (NQNS.LT.0)               GO TO 91
      IF (NQNL.GT.64000)           GO TO 91
      IF (NQND.LT.0)               GO TO 91
      IF (NQBIA.GE.3)              GO TO 91
      IF (LQSUP.EQ.0)              GO TO 25
      CALL MZCHLS (-7,LQSUP)
      IF (IQFOUL.NE.0)             GO TO 92
      IF (NQBIA.EQ.1)              GO TO 26
      IF (JBIT(IQ(KQS+LQSUP),IQDROP).NE.0)  GO TO 92
      IF (IQNS+NQBIA.LT.0)         GO TO 93
      GO TO 26
   25 IF (NQBIA.LE.0)              GO TO 92
   26 CONTINUE
      IDN   = 1
      LS    = LQSUP
      LSAME = LQSUP
      LNEXT = LQSUP
      IF (NQBIA.GT.0)              GO TO 38
      LNEXT = LQ(KQS+LNEXT+NQBIA)
      IF (LNEXT.EQ.0)              GO TO 36
      LSAME = LNEXT
      LS    = LNEXT
      CALL MZCHLS (-7,LNEXT)
      IF (IQFOUL.NE.0)             GO TO 94
      IDN = IQ(KQS+LNEXT-5) + 1
      GO TO 39
   36 IF (NQBIA.EQ.0)              GO TO 37
      LSAME = 0
      IDN   = -NQBIA
      GO TO 39
   37 IDN = IQ(KQS+LSAME-5) + 1
      GO TO 39
   38 IF (LNEXT.NE.0)  IDN=IQ(KQS+LNEXT-5)+1
   39 CONTINUE
      IF (ICHORG.LT.0)             GO TO 47
      IF (ICHORG.LT.8)  THEN
          NQNIO = 0
          NQIOCH(1) = ISHFT (ICHORG, 16)
          GO TO 49
        ENDIF
      IF (ICHORG-11)         45, 43, 47
   43 IF (LSAME.EQ.0)              GO TO 45
      NQNIO = IQNIO
      IF (NQNIO.EQ.0)  THEN
          NQIOCH(1) = LQ(KQS+IQLN)
          GO TO 49
        ELSE
          CALL UCOPYI (LQ(KQS+IQLN),NQIOCH,NQNIO+1)
          NQIOSV(1) = 0
          GO TO 49
        ENDIF
   45 LID  = LQFORM
      IF (LID.EQ.0)                GO TO 95
      LIOD = LQ(KQSP+LID-2)
      IF (NQID.LT.0)  LID=LQ(KQSP+LID)
      IF (NQID.EQ.IQ(KQSP+LID+3))  THEN
          IXIO = IQ(KQSP+LID+2)
        ELSE
          N = IQ(KQSP+LID+1)
          IF (N.EQ.0)              GO TO 95
          J = IUCOMP (NQID,IQ(KQSP+LID+4),N)
          IF (J.EQ.0)              GO TO 95
          LIX  = LQ(KQSP+LID-1)
          IXIO = IQ(KQSP+LIX+J)
          IQ(KQSP+LID+2) = IXIO
          IQ(KQSP+LID+3) = NQID
        ENDIF
      NQNIO = JBYT (IQ(KQSP+LIOD+IXIO+1),7,5) - 1
      GO TO 48
   47 J     = JBYT (ICHORG,1,6)
      NQNIO = JBYT (ICHORG,7,5) - 1
      IOTH  = JBYT (ICHORG,12,5)
      IF (J.EQ.1)  THEN
          IF (NQNIO.NE.IOTH)       GO TO 96
          GO TO 49
        ENDIF
      IF (J.NE.2)                  GO TO 96
      IF (IOTH.NE.0)               GO TO 96
      IXIO = JBYT (ICHORG,17,16)
      IF (IXIO.EQ.0)               GO TO 96
      LIOD = LQ(KQSP+LQFORM-2)
      IF (IXIO.GE.IQ(KQSP+LIOD+1))    GO TO 96
   48 IF (IXIO.EQ.NQIOSV(1))  THEN
          NQIOCH(1) = NQIOSV(2)
          GO TO 49
        ENDIF
      NQIOSV(1) = 0
      IF (NQNIO.GE.16)             GO TO 96
      CALL UCOPYI (IQ(KQSP+LIOD+IXIO+1),NQIOCH,NQNIO+1)
      IOTH = JBYT (NQIOCH(1),12,5)
      IF (NQNIO.NE.IOTH)           GO TO 96
      NQIOSV(1) = IXIO
      NQIOSV(2) = NQIOCH(1)
   49 NTOT = NTOT + NQNIO
      IF (JQDIVI.NE.0)             GO TO 59
      IF (LS.LT.LQSTA(KQT+1))         GO TO 58
      IF (LS.GE.LQEND(KQT+20))        GO TO 58
      IF (LS.GE.LQEND(KQT+JQDVLL))    GO TO 54
      IF (LS.LT.LQEND(KQT+2))         GO TO 57
      JQDIVI = 3
      GO TO 55
   54 JQDIVI = JQDVSY
   55 IF (LS.LT.LQEND(KQT+JQDIVI))    GO TO 61
      JQDIVI = JQDIVI + 1
      GO TO 55
   57 JQDIVI = 1
      IF (LS.LT.LQSTA(KQT+2))         GO TO 61
   58 JQDIVI = 2
      GO TO 61
   59 IF (LSAME.EQ.0)               GO TO 61
      IF (LSAME.LT.LQSTA(KQT+JQDIVI))  GO TO 97
      IF (LSAME.GE.LQEND(KQT+JQDIVI))  GO TO 97
   61 CALL MZRESV
      NQRESV = NQRESV - NTOT
      IF (NQRESV.LT.0)             GO TO 81
      IF (JQMODE.NE.0)             GO TO 63
      NQLN  = LQEND(KQT+JQDIVI)
      LE    = NQLN + NTOT
      LQEND(KQT+JQDIVI) = LE
      GO TO 65
   63 LE    = LQSTA(KQT+JQDIVI)
      NQLN  = LE - NTOT
      LQSTA(KQT+JQDIVI) = NQLN
   65 NZ = MIN (NZERO,NQND)
      IF (NZ.EQ.0)  NZ=NQND
      IF (NZ.LT.0)  NZ=0
      NST  = NQNIO + NQNL
      NQLS = NQLN + NST + 1
      CALL VZEROI (LQ(KQS+NQLN+NQNIO+1),NQNL+NZ+9)
      NQIOCH(1) = MSBYT (NST+12,NQIOCH(1),1,16)
      DO 67  J=0,NQNIO
   67 LQ(KQS+NQLN+J) = NQIOCH(J+1)
      IQ(KQS+NQLS-5) = IDN
      IQ(KQS+NQLS-4) = NQID
      IQ(KQS+NQLS-3) = NQNL
      IQ(KQS+NQLS-2) = NQNS
      IQ(KQS+NQLS-1) = NQND
      IQ(KQS+NQLS)   = ISHFT (NQNIO,18)
      IF   (NQBIA-1)         72, 73, 79
   72 LUP   = LQSUP
      KADR  = LQSUP + NQBIA
      LNEXT = LQ(KQS+KADR)
      IF (NQBIA.NE.0)              GO TO 77
      LUP   = LQ(KQS+LUP+1)
      GO TO 77
   73 LNEXT = LQSUP
      IF (LNEXT.NE.0)              GO TO 74
      LUP  = 0
      KADR = LOCF (LSUPP(1)) - LQSTOR
      IF (KADR.LT.LQSTA(KQT+1))       GO TO 78
      IF (KADR.LT.LQSTA(KQT+21))      GO TO 98
      GO TO 78
   74 LUP  = LQ(KQS+LNEXT+1)
      KADR = LQ(KQS+LNEXT+2)
   77 IF (LNEXT.EQ.0)              GO TO 78
      LQ(KQS+NQLS)    = LNEXT
      LQ(KQS+LNEXT+2) = NQLS
   78 LQ(KQS+NQLS+1) = LUP
      LQ(KQS+NQLS+2) = KADR
      LQ(KQS+KADR)   = NQLS
   79 LP(1) = NQLS
      IF (NQLOGL.GE.2)
     + WRITE (IQLOG,9079) JQSTOR,JQDIVI,NQLS,LQSUP,NQBIA,
     +                                  NQID,NQNL,NQNS,NQND
 9079 FORMAT (' MZLIFT-  Store/Div',2I3,' L/LSUP/JBIAS=',2I9,I6,
     F' ID,NL,NS,ND= ',A4,2I6,I9)
  999 NQTRAC = NQTRAC - 2
      RETURN
   81 LQMST(KQT+1) = LQSUP
      CALL MZGAR1
      LQSUP = LQMST(KQT+1)
      IF (NQBIA.GE.1)              GO TO 61
      KADR = LOCF (LSUPP(1)) - LQSTOR
      IF (KADR.LT.LQSTA(KQT+1))       GO TO 83
      IF (KADR.LT.LQSTA(KQT+21))      GO TO 61
   83 LSUPP(1) = LQSUP
      GO TO 61
   98 NQCASE = 8
      NQFATA = 1
      IQUEST(18) = KADR
      GO TO 90
   97 NQCASE = 7
      NQFATA = 1
      IQUEST(18) = LSAME
      GO TO 90
   94 NQCASE = 4
      NQFATA = 1
      IQUEST(18) = LNEXT
      GO TO 90
   96 NQCASE = 1
   95 NQCASE = NQCASE + 2
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
   91 NQCASE = NQCASE + 1
   90 NQFATA = NQFATA + 7
      IQUEST(11) = LQSUP
      IQUEST(12) = NQBIA
      IQUEST(13) = NQID
      IQUEST(14) = NQNL
      IQUEST(15) = NQNS
      IQUEST(16) = NQND
      IQUEST(17) = ICHORG
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZLINK (IXSTOR,CHNAME,LAREA,LREF,LREFL)
      COMMON /ZBCD/  IQNUM2(11),IQLETT(26),IQNUM(10),   IQPLUS,IQMINS
     +,              IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +,              IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +,              IQUNDE,IQCLSQ,IQAND, IQAT,  IQQUES,IQOPSQ,IQGREA
     +,              IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC,  IQLOWL(26)
     +,              IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV,  IQILEG
     +,              NQHOL0,NQHOLL(95)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION    LAREA(9),LREF(9),LREFL(9),NAME(2)
      CHARACTER    *(*) CHNAME
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HMZLI, 4HNK   /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR)  CALL MZSDIV (IXSTOR,-7)
******IF (IQVSTA.NE.0)       CALL ZVAUTX
      LSYS  = LQSYSS(KQT+1)
      NWTAB = IQ(KQS+LSYS+1)
      IF (NWTAB+5.GT.IQ(KQS+LSYS-1))  THEN
          JQDIVI = JQDVSY
          CALL MZPUSH (-7,LSYS,0,100,'I')
          LQSYSS(KQT+1) = LSYS
        ENDIF
      LSTO = LSYS + NWTAB
      LOCAR = LOCF (LAREA(1)) - LQSTOR
      LOCR  = LOCF (LREF(1))  - LQSTOR
      LOCRL = LOCF (LREFL(1)) - LQSTOR
      NS = LOCR    - LOCAR
      NL = LOCRL+1 - LOCAR
      IF (NL.EQ.1)  THEN
          NS = NS + 1
          NL = NS
        ENDIF
      LOCARE = LOCAR + NL
      MODAR  = NS
      NAME(1) = IQBLAN
      NAME(2) = IQBLAN
      N = MIN (8, LEN(CHNAME))
      IF (N.NE.0)  CALL UCTOH (CHNAME,NAME,4,N)
      IQ(KQS+LSTO+1) = LOCAR
      IQ(KQS+LSTO+2) = LOCARE
      IQ(KQS+LSTO+3) = MODAR
      IQ(KQS+LSTO+4) = NAME(1)
      IQ(KQS+LSTO+5) = NAME(2)
      IQTABV(KQT+13) = MIN (IQTABV(KQT+13), LOCAR)
      IQTABV(KQT+14) = MAX (IQTABV(KQT+14), LOCARE)
      IF (NQLOGL.GE.0)
     +WRITE (IQLOG,9039) NAME,JQSTOR,NL,NS
 9039 FORMAT (1X/' MZLINK.  Initialize Link Area  ',2A4,'  for Store'
     F,I3,' NL/NS=',2I6)
      IF (LOCR .LT.LOCAR)          GO TO 91
      IF (LOCRL.LT.LOCAR)          GO TO 91
      IF (NL.LT.NS)                GO TO 91
      KLA = KQS + LOCAR
      KLE = KQS + LOCARE
      DO 47  JSTO=1,NQSTOR+1
      IF (NQALLO(JSTO).NE.0)       GO TO 47
      JT  = NQOFFT(JSTO)
      JS  = NQOFFS(JSTO)
      JSA = JS  - IQTABV(JT+2) + 1
      JSE = JS  + LQSTA(JT+21) + 1
      JTA = JT  + LQBTIS       + 1
      JTE = JTA + NQTSYS
      IF (KLE.GT.JTA .AND. KLA.LT.JTE)    GO TO 92
      IF (KLE.GT.JSA .AND. KLA.LT.JSE)    GO TO 93
      L = JS+ LQSYSS(JT+1)
      N = IQ(L+1)
      IF (N.LT.12)                 GO TO 47
      DO 44  J=12,N,5
      JLA = JS + IQ(L+J)
      JLE = JS + IQ(L+J+1)
      IF (KLE.GT.JLA .AND. KLA.LT.JLE)    GO TO 94
   44 CONTINUE
   47 CONTINUE
   61 IQ(KQS+LSYS+1) = NWTAB + 5
      CALL VZEROI (LAREA,NL)
  999 NQTRAC = NQTRAC - 2
      RETURN
   94 NQCASE = 1
      NQFATA = 3
      IQUEST(21) = IQ(L+J+3)
      IQUEST(22) = IQ(L+J+4)
      IQUEST(23) = JLA + LQSTOR
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 3
      IQUEST(18) = JSTO - 1
      IQUEST(19) = NQPNAM(JT+1)
      IQUEST(20) = NQPNAM(JT+2)
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 7
      IQUEST(11) = NAME(1)
      IQUEST(12) = NAME(2)
      IQUEST(13) = LOCAR + LQSTOR
      IQUEST(14) = LOCR  + LQSTOR
      IQUEST(15) = LOCRL + LQSTOR
      IQUEST(16) = NS
      IQUEST(17) = NL
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZPUSH (IXDIV,LORGP,INCNLP,INCNDP,CHOPT)
      COMMON /ZBCD/ IQNUM2(11),IQLETT(26),IQNUM(10), IQPLUS,IQMINS
     +, IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +, IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +, IQUNDE,IQCLSQ,IQAND, IQAT, IQQUES,IQOPSQ,IQGREA
     +, IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC, IQLOWL(26)
     +, IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV, IQILEG
     +, NQHOL0,NQHOLL(95)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCL/ NQLN,NQLS,NQNIO,NQID,NQNL,NQNS,NQND,NQIOCH(16)
     +, LQSUP,NQBIA, NQIOSV(3)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION IXDIV(9),LORGP(9),INCNLP(9),INCNDP(9)
      CHARACTER *(*) CHOPT
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZPU, 4HSH  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     + IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     + ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (IXDIV(1).EQ.-7) GO TO 12
      CALL MZSDIV (IXDIV,0)
   12 CALL MZCHNB (LORGP)
      LORG = LORGP(1)
      INCNL = INCNLP(1)
      INCND = INCNDP(1)
      CALL UOPTC (CHOPT,'RI',IQUEST)
      IFLAG = MIN (2, IQUEST(1)+2*IQUEST(2))
      IF ((INCNL.EQ.0) .AND. (INCND.EQ.0)) GO TO 999
      LQSYSR(KQT+1) = LORG
      JQDIVI = MZFDIV (-7, LORG)
      IF (JQDIVI.EQ.0) GO TO 91
      CALL MZCHLS (-7,LORG)
      IF (IQFOUL.NE.0) GO TO 91
      NL = IQNL
      NS = IQNS
      ND = IQND
      NQNIO = IQNIO
      NQID = IQID
      NQNL = NL + INCNL
      NQNS = MIN (NS,NQNL)
      NQND = ND + INCND
      IF (NS.EQ.NL) NQNS = NQNL
      IF (NQLOGL.GE.2)
     + WRITE (IQLOG,9032) JQSTOR,JQDIVI,LORG,NQID,INCNL,INCND,CHOPT
 9032 FORMAT (' MZPUSH-  Store/Div',2I3,' L/ID/INCNL/INCND/OPT=',
     FI9,1X,A4,2I7,1X,A)
      IF (JBIT(IQ(KQS+LORG),IQDROP).NE.0) GO TO 92
      IF (NQND+NQNL.GE.LQSTA(KQT+21)) GO TO 93
      IF (NQND.LT.0) GO TO 93
      IF (NQNL.GT.64000) GO TO 93
      IF (NQNS.LT.0) GO TO 93
      NLC = MIN (NL,NQNL)
      NSC = MIN (NS,NQNS)
      NDC = MIN (ND,NQND)
      IF (NQNS.GE.NS) GO TO 36
      L = LORG - NS - 1
      LD = LORG - NQNS
   34 L = L + 1
      IF (L.GE.LD) GO TO 36
      LNZ = LQ(KQS+L)
   35 IF (LNZ.EQ.0) GO TO 34
      IF (LQ(KQS+LNZ+2).NE.L) GO TO 34
      IF (JBIT(IQ(KQS+LNZ),IQDROP).EQ.0) GO TO 94
      LNZ = LQ(KQS+LNZ)
      GO TO 35
   36 LN = LORG - NL - NQNIO - 1
      CALL UCOPYI (LQ(KQS+LN),NQIOCH,NQNIO+1)
      IF (NQNIO.NE.0) NQIOSV(1)=0
      NQIOCH(1) = MSBYT (NQNL+NQNIO+12,NQIOCH(1),1,16)
   41 LE = LORG + ND + 9
      INCTT = INCNL + INCND
      INCMX = MAX (INCNL,INCND)
      INCMI = MIN (INCNL,INCND)
      CALL MZRESV
      IF (JQMODE.NE.0) GO TO 45
      IF (LE.NE.LQEND(KQT+JQDIVI)) GO TO 51
      IF (INCNL.GE.0) GO TO 42
      IF (IFLAG.NE.1) GO TO 42
      IF ((NQRESV.GE.INCTT).AND.(NQRESV.LT.INCND)) GO TO 42
      LNN = LN - INCNL
      CALL UCOPYI (NQIOCH,LQ(KQS+LNN),NQNIO+1)
      IQ(KQS+LORG-3) = NQNL
      IQ(KQS+LORG-2) = NQNS
      NWD = -INCNL
      CALL MZPUDX (LN,NWD)
      INCNL = 0
      INCTT = INCND
      LN = LNN
      NL = NQNL
   42 NQRESV = NQRESV - INCTT
      IF (NQRESV.LT.0) GO TO 49
      NDELTA = INCNL
      LNEW = LORG + NDELTA
      LQEND(KQT+JQDIVI) = LQEND(KQT+JQDIVI) + INCTT
      IF (NDELTA.EQ.0) THEN
          IQ(KQS+LNEW-1) = NQND
          IF (IFLAG.NE.0) GO TO 81
          IF (INCMI.GE.0) GO TO 81
          GO TO 71
        ELSE
          CALL UCOPY2 (LQ(KQS+LORG-NLC),LQ(KQS+LNEW-NLC),NLC+NDC+9)
          IF (INCNL.GT.0) CALL VZEROI (LQ(KQS+LNEW-NQNL),INCNL)
          LQ(KQS+LN) = NQIOCH(1)
          IQ(KQS+LNEW-3) = NQNL
          IQ(KQS+LNEW-2) = NQNS
          IQ(KQS+LNEW-1) = NQND
          GO TO 61
        ENDIF
   45 IF (LN.NE.LQSTA(KQT+JQDIVI)) GO TO 51
      IF (INCND.GE.0) GO TO 47
      IF (IFLAG.NE.1) GO TO 47
      IF ((NQRESV.GE.INCTT).AND.(NQRESV.LT.INCNL)) GO TO 47
      IQ(KQS+LORG-1) = NQND
      L = LE + INCND
      NWD = -INCND
      CALL MZPUDX (L,NWD)
      INCND = 0
      INCTT = INCNL
      ND = NQND
   47 NQRESV = NQRESV - INCTT
      IF (NQRESV.LT.0) GO TO 49
      LNN = LN - INCTT
      NDELTA = -INCND
      LQSTA(KQT+JQDIVI) = LNN
      LNEW = LORG + NDELTA
      IF (NDELTA.NE.0) CALL UCOPY2 (LQ(KQS+LORG-NLC)
     +, LQ(KQS+LNEW-NLC), NLC+NDC+9)
      IF (INCNL.GT.0) CALL VZEROI (LQ(KQS+LNEW-NQNL),INCNL)
      CALL UCOPYI (NQIOCH,LQ(KQS+LNN),NQNIO+1)
      IQ(KQS+LNEW-3) = NQNL
      IQ(KQS+LNEW-2) = NQNS
      IQ(KQS+LNEW-1) = NQND
      IF (NDELTA.NE.0) GO TO 61
      IF (IFLAG.NE.0) GO TO 81
      IF (INCMI.GE.0) GO TO 81
      GO TO 71
   49 CALL MZGAR1
      LORG = LQSYSR(KQT+1)
      LN = LORG - NL - NQNIO - 1
      GO TO 41
   51 IF (INCMX.GT.0) GO TO 56
      IF (INCNL.EQ.0) GO TO 52
      LNN = LN - INCNL
      CALL UCOPYI (NQIOCH,LQ(KQS+LNN),NQNIO+1)
      IQ(KQS+LORG-3)= NQNL
      IQ(KQS+LORG-2)= NQNS
      CALL MZPUDX (LN,-INCNL)
      IF (INCND.EQ.0) GO TO 54
   52 IQ(KQS+LORG-1) = NQND
      LD = LE + INCND
      NWD = -INCND
      CALL MZPUDX (LD,NWD)
   54 LNEW = LORG
      NDELTA = 0
      IF (IFLAG.NE.0) GO TO 999
      GO TO 71
   56 J = 64*(32*NQNIO + NQNIO + 1) + 1
      NQIOCH(1) = MSBYT (J,NQIOCH(1),1,16)
      NQBIA = 2
      CALL MZLIFT (-7,LNEW,0,63,NQID,-1)
      LORG = LQSYSR(KQT+1)
      NDELTA = LNEW - LORG
      CALL UCOPYI (LQ(KQS+LORG-NLC),LQ(KQS+LNEW-NLC),NLC+4)
      CALL UCOPYI (IQ(KQS+LORG), IQ(KQS+LNEW), NDC+1)
      IQ(KQS+LORG) = MSBIT1 (IQ(KQS+LORG),IQDROP)
   61 IF (IFLAG.LT.2) GO TO 71
      K = LQ(KQS+LNEW+2)
      IF (K.EQ.0) GO TO 62
      IF (LQ(KQS+K).NE.LORG) GO TO 95
      LQ(KQS+K) = LNEW
   62 K = LNEW
      L = LQ(KQS+K)
      IF (L.EQ.0) GO TO 65
      IF (L.EQ.LORG) GO TO 64
      LQ(KQS+L+2) = K
   63 K = L
      L = LQ(KQS+K)
      IF (L.EQ.0) GO TO 65
      IF (L.NE.LORG) GO TO 63
   64 LQ(KQS+K) = LNEW
   65 K = LNEW - NSC - 1
   66 K = K + 1
      IF (K.GE.LNEW) GO TO 81
      L = LQ(KQS+K)
      IF (L.EQ.0) GO TO 66
      IF (LQ(KQS+L+2).NE.K-NDELTA) GO TO 66
      LQ(KQS+L+2) = K
      LF = L
   68 LQ(KQS+L+1) = LNEW
      L = LQ(KQS+L)
      IF (L.EQ.LF) GO TO 66
      IF (L.NE.0) GO TO 68
      GO TO 66
   71 MQDVGA = 0
      MQDVWI = 0
      JQSTMV = -1
      IF (NQLOGL.GE.1)
     + WRITE (IQLOG,9071) JQSTOR,JQDIVI,LORG,NQID
 9071 FORMAT (' MZPUSH-  Store/Div',2I3,' Relocation pass for L/ID ='
     F,I9,1X,A4)
      CALL MZTABM
      LMT = LQMTA - 8
   74 LMT = LMT + 8
      IF (LQ(LMT).NE.JQDIVI) GO TO 74
      LQ(LMT+1) = 2
      CALL MZTABX
      LQMTE = LQMTLU
      LQ(LQTA-1) = LORG - NL - NQNIO - 1
      LQ(LQTA) = LORG - NLC
      LQ(LQTA+1) = LORG + NDC + 9
      LQ(LQTA+2) = NDELTA
      LQ(LQTA+3) = 0
      LQ(LQTA+4) = LORG + ND + 9
      LQTE = LQTA + 4
      CALL MZRELX
      NQDPSH(KQT+JQDIVI) = NQDPSH(KQT+JQDIVI) + 1
   81 LORGP(1) = LNEW
      IF (INCND.GT.0) CALL VZEROI (IQ(KQS+LNEW+ND+1),INCND)
  999 NQTRAC = NQTRAC - 2
      RETURN
   95 NQCASE = 3
      NQFATA = 1
      IQUEST(19) = K
      GO TO 92
   94 NQCASE = 1
      NQFATA = 2
      IQUEST(19) = L - LORG
      IQUEST(20) = LQ(KQS+L)
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 7
      IQUEST(12) = NQID
      IQUEST(13) = NS
      IQUEST(14) = NL
      IQUEST(15) = ND
      IQUEST(16) = NQNIO
      IQUEST(17) = INCNL
      IQUEST(18) = INCND
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 1
      IQUEST(11) = LORG
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZNEED (IXDIV,NEEDP,CHOPT)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION IXDIV(9),NEEDP(9)
      CHARACTER *(*) CHOPT
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZNE, 4HED  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      JDV = IXDIV(1)
      NEED = NEEDP(1)
      IF (JBYT(JDV,27,6).NE.JQSTOR) GO TO 22
      JQDIVI = JBYT (JDV,1,26)
      IF (JQDIVI.EQ.0) GO TO 22
      IF (JQDIVI.LT.21) GO TO 23
   22 CALL MZSDIV (JDV,4)
   23 CONTINUE
      CALL MZRESV
      NQRESV = NQRESV - NEED
      IF (NQRESV.LT.0) GO TO 41
   28 IQUEST(11) = NQRESV
      IQUEST(12) = LQEND(KQT+JQDIVI) - LQSTA(KQT+JQDIVI)
      IQUEST(13) = NQDMAX(KQT+JQDIVI)
      IF (NQLOGL.GE.2)
     + WRITE (IQLOG,9029) JQSTOR,JQDIVI,NEED,NQRESV,CHOPT
 9029 FORMAT (' MZNEED-  Store/Div',2I3,' NEED/Excess=',2I8
     F,' Opt=',A)
  999 NQTRAC = NQTRAC - 2
      RETURN
   41 CALL UOPTC (CHOPT,'G',IQUEST)
      IF (IQUEST(1).EQ.0) GO TO 28
      NQPERM = 1
      CALL MZGAR1
      NQPERM = 0
      GO TO 28
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZDATE(IWORD,IDATE,ITIME,ICASE)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IF(ICASE.EQ.1)THEN
         ICONT = JBYT(IWORD,9,24)
         IMINUT= MOD(ICONT,60)
         IM1 = ICONT-IMINUT
         IHOUR = MOD(IM1/60,24)
         ITIME = 100*IHOUR+IMINUT
         IM2 = IM1-60*IHOUR
         IDAY = MOD(IM2/1440,31)
         IF(IDAY.EQ.0)IDAY=31
         IM3 = IM2-1440*IDAY
         IMONTH= MOD(IM3/44640,12)
         IF(IMONTH.EQ.0)IMONTH=12
         IYEAR = (IM3-44640*IMONTH)/535680
         IF(IYEAR.GE.14) THEN
            IDATE = 10000*(IYEAR-14)+100*IMONTH+IDAY
         ELSE
            IDATE = 860000+10000*IYEAR+100*IMONTH+IDAY
         ENDIF
      ELSE
******   IF(ICASE.NE.3)CALL DATIME(IDATE,ITIME)
         IF(IDATE.GE.860000) THEN
            IDAT2 = IDATE - 860000
         ELSE
            IDAT2 = IDATE + 140000
         ENDIF
         IYEAR = IDAT2/10000
         IMONTH= (IDAT2-10000*IYEAR)/100
         IDAY = MOD(IDAT2,100)
         IHOUR = ITIME/100
         IMINUT= MOD(ITIME,100)
         ICONT2= IDAY+31*(IMONTH+12*IYEAR)
         ICONT = IMINUT+60*(IHOUR+24*ICONT2)
         CALL SBYT(ICONT,IWORD,9,24)
      ENDIF
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZCDIR(CHPATH,CHOPT)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      PARAMETER (NLPATM=100)
      COMMON /RZDIRN/NLCDIR,NLNDIR,NLPAT
      COMMON /RZDIRC/CHCDIR(NLPATM),CHNDIR(NLPATM),CHPAT(NLPATM)
      CHARACTER*16 CHNDIR, CHCDIR, CHPAT
      COMMON /RZCH/ CHWOLD,CHL
      CHARACTER*255 CHWOLD,CHL
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     + KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     + KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     + KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     + KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      DIMENSION IOPTV(5)
      EQUIVALENCE (IOPTR,IOPTV(1)), (IOPTP,IOPTV(2)), (IOPTU,IOPTV(3))
      EQUIVALENCE (IOPTK,IOPTV(4)), (IOPTQ,IOPTV(5))
      CHARACTER*(*) CHPATH,CHOPT
      CHARACTER*1 COPTQ
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IQUEST(1)=0
      CALL UOPTC (CHOPT,'RPUKQ',IOPTV)
      IF(IOPTK.NE.0) IOPTU=0
      IF(IOPTR.NE.0) CHPATH = ' '
      IF(LQRS.EQ.0) THEN
         IQUEST(1) = 4
         GOTO 999
      ENDIF
      LRZ=LQRS
   10 IF(LRZ.EQ.0) GOTO 20
      IF(IQ(KQSP+LRZ-5).NE.0) GOTO 30
      LRZ=LQ(KQSP+LRZ)
      GO TO 10
   20 CONTINUE
      IQUEST(1) = 5
      GOTO 999
   30 CONTINUE
      IF(IOPTR.NE.0)THEN
         CALL RZPAFF(CHCDIR,NLCDIR,CHPATH)
         GO TO 999
      ENDIF
      IF(IOPTP.NE.0)THEN
         CALL RZPAFF(CHCDIR,NLCDIR,CHL)
         WRITE(IQPRNT,10000)CHL(1:LENOCC(CHL))
10000 FORMAT(' Current Working Directory = ',A)
         GO TO 999
      ENDIF
      COPTQ = ' '
      IF(IOPTQ.NE.0) COPTQ = 'Q'
      IF(LCDIR.NE.0.AND.ISAVE.NE.0.AND.IOPTK.EQ.0)THEN
         LBANK=LCDIR
   40 IF(LBANK.NE.LTOP)THEN
            LUP=LQ(KQSP+LBANK+1)
            IF(IOPTU.EQ.0)THEN
               CALL SBIT1(IQ(KQSP+LBANK),IQDROP)
            ELSE
               CALL MZDROP(JQPDVS,LBANK,' ')
               IQ(KQSP+LTOP+KIRIN)=0
            ENDIF
            LBANK=LUP
            IF(LBANK.NE.0)GO TO 40
         ENDIF
      ENDIF
      IF(IOPTU.NE.0)THEN
         print*,'>>>>>> CALL RZRTOP'
******   CALL RZRTOP
      ENDIF
      IF(ISAVE.NE.0)THEN
         CALL RZSAVE
      ENDIF
      CALL RZPATH(CHPATH)
      CALL RZFDIR('RZCDIR',LT,LDIR,COPTQ)
      IF(LDIR.NE.0)THEN
         NLCDIR= NLPAT
         LCDIR = LDIR
         LTOP = LT
         DO 50 I=1,NLPAT
            CHCDIR(I)=CHPAT(I)
   50 CONTINUE
      ELSE
         IF(LCDIR.NE.0)CALL SBIT0(IQ(KQSP+LCDIR),IQDROP)
         GO TO 999
      ENDIF
      LFREE = LQ(KQSP+LTOP-2)
      LUSED = LQ(KQSP+LTOP-3)
      LPURG = LQ(KQSP+LTOP-5)
      LROUT = LQ(KQSP+LTOP-6)
      LRIN = LQ(KQSP+LTOP-7)
      LB = IQ(KQSP+LTOP+KLB)
      LREC = IQ(KQSP+LTOP+LB+1)
      LUN = IQ(KQSP+LTOP-5)
      IZRECL = IQ(KQSP+LTOP+LB+1)
      IMODEC = JBIT(IQ(KQSP+LTOP),5)
      IMODEH = JBIT(IQ(KQSP+LTOP),6)
      IMODEX = JBIT(IQ(KQSP+LTOP+KPW1+2),12)
      IQUEST(7)=IQ(KQSP+LCDIR+KNKEYS)
      IQUEST(8)=IQ(KQSP+LCDIR+KNWKEY)
      IQUEST(9)=IQ(KQSP+LCDIR+KNSD)
      IQUEST(10)=IQ(KQSP+LCDIR+KQUOTA)
      IQUEST(11)=LCDIR
      IQUEST(12)=LTOP
      IQUEST(13)=IQ(KQSP+LCDIR+KLK)
      CALL RZDATE(IQ(KQSP+LCDIR+KDATEC),IDATEC,ITIMEC,1)
      CALL RZDATE(IQ(KQSP+LCDIR+KDATEM),IDATEM,ITIMEM,1)
      IQUEST(14)=IDATEC
      IQUEST(15)=ITIMEC
      IQUEST(16)=IDATEM
      IQUEST(17)=ITIMEM
      IQUEST(18)=IQ(KQSP+LCDIR+KRUSED)
      IQUEST(19)=IQ(KQSP+LCDIR+KMEGA)
      IQUEST(20)=IQ(KQSP+LCDIR+KWUSED)
      IQUEST(21)=IQ(KQSP+LCDIR+IQ(KQSP+LCDIR+KLD))
      IF(JBYT(IQ(KQSP+LCDIR+KPW1+2),6,5).NE.0)THEN
         IF(IQ(KQSP+LCDIR+KPW1).NE.IHPWD(1).OR.
     + IQ(KQSP+LCDIR+KPW1+1).NE.IHPWD(2))THEN
            CALL SBIT1(IQ(KQSP+LCDIR),1)
         ELSE
            CALL SBIT0(IQ(KQSP+LCDIR),1)
         ENDIF
      ENDIF
      IF(JBIT(IQ(KQSP+LTOP),1).NE.0)CALL SBIT1(IQ(KQSP+LCDIR),1)
  999 END

*-------------------------------------------------------------------------------

      SUBROUTINE RZFILE(LUNIN,CHDIR,CHOPT)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      PARAMETER (NLPATM=100)
      COMMON /RZDIRN/NLCDIR,NLNDIR,NLPAT
      COMMON /RZDIRC/CHCDIR(NLPATM),CHNDIR(NLPATM),CHPAT(NLPATM)
      CHARACTER*16 CHNDIR, CHCDIR, CHPAT
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     + KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     + KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     + KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     + KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      COMMON/RZCKEY/IHEAD(3),KEY(100),KEY2(100),KEYDUM(50)
      INTEGER KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON/RZCYCLE/KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON /RZBUFF/ ITEST(8704)
      CHARACTER CHOPT*(*),CHDIR*(*)
      CHARACTER*16 CHTOP
      DIMENSION IOPTV(10)
      EQUIVALENCE (IOPTM,IOPTV(1)), (IOPTU,IOPTV(2))
      EQUIVALENCE (IOPTS,IOPTV(3)), (IOPTL,IOPTV(4))
      EQUIVALENCE (IOPT1,IOPTV(5)), (IOPTD,IOPTV(6))
      EQUIVALENCE (IOPTC,IOPTV(7)), (IOPTX,IOPTV(8))
      EQUIVALENCE (IOPTB,IOPTV(9)), (IOPTH,IOPTV(10))
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IQUEST(1)=0
      LOGLV = MIN(NQLOGD,4)
      LOGLV = MAX(LOGLV,-3)
      LUNSA = LUN
      LUNP = LUNIN
      CALL RZSAVE
      CALL UOPTC (CHOPT,'MUSL1DCXBH',IOPTV)
      IRELAT=0
      IMODEC=IOPTC
      IMODEH=IOPTH
      IMODEX=IOPTX
      IF(IOPTC.NE.0) THEN
         LRECP = IQUEST(10)
         LUNP = IQUEST(11)
      ENDIF
      IF(IOPTH.NE.0) THEN
         LRECP = IQUEST(10)
         LUN = IQUEST(11)
         LUSER = LUNIN
      ENDIF
      IF(IOPTM.NE.0)THEN
         LRECP=1024
         LUN=-99
      ELSEIF(IOPTH.EQ.0) THEN
         IZRECL=LRECP
         CALL RZIODO(LUNP,50,2,ITEST,1)
         IF(IOPTX.EQ.0) THEN
            CALL VXINVB(ITEST(9),1)
            IF(JBIT(ITEST(9),12).NE.0)THEN
               IMODEX=1
               CALL RZIODO(LUNP,50,2,ITEST,1)
            ELSE
               CALL VXINVB(ITEST(9),1)
            ENDIF
         ENDIF
         IF(IQUEST(1).NE.0)GO TO 30
         LB=ITEST(KLB)
         IF(LB.GT.48)CALL RZIODO(LUNP,LB+6,2,ITEST,1)
         IF(LB.GT.100)THEN
            IF(LOGLV.GE.-1) WRITE(IQLOG,10000)
10000 FORMAT(' RZFILE. WARNING!! Top directory is big')
         ENDIF
         LRECP=ITEST(LB+1)
         LUN=LUNP
         IQUEST(1)=0
      ENDIF
      IF(LOGLV.GE.0) WRITE(IQLOG,10200) LUN,LRECP,CHOPT
10200 FORMAT(' RZFILE. UNIT ',I6,' Initializing with LREC=',I6,
     +', OPT= ',A)
      CALL MZSDIV (0,-7)
      LRZ=LQRS
   10 IF(LRZ.NE.0)THEN
         IF(IQ(KQSP+LRZ-5).EQ.LUN)THEN
            IQUEST(1)=1
            IF(LOGLV.GE.-2) WRITE(IQLOG,10300)
10300 FORMAT(' RZFILE. Unit is already in use')
            LUN=LUNSA
            GO TO 30
         ELSE
            LRZ=LQ(KQSP+LRZ)
            GO TO 10
         ENDIF
      ENDIF
      IF(LQRS.EQ.0)THEN
         CALL MZLINK(JQPDVS,'RZCL',LTOP,LTOP,LFROM)
         CALL MZBOOK (JQPDVS,LRZ0,LQRS,1,'RZ0 ',2,2,36,2,0)
         IQ(KQSP+LRZ0-5)=0
         ISAVE = 1
         NHPWD = 0
         CALL VBLANK(IHPWD,2)
      ENDIF
      NCHD = LEN(CHDIR)
      IF(NCHD.GT.16)NCHD=16
      CHTOP = CHDIR(1:NCHD)
      CALL MZBOOK(JQPDVS,LTOP,LQRS,1,'RZ  ',10,9,LRECP,2,0)
      IF(IOPTM.EQ.0)THEN
         IQ(KQSP+LTOP-5) = LUN
         IF(IOPTC.NE.0) CALL SBIT1(IQ(KQSP+LTOP),5)
         IF(IOPTH.NE.0) THEN
            CALL SBIT1(IQ(KQSP+LTOP),6)
            CALL SBYT(LUSER,IQ(KQSP+LTOP),7,7)
         ENDIF
      ELSE
         NMEM=IQ(KQSP+LRZ0)+1
         IQ(KQSP+LRZ0)=NMEM
         IQ(KQSP+LTOP-5)=-NMEM
         IF(2*NMEM.GT.IQ(KQSP+LRZ0-1))THEN
            CALL MZPUSH(JQPDVS,LRZ0,0,10,' ')
         ENDIF
         IQ(KQSP+LRZ0+2*NMEM-1)=LOCF(LUNP)-LOCF(IQ(1))+1
         IQ(KQSP+LRZ0+2*NMEM )=LRECP
         LUN=-NMEM
      ENDIF
      CALL RZIODO(LUN,LRECP,2,IQ(KQSP+LTOP+1),1)
      IF(IQUEST(1).NE.0)GO TO 30
      LD = IQ(KQSP+LTOP+KLD)
      LB = IQ(KQSP+LTOP+KLB)
      LREC = IQ(KQSP+LTOP+LB+1)
      NRD = IQ(KQSP+LTOP+LD)
      IMODEX=JBIT(IQ(KQSP+LTOP+KPW1+2),12)
      NPUSH=NRD*LREC-LRECP
      IF(NPUSH.NE.0)CALL MZPUSH(JQPDVS,LTOP,0,NPUSH,'I')
      DO 20 I=2,NRD
         CALL RZIODO(LUN,LREC,IQ(KQSP+LTOP+LD+I),
     + IQ(KQSP+LTOP+(I-1)*LREC+1),1)
         IF(IQUEST(1).NE.0)GO TO 30
   20 CONTINUE
      CALL VBLANK(IQ(KQSP+LTOP+1),4)
      CALL UCTOH(CHDIR,IQ(KQSP+LTOP+1),4,NCHD)
      CALL ZHTOI(IQ(KQSP+LTOP+1),IQ(KQSP+LTOP+1),4)
      CALL SBYT(NCHD,IQ(KQSP+LTOP+KPW1+2),1,5)
      CALL UCOPYI(IQ(KQSP+LTOP+KPW1),IHPWD,2)
      NHPWD=JBYT(IQ(KQSP+LTOP+KPW1+2),6,5)
      IQ(KQSP+LTOP+KIRIN)=0
      IQ(KQSP+LTOP+KIROUT)=0
      LFREE = 0
      LUSED = 0
      LRIN = 0
      LPURG = 0
      LROUT = 0
      LCDIR = LTOP
      NLCDIR= 1
      NLNDIR= 1
      NLPAT = 1
      CHCDIR(1)=CHTOP
      CHNDIR(1)=CHTOP
      IF(IOPTD.NE.0)THEN
         print*,'>>>>>> CALL RZDLOK'
******   CALL RZDLOK
      ENDIF
      IF(IOPTL.NE.0)THEN
         print*,'>>>>>> CALL RZLLOK'
******   CALL RZLLOK
      ENDIF
      LOGL = LOGLV + 3
      CALL SBYT(LOGL,IQ(KQSP+LTOP),15,3)
      CALL RZVCYC(LTOP)
      IQUEST(13) = IQ(KQSP+LTOP+KRZVER)
      IF(IOPTB.NE.0) THEN
         print*,'>>>>>> CALL RZVERI(...)'
      ENDIF
      CALL SBIT1(IQ(KQSP+LTOP),1)
      IF(IOPTU.NE.0.OR.IOPT1.NE.0)THEN
         CALL SBIT0(IQ(KQSP+LTOP),1)
         CALL MZBOOK(JQPDVS,LFREE,LTOP,-2,'RZFR',0,0,21,2,0)
         IQ(KQSP+LFREE-5)=LUN
         IF(IOPTS.EQ.0)THEN
            CALL SBIT1(IQ(KQSP+LTOP),3)
            print*,'>>>>>> CALL RZLLOK'
******      CALL RZLOCK('RZFILE')
            IF(IQUEST(1).NE.0)THEN
               CALL SBIT1(IQ(KQSP+LTOP),1)
               IQ1=IQUEST(1)
               CALL MZDROP(JQPDVS,LFREE,' ')
               LFREE=0
               IQUEST(1)=2+IQ1
               GO TO 30
            ENDIF
         ELSE
            CALL SBIT0(IQ(KQSP+LTOP),3)
         ENDIF
         CALL MZBOOK(JQPDVS,LUSED,LTOP,-3,'RZUS',0,0,21,2,0)
         IQ(KQSP+LUSED-5)=LUN
      ENDIF
      IQUEST(7)=IQ(KQSP+LCDIR+KNKEYS)
      IQUEST(8)=IQ(KQSP+LCDIR+KNWKEY)
   30 RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZFDIR(CHROUT,LT,LDIR,CHOPT)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      PARAMETER (NLPATM=100)
      COMMON /RZDIRN/NLCDIR,NLNDIR,NLPAT
      COMMON /RZDIRC/CHCDIR(NLPATM),CHNDIR(NLPATM),CHPAT(NLPATM)
      CHARACTER*16 CHNDIR, CHCDIR, CHPAT
      COMMON /RZCH/ CHWOLD,CHL
      CHARACTER*255 CHWOLD,CHL
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     + KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     + KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     + KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     + KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      INTEGER KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON/RZCYCLE/KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      CHARACTER*(*) CHROUT
      CHARACTER*(*) CHOPT
      DIMENSION IHDIR(4)
      LOGICAL RZSAME
      INTEGER FQUOTA
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IOPTQ = INDEX(CHOPT,'Q')
      LT=0
      LDIR=0
      IF(LQRS.EQ.0) GOTO 110
      IF(NLPAT.LE.0)THEN
         CHL='??? '
         GOTO 90
      ENDIF
      CALL VBLANK(IHDIR,4)
      CALL UCTOH(CHPAT(1),IHDIR,4,16)
      CALL ZHTOI(IHDIR,IHDIR,4)
      LRZ=LQRS
   10 IF(.NOT.RZSAME(IHDIR,IQ(KQSP+LRZ+1),4))THEN
         LRZ = LQ(KQSP+LRZ)
         IF(LRZ.GT.0)GOTO 10
         GOTO 80
      ENDIF
      LTEMP = LRZ
      LT = LRZ
      LDIR = LRZ
      CALL RZVCYC(LT)
      IF(NLPAT.LT.2)GOTO 110
      LBT = IQ(KQSP+LRZ+KLB)
      LREF = IQ(KQSP+LRZ+LBT+1)
      LUNF = IQ(KQSP+LRZ-5)
      FQUOTA = IQ(KQSP+LRZ+KQUOTA)
      LOGLV = JBYT(IQ(KQSP+LT),15,3)-3
      IZRECL = IQ(KQSP+LT+LBT+1) ! SWON: Needed by CFSEEK in RZIODO
      IMODEX = JBIT(IQ(KQSP+LT+KPW1+2),12)
      IMODEC = JBIT(IQ(KQSP+LT),5)
      IMODEH = JBIT(IQ(KQSP+LT),6)
      DO 60 IL=2,NLPAT
         CALL VBLANK(IHDIR,4)
         CALL UCTOH(CHPAT(IL),IHDIR,4,16)
         CALL ZHTOI(IHDIR,IHDIR,4)
         CALL SBIT0(IQ(KQSP+LRZ),IQDROP)
         NSDIR=IQ(KQSP+LRZ+KNSD)
         LS =IQ(KQSP+LRZ+KLS)
         IF(NSDIR.LE.0)GOTO 80
         DO 50 I=1,NSDIR
            IH=LS+7*(I-1)
            IF(RZSAME(IHDIR,IQ(KQSP+LRZ+IH),4))THEN
               IF (KVSCYC.EQ.0) THEN
                  IRS = JBYT(IQ(KQSP+LRZ+IH+5),1,18)
               ELSE
                  IRS = IQ(KQSP+LRZ+IH+5)
               ENDIF
               IQUEST(20) = 0
               IF(IRS.LE.0.OR.IRS.GT.FQUOTA) GOTO 100
               LRN = LQ(KQSP+LRZ-1)
   20 IF(LRN.EQ.0)THEN
                  CALL MZBOOK(JQPDVS,LDIR,LRZ,-1,'RZ  ',6,6,LREF,2,-1)
                  LRZ=LDIR
                  CALL RZIODO(LUNF,LREF,IRS,IQ(KQSP+LRZ+1),1)
                  IF(IQUEST(1).NE.0) GOTO 70
                  LDS=IQ(KQSP+LRZ+KLD)
                  IF(LDS.GT.IQ(KQSP+LRZ-1)) GOTO 100
                  IF(LDS.LE.0) GOTO 100
                  NRDS=IQ(KQSP+LRZ+LDS)
                  IF(NRDS.GT.1)THEN
                     CALL MZPUSH(JQPDVS,LRZ,0,LREF*(NRDS-1),' ')
                     LDIR=LRZ
                     IQUEST(20) = NRDS
                     IQUEST(21) = IRS
                     DO 30 IR=2,NRDS
                        IRS=IQ(KQSP+LRZ+LDS+IR)
                        JR = 20 + IR
                        IF(JR.LE.100) IQUEST(JR) = IRS
                        IF(IRS.LE.0.OR.IRS.GT.FQUOTA) GOTO 100
                        CALL RZIODO(LUNF,LREF,IRS,
     + IQ(KQSP+LRZ+(IR-1)*LREF+1),1)
                        IF(IQUEST(1).NE.0)GOTO 70
   30 CONTINUE
                  ENDIF
               ELSE
   40 IF(RZSAME(IHDIR,IQ(KQSP+LRN+1),4))THEN
                     LRZ = LRN
                     LDIR= LRN
                     GOTO 60
                  ELSE
                     LRN=LQ(KQSP+LRN)
                     GOTO 20
                  ENDIF
               ENDIF
               GOTO 60
            ENDIF
   50 CONTINUE
         GOTO 80
   60 CONTINUE
      CALL SBIT0(IQ(KQSP+LDIR),IQDROP)
      LT=LTEMP
      GOTO 110
   70 CONTINUE
      LDIR = 0
      IQUEST(1) = 1
      GOTO 110
   80 CALL RZPAFF(CHPAT,NLPAT,CHL)
   90 LDIR=0
      IQUEST(1) = 2 ! SWON: Write a message if "Unknown directory"
      IF(LOGLV.GE.-2.AND.IOPTQ.EQ.0)THEN
         WRITE(IQLOG,10000)CHROUT,CHL(1:LENOCC(CHL))
10000 FORMAT(1X,A,'. Unknown directory ',A)
      ENDIF
      GOTO 110
  100 CALL RZPAFF(CHPAT,NLPAT,CHL)
      IQUEST(1) = 3
      LDIR=0
      IF(LOGLV.GE.-2)THEN ! SWON: Write a message if RZ is in trouble
         WRITE(IQLOG,10100)CHROUT,CHL(1:LENOCC(CHL))
10100 FORMAT(1X,A,'. Directory overwritten ',A)
      ENDIF
  110 RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE FZIMTB
      COMMON /ZBCD/ IQNUM2(11),IQLETT(26),IQNUM(10), IQPLUS,IQMINS
     +, IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +, IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +, IQUNDE,IQCLSQ,IQAND, IQAT, IQQUES,IQOPSQ,IQGREA
     +, IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC, IQLOWL(26)
     +, IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV, IQILEG
     +, NQHOL0,NQHOLL(95)
      PARAMETER (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +, NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
                   EQUIVALENCE (LQFS,LQSYSS(4)), (LQFF,LQSYSR(4))
     +, (LQFI,LQSYSR(5)), (LQFX,LQSYSR(6))
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      COMMON /FZCI/ LUNI,LUNNI,IXDIVI,LTEMPI,IEVFLI
     +, MSTATI,MEDIUI,IFIFOI,IDAFOI,IACMOI,IUPAKI
     +, IADOPI,IACTVI,INCBPI,LOGLVI,MAXREI, ISTENI
     +, LBPARI, L4STOI,L4STAI,L4CURI,L4ENDI
     +, IFLAGI,NFASTI,N4SKII,N4RESI,N4DONI,N4ENDI
     +, IOPTIE,IOPTIR,IOPTIS,IOPTIA,IOPTIT,IOPTID
     +, IOPTIF,IOPTIG,IOPTIH,IOPTI2(4)
     +, IDI(2),IPILI(4),NWTXI,NWSEGI,NWTABI,NWBKI,LENTRI
     +, NWUHCI,IOCHI(16),NWUMXI,NWUHI,NWIOI
     +, NWRDAI,NRECAI,LUHEAI,JRETCD,JERROR,NWERR
      PARAMETER (JAUIOC=50, JAUSEG=68, JAUEAR=130)
      COMMON /FZCSEG/NQSEG,IQSEGH(2,20),IQSEGD(20),IQSGLU,IQSGWK
      COMMON /FZCOCC/NQOCC,IQOCDV(20),IQOCSP(20)
      DIMENSION ITOSOR(20), ISORDV(20), ISORSP(20)
      DIMENSION LSTAV(20), LENDV(20)
      EQUIVALENCE (LSTAV(1),IQUEST(60)), (LENDV(1),IQUEST(80))
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HFZIM, 4HTB  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IFLGAR = 0
      IF (NQSEG.LE.0) THEN
          NQSEG = 1
          NSOR = 1
          NOCC = 1
          ITOSOR(1) = 1
          IQSEGD(1) = JQDIVI
          ISORDV(1) = JQDIVI
          IQOCDV(1) = JQDIVI
          ISORSP(1) = NWBKI
          IQOCSP(1) = NWBKI
          GO TO 41
        ENDIF
      IF (LOGLVI.GE.3) WRITE (IQLOG,9016) (J,
     + IQSEGH(1,J),IQSEGH(2,J),IQSEGD(J),J=1,NQSEG)
 9016 FORMAT (1X/' FZIMTB-  Segment Selection Table as set by the user'
     F/(10X,I2,1X,2A4,Z17))
      LFISEG = LQFI + JAUSEG
      IF (3*NQSEG.NE.IQ(KQSP+LFISEG)) GO TO 715
      LSPACE = KQSP + LFISEG + 2*NQSEG
      DO 27 JS=1,NQSEG
      IXDIV = IQSEGD(JS)
      IF (IXDIV) 22, 23, 24
   22 IF (IXDIV.LT.-7) GO TO 714
      ITOSOR(JS) = -IQ(LSPACE+JS)
      GO TO 27
   23 JDIV = JQDIVI
      GO TO 25
   24 JDIV = JBYT (IXDIV,1,26)
      IF (JDIV.GT.20) GO TO 714
      JSTO = JBYT (IXDIV,27,4)
      IF (JSTO.NE.0) THEN
          IF (JSTO.NE.JQSTOR) GO TO 714
        ENDIF
      IF (JDIV.EQ.0) GO TO 23
      IF (JDIV.GT.JQDVLL) THEN
          IF (JDIV.LT.JQDVSY) GO TO 714
        ENDIF
   25 IQSEGD(JS) = JDIV
      ITOSOR(JS) = 0
   27 CONTINUE
      NSOR = 0
      NOCC = 0
      JANX = 1
      JENX = NQSEG
   31 JDVBIG = 0
      JA = JANX
      JANX = 0
      JE = JENX
      JENX = 0
      DO 35 JS=JA,JE
      IF (ITOSOR(JS).NE.0) GO TO 35
      JENX = JS
      IF (JANX.EQ.0) JANX=JS
      JDIV = IQSEGD(JS)
      IF (JDIV.LE.JDVBIG) GO TO 35
      JDVBIG = JDIV
      JSBIG = JS
   35 CONTINUE
      IF (JDVBIG.EQ.0) GO TO 41
      NSOR = NSOR + 1
      ITOSOR(JSBIG) = NSOR
      ISORDV(NSOR) = JDVBIG
      ISORSP(NSOR) = IQ(LSPACE+JSBIG)
      NOCC = NOCC + 1
      IQOCDV(NOCC) = JDVBIG
      IQOCSP(NOCC) = IQ(LSPACE+JSBIG)
      IF (JSBIG.EQ.JENX) GO TO 31
      DO 37 JS=JSBIG+1,JENX
      IF (ITOSOR(JS).NE.0) GO TO 37
      IF (IQSEGD(JS).NE.JDVBIG) GO TO 37
      NSOR = NSOR + 1
      ITOSOR(JS) = NSOR
      ISORDV(NSOR) = JDVBIG
      ISORSP(NSOR) = IQ(LSPACE+JS)
      IQOCSP(NOCC) = IQOCSP(NOCC) + IQ(LSPACE+JS)
   37 CONTINUE
      GO TO 31
   41 IF (NOCC.EQ.0) GO TO 81
      JOCC = 0
   42 JOCC = JOCC + 1
      JQDIVI = IQOCDV(JOCC)
      NW = IQOCSP(JOCC)
      CALL MZRESV
      NQRESV = NQRESV - NW
      IF (NQRESV.LT.0) CALL MZGAR1
      IF (JQMODE.EQ.0) THEN
          IQLN = LQEND(KQT+JQDIVI)
          IQNX = IQLN + NW
          LQEND(KQT+JQDIVI) = IQNX
        ELSE
          IQNX = LQSTA(KQT+JQDIVI)
          IQLN = IQNX - NW
          LQSTA(KQT+JQDIVI) = IQLN
        ENDIF
      NQOCC = JOCC
      LQ(KQS+IQLN) = 12
      LQ(KQS+IQLN+1) = 0
      LQ(KQS+IQLN+2) = 0
      LQ(KQS+IQLN+3) = 0
      LQ(KQS+IQLN+5) = IQLETT(1)
      LQ(KQS+IQLN+6) = 0
      LQ(KQS+IQLN+7) = 0
      LQ(KQS+IQLN+8) = NW - 10
      LQ(KQS+IQLN+9) = 0
      IF (JOCC.NE.NOCC) GO TO 42
   46 NWTR = 2*NWTABI + 2
      NWTM = 8*NQSEG
      IF (NWTR+NWTM.LT.NQWKTB) THEN
          LQMTA = LQWKTB
          LQRTA = LQMTA + NWTM
        ELSE
          JQSTMV = -1
          CALL MZFGAP
          IF (NQGAPN.EQ.0) GO TO 61
          IF (IQGAP(1,1).LT.NWTR) THEN
              IF (NQWKTB.LT.NWTR) GO TO 61
              LQMTA = IQGAP(2,1)
              LQRTA = LQWKTB
            ELSE
              LQMTA = LQWKTB
              LQRTA = IQGAP(2,1)
            ENDIF
        ENDIF
      LQMTE = LQMTA + NWTM
      LQTA = LQRTA + 1
      LQTE = LQTA + 2*NWTABI
      LQRTE = LQTE
      JSOR = 1
      JOCC = 1
   52 JQDIVI = ISORDV(JSOR)
      IF (IQMODE(KQT+JQDIVI).EQ.0) THEN
          LSTA = LQEND(KQT+JQDIVI) - IQOCSP(JOCC)
        ELSE
          LSTA = LQSTA(KQT+JQDIVI)
        ENDIF
      LEND = LSTA + ISORSP(JSOR)
      LENDV(JSOR) = LEND
      LSTAV(JSOR) = LSTA
      JOCC = JOCC + 1
   54 IF (JSOR.EQ.NSOR) GO TO 55
      JSOR = JSOR + 1
      IF (ISORDV(JSOR).NE.JQDIVI) GO TO 52
      LSTA = LEND
      LEND = LSTA + ISORSP(JSOR)
      LENDV(JSOR) = LEND
      LSTAV(JSOR) = LSTA
      GO TO 54
   55 LMT = LQMTA
      DO 59 JS=1,NQSEG
      JSOR = ITOSOR(JS)
      IF (JSOR.GE.0) GO TO 57
      LQ(LMT) = 0
      LQ(LMT+1) = 0
      LQ(LMT+2) = 0
      LQ(LMT+3) = JSOR
      LQ(LMT+4) = JSOR
      LQ(LMT+5) = 0
      LQ(LMT+6) = 0
      LQ(LMT+7) = 0
      IF (LOGLVI.GE.3) WRITE (IQLOG,9055) JS, -JSOR
 9055 FORMAT (' FZIMTB-  skip segment',I3,I9,' WORDS')
      GO TO 59
   57 LQ(LMT) = ISORDV(JSOR)
      LQ(LMT+1) = 1
      LQ(LMT+2) = 0
      LQ(LMT+3) = LSTAV(JSOR)
      LQ(LMT+4) = LENDV(JSOR)
      LQ(LMT+5) = 0
      LQ(LMT+6) = 0
      LQ(LMT+7) = 0
      IF (LOGLVI.GE.3) THEN
          WRITE (IQLOG,9058) JS,LQ(LMT),LQ(LMT+3),LQ(LMT+4)
        ENDIF
 9058 FORMAT (' FZIMTB-  read segment',I3,' into division/from/to'
     F,I3,2I9)
   59 LMT = LMT + 8
  999 NQTRAC = NQTRAC - 2
      RETURN
   61 IF (IFLGAR.GE.2) GO TO 721
      IXSTOR = ISHFT (JQSTOR,26)
      IF (IFLGAR.NE.0) GO TO 63
      IXSTOR = MZIXCO (IXSTOR+21,22,23,24)
      CALL MZGARB (IXSTOR, 0)
      IFLGAR = 1
      IF (JQSTOR.NE.0) GO TO 46
      IFLGAR = 2
      GO TO 46
   63 IFLGAR = 2
      J = MZIXCO (21,22,23,24)
      CALL MZGARB (J, 0)
      CALL MZSDIV (IXSTOR,-7)
      GO TO 46
   81 NWBKI = 0
      JRETCD = -4
      IF (LOGLVI.GE.3) WRITE (IQLOG,9081)
 9081 FORMAT (' FZIMTB-  skip all segments')
      GO TO 999
  715 JERROR = 15
      IQUEST(14)= NQSEG
      IQUEST(15)= IQ(KQSP+LFISEG)
      NWERR = 2
      GO TO 719
  714 JERROR = 14
      IQUEST(14)= JS
      IQUEST(15)= 0
      IQUEST(16)= IXDIV
      NWERR = 3
  719 JRETCD = 4
      GO TO 999
  721 JERROR = 21
      JRETCD = 3
      GO TO 999
      END

*-------------------------------------------------------------------------------

      SUBROUTINE IZHNUM (HOLL,INTV,NP)
      INTEGER INTV(99), HOLL(99)
      DO 39 JWH=1,NP
   39 INTV(JWH) = IAND (HOLL(JWH), 255)
      RETURN
      END

*-------------------------------------------------------------------------------

      FUNCTION IUCOMP (ITEXT,IVECT,N)
      DIMENSION IVECT(9)
      IF (N.EQ.0) GO TO 18
      DO 12 J=1,N
      IF (ITEXT.EQ.IVECT(J)) GO TO 24
   12 CONTINUE
   18 J=0
   24 IUCOMP=J
      END

*-------------------------------------------------------------------------------

      SUBROUTINE IZBCDT (NP,ITABT)
      COMMON /QUEST/ IQUEST(100)
      PARAMETER (NQTCET=256)
      COMMON /ZCETA/ IQCETA(256),IQTCET(256)
      COMMON /ZKRAKC/IQHOLK(120), IQKRAK(80), IQCETK(122)
      DIMENSION NP(9), ITABT(99)
      N = NP(1)
      LIM = ITABT(1)
      JGOOD = 0
      JBAD = 0
      DO 29 JWH=1,N
      JV = IAND (IQHOLK(JWH),255)
      JV = IQTCET(JV+1)
      IF (JV.GT.LIM) GO TO 27
      JV = ITABT(JV+1)
      IF (JV+1) 29, 27, 24
   24 JGOOD = JGOOD + 1
      IQCETK(JGOOD) = JV
      GO TO 29
   27 JBAD = JBAD + 1
   29 CONTINUE
      IQUEST(1) = JGOOD
      IQUEST(2) = JBAD
      END

*-------------------------------------------------------------------------------

      LOGICAL FUNCTION RZSAME(IH1,IH2,N)
      DIMENSION IH1(N),IH2(N)
      IF(N.LE.0)GO TO 20
      DO 10 I=1,N
         IF(IH1(I).NE.IH2(I))GO TO 20
  10  CONTINUE
      RZSAME=.TRUE.
      GO TO 99
  20  RZSAME=.FALSE.
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZINK(KEYU,ICYCLE,CHOPT)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      COMMON /RZCOUT/IP1,IR1,IR2,IROUT,IRLOUT,IOPTRR
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     + KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     + KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     + KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     + KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      COMMON/RZCKEY/IHEAD(3),KEY(100),KEY2(100),KEYDUM(50)
      INTEGER KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON/RZCYCLE/KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     + KCNCYC, KNWCYC, KKYCYC, KVSCYC
      CHARACTER*(*) CHOPT
      DIMENSION KEYU(*)
      EQUIVALENCE (IOPTA,IQUEST(91)), (IOPTC,IQUEST(92))
     +, (IOPTD,IQUEST(93)), (IOPTN,IQUEST(94)), (IOPTR,IQUEST(95))
     +, (IOPTS,IQUEST(96))
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IQUEST(1)=0
      CALL UOPTC(CHOPT,'ACDNRS',IQUEST(91))
      LK=IQ(KQSP+LCDIR+KLK)
      NKEYS=IQ(KQSP+LCDIR+KNKEYS)
      NWKEY=IQ(KQSP+LCDIR+KNWKEY)
      IQUEST(7)=NKEYS
      IQUEST(8)=NWKEY
      IF(NKEYS.EQ.0)GO TO 90
      IF(IOPTS.NE.0)THEN
         IK1=KEYU(1)
         IK2=IK1
         IF(IK1.GT.NKEYS.OR.IK1.LE.0)THEN
            IQUEST(1)=1
            IQUEST(2)=IK1
            RETURN
         ENDIF
      ELSE
         IK1=1
         IK2=NKEYS
         DO 5 I=1,NWKEY
            IKDES=(I-1)/10
            IKBIT1=3*I-30*IKDES-2
            IF(JBYT(IQ(KQSP+LCDIR+KKDES+IKDES),IKBIT1,3).LT.3)THEN
               KEY(I)=KEYU(I)
            ELSE
               CALL ZHTOI(KEYU(I),KEY(I),1)
            ENDIF
   5  CONTINUE
      ENDIF
      DO 30 I=IK1,IK2
         LKC=LK+(NWKEY+1)*(I-1)
         IF(IOPTS.EQ.0)THEN
            DO 10 K=1,NWKEY
               IF(IQ(KQSP+LCDIR+LKC+K).NE.KEY(K))GO TO 30
  10  CONTINUE
         ELSE
            DO 15 K=1,NWKEY
               IF(K.LT.10)THEN
                  IKDES=(K-1)/10
                  IKBIT1=3*K-30*IKDES-2
                  IF(JBYT(IQ(KQSP+LCDIR+KKDES+IKDES),IKBIT1,3).LT.3)THEN
                     IQUEST(20+K)=IQ(KQSP+LCDIR+LKC+K)
                  ELSE
                     CALL ZITOH(IQ(KQSP+LCDIR+LKC+K),IQUEST(20+K),1)
                  ENDIF
               ENDIF
  15  CONTINUE
         ENDIF
         IQUEST(20)=I
         LCYC=IQ(KQSP+LCDIR+LKC)
*         IF (KVSCYC.NE.0) THEN
*            IF (IQ(KQSP+LCDIR+LCYC+KKYCYC).NE.IQ(KQSP+LCDIR+LKC+1)) THEN
*               IQUEST(1) = 11
*               GO TO 99
*            ENDIF
*         ENDIF
         NC=0
  20  NC=NC+1
         ICY = JBYT(IQ(KQSP+LCDIR+LCYC+KCNCYC),21,12)
         IF(ICY.EQ.ICYCLE)GO TO 50
         IF(NC.EQ.1.AND.ICYCLE.GT.ICY)GO TO 50
         IF (KVSCYC.EQ.0) THEN
            LCOLD = JBYT(IQ(KQSP+LCDIR+LCYC+KPPCYC),1,16)
         ELSE
            LCOLD = IQ(KQSP+LCDIR+LCYC+KPPCYC)
         ENDIF
         IF(LCOLD.EQ.0.AND.LCOLD.NE.LCYC.AND.ICYCLE.EQ.0)GO TO 50
         LCYC=LCOLD
         IF(LCYC.NE.0)GO TO 20
         GO TO 90
  30  CONTINUE
      GO TO 90
  50  IF (KVSCYC.EQ.0) THEN
         IR1 = JBYT(IQ(KQSP+LCDIR+LCYC+KFRCYC),17,16)
         IR2 = JBYT(IQ(KQSP+LCDIR+LCYC+KSRCYC),17,16)
         IP1 = JBYT(IQ(KQSP+LCDIR+LCYC+KORCYC), 1,16)
         NW = JBYT(IQ(KQSP+LCDIR+LCYC+KNWCYC), 1,20)
      ELSE
         IR1 = IQ(KQSP+LCDIR+LCYC+KFRCYC)
         IR2 = IQ(KQSP+LCDIR+LCYC+KSRCYC)
         IP1 = JBYT(IQ(KQSP+LCDIR+LCYC+KORCYC), 1,20)
         NW = IQ(KQSP+LCDIR+LCYC+KNWCYC)
      ENDIF
      N1 = NW
      IQUEST(2)=1
      IF(IR2.NE.0)IQUEST(2)=(NW-N1-1)/LREC+2
      IQUEST(3)=IR1
      IQUEST(4)=IP1
      IQUEST(5)=IR2
      IQUEST(6)=ICY
      IQUEST(12)=NW
      IQUEST(14)=IQ(KQSP+LCDIR+LCYC+1)
      IQUEST(15)=LCYC
      IF(IOPTC.NE.0)THEN
         IQUEST(50)=0
         LC1=LCYC
  51  IQUEST(50)=IQUEST(50)+1
         IF (KVSCYC.EQ.0) THEN
            LCOLD = JBYT(IQ(KQSP+LCDIR+LC1+KPPCYC),1,16)
         ELSE
            LCOLD = IQ(KQSP+LCDIR+LC1+KPPCYC)
         ENDIF
         IF(IQUEST(50).LE.19)THEN
            NC=IQUEST(50)
            IQUEST(50+NC)=JBYT(IQ(KQSP+LCDIR+LC1+KCNCYC),21,12)
            IQUEST(70+NC)=IQ(KQSP+LCDIR+LC1+KFLCYC)
         ENDIF
         IF(LCOLD.NE.0.AND.LCOLD.NE.LC1)THEN
            LC1=LCOLD
            GO TO 51
         ENDIF
      ENDIF
      IF(IOPTN.NE.0)THEN
         IF(I.EQ.1)THEN
            IQUEST(30)=0
         ELSE
            IQUEST(30)=NWKEY
            DO 52 J=1,NWKEY
               IF(J.LT.10)THEN
                  LKCJ=LK+(NWKEY+1)*(I-2)
                  IQUEST(30+J)=IQ(KQSP+LCDIR+LKCJ+J)
                  IKDES=(J-1)/10
                  IKBIT1=3*J-30*IKDES-2
                  IF(JBYT(IQ(KQSP+LCDIR+KKDES+IKDES),IKBIT1,3).GE.3)THEN
                     CALL ZITOH(IQUEST(30+J),IQUEST(30+J),1)
                  ENDIF
               ENDIF
  52  CONTINUE
         ENDIF
         IF(I.EQ.NKEYS)THEN
            IQUEST(40)=0
         ELSE
            IQUEST(40)=NWKEY
            DO 53 J=1,NWKEY
               IF(J.LT.10)THEN
                  LKCJ=LK+(NWKEY+1)*I
                  IQUEST(40+J)=IQ(KQSP+LCDIR+LKCJ+J)
                  IKDES=(J-1)/10
                  IKBIT1=3*J-30*IKDES-2
                  IF(JBYT(IQ(KQSP+LCDIR+KKDES+IKDES),IKBIT1,3).GE.3)THEN
                     CALL ZITOH(IQUEST(40+J),IQUEST(40+J),1)
                  ENDIF
               ENDIF
  53  CONTINUE
         ENDIF
      ENDIF
      GO TO 99
  90  IQUEST(1)=1
      IF(IOPTN.NE.0)THEN
         IF(NKEYS.GT.0)THEN
            IQUEST(30)=NWKEY
            IQUEST(40)=NWKEY
            DO 91 J=1,NWKEY
               IF(J.GE.10)GO TO 91
               LKCJ=LK+(NWKEY+1)*(NKEYS-1)
               IQUEST(30+J)=IQ(KQSP+LCDIR+LK+J)
               IQUEST(40+J)=IQ(KQSP+LCDIR+LKCJ+J)
               IKDES=(J-1)/10
               IKBIT1=3*J-30*IKDES-2
               IF(JBYT(IQ(KQSP+LCDIR+KKDES+IKDES),IKBIT1,3).GE.3)THEN
                  CALL ZITOH(IQUEST(30+J),IQUEST(30+J),1)
                  CALL ZITOH(IQUEST(40+J),IQUEST(40+J),1)
               ENDIF
  91  CONTINUE
         ENDIF
      ENDIF
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZPAFF(CH,NL,CHPATH)
      CHARACTER*(*) CHPATH,CH(*)
      CHARACTER*255 CHTEMP
      CHARACTER*16  CHL
      COMMON /QUEST/ IQUEST(100)
      MAXLEN=LEN(CHPATH)
      IF(MAXLEN.GT.255)MAXLEN=255
      IQUEST(1) = 0
      CHPATH='//'//CH(1)
      LENG=LENOCC(CHPATH)
      IF(LENG.EQ.2) THEN
        CHPATH='//HOME'
        LENG=6
      ENDIF
      IF(NL.EQ.1) GOTO 99
      DO 20 I=2,NL
         CHL=CH(I)
         NMAX=LENOCC(CHL)
         IF(NMAX.EQ.0) THEN
            IQUEST(1) = 1
            GOTO 99
         ENDIF
         IF(LENG+NMAX.GT.MAXLEN)NMAX=MAXLEN-LENG
         CHTEMP=CHPATH(1:LENG)//'/'//CHL(1:NMAX)
         CHPATH=CHTEMP
         LENG=LENG+NMAX+1
         IF(LENG.EQ.MAXLEN) THEN
            IQUEST(1) = 2
            GOTO 99
         ENDIF
  20  CONTINUE
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZPATH(CHPATH)
      CHARACTER  CQALLC*96
      COMMON /ZBCDCH/ CQALLC
                      CHARACTER*1  CQLETT(96), CQNUM(10)
                      EQUIVALENCE (CQLETT(1),CQALLC(1:1))
                      EQUIVALENCE (CQNUM(1), CQALLC(27:27))
      CHARACTER*1  BSLASH,KTILDE
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      PARAMETER (NLPATM=100)
      COMMON /RZDIRN/NLCDIR,NLNDIR,NLPAT
      COMMON /RZDIRC/CHCDIR(NLPATM),CHNDIR(NLPATM),CHPAT(NLPATM)
      CHARACTER*16   CHNDIR,    CHCDIR,    CHPAT
      CHARACTER*(*) CHPATH
      CHARACTER*1 CH1
      CHARACTER*2 CH2
      BSLASH=CQALLC(61:61)
      KTILDE=CQALLC(94:94)
      NCHP=LEN(CHPATH)
      NLPAT=0
  10  IF(CHPATH(NCHP:NCHP).EQ.' ')THEN
         NCHP=NCHP-1
         IF(NCHP.GT.0)GO TO 10
         NLPAT=NLCDIR
         DO 20 I=1,NLCDIR
            CHPAT(I)=CHCDIR(I)
  20     CONTINUE
         GO TO 99
      ENDIF
      IS1=1
  30  IF(CHPATH(IS1:IS1).EQ.' ')THEN
         IS1=IS1+1
         GO TO 30
      ENDIF
      CH1=CHPATH(IS1:IS1)
      CH2=CHPATH(IS1:IS1+1)
      IF(CH1.EQ.'/')THEN
         IF(CHPATH(IS1+1:IS1+1).EQ.'/')THEN
            IS=IS1+2
            IF(IS.GT.NCHP)GO TO 99
  40        IF(CHPATH(IS:IS).EQ.'/')THEN
               IF(IS.EQ.IS1+2)GO TO 90
               NLPAT=1
               CHPAT(1)=CHPATH(IS1+2:IS-1)
               IS1=IS+1
               IS=IS1
               GO TO 50
            ELSE
               IS=IS+1
               IF(IS.LT.NCHP)GO TO 40
               NLPAT=1
               CHPAT(1)=CHPATH(IS1+2:IS)
               GO TO 99
            ENDIF
         ENDIF
         IF(CHPATH(IS1+1:IS1+1).EQ.BSLASH)GO TO 90
         IF(CHPATH(IS1+1:IS1+1).EQ.KTILDE)GO TO 90
         NLPAT=1
         CHPAT(1)=CHCDIR(1)
         IS=IS1+1
         IS1=IS
  50     IF(IS.EQ.NCHP)THEN
            IF(CHPATH(IS1:IS).NE.'..'.AND.
     +         CHPATH(IS1:IS).NE.BSLASH) THEN
               NLPAT=NLPAT+1
               IF(NLPAT.GT.NLPATM)GO TO 90
               CHPAT(NLPAT)=CHPATH(IS1:IS)
            ELSE
               NLPAT = NLPAT -1
            ENDIF
            GO TO 99
         ELSE
            IF(CHPATH(IS:IS).EQ.'/')THEN
               IF(NLPAT.GT.NLPATM)GO TO 90
               IF(CHPATH(IS1:IS-1).NE.'..'.AND.
     +            CHPATH(IS1:IS-1).NE.BSLASH) THEN
                  NLPAT=NLPAT+1
                  CHPAT(NLPAT)=CHPATH(IS1:IS-1)
               ELSE
                  NLPAT = NLPAT - 1
               ENDIF
               IS1=IS+1
            ENDIF
            IS=IS+1
            GO TO 50
         ENDIF
      ENDIF
      IF(CH1.EQ.KTILDE)THEN
         NLPAT=NLNDIR
         DO 60 I=1,NLNDIR
            CHPAT(I)=CHNDIR(I)
  60     CONTINUE
         IF(IS1.EQ.NCHP)GO TO 99
         IS1=IS1+1
         CH1=CHPATH(IS1:IS1)
         GO TO 75
      ENDIF
      DO 70 I=1,NLCDIR
         CHPAT(I)=CHCDIR(I)
  70  CONTINUE
      NLPAT=NLCDIR
  75  IF(CH1.EQ.BSLASH)THEN
         NLPAT=NLPAT-1
         IF(NLPAT.EQ.0)NLPAT=1
         IF(IS1.EQ.NCHP)GO TO 99
         IS1=IS1+1
         CH1=CHPATH(IS1:IS1)
         GO TO 75
      ENDIF
      IS=IS1
  76  IF(CH2.EQ.'..')THEN
         NLPAT=NLPAT-1
         IF(NLPAT.EQ.0)NLPAT=1
         IF(IS1+1.EQ.NCHP)GO TO 99
         IF(CHPATH(IS1+2:IS1+2).NE.'/') GOTO 90
         IS =IS1
         IS1=IS1+3
         CH2=CHPATH(IS1:IS1+1)
         GO TO 76
      ENDIF
  80  IF(IS.EQ.NCHP)THEN
         NLPAT=NLPAT+1
         IF(NLPAT.GT.NLPATM)GO TO 90
         CHPAT(NLPAT)=CHPATH(IS1:IS)
         GO TO 99
      ELSE
         IF(CHPATH(IS:IS).EQ.'/')THEN
            IF(IS.GT.IS1)THEN
               NLPAT=NLPAT+1
               IF(NLPAT.GT.NLPATM)GO TO 90
               CHPAT(NLPAT)=CHPATH(IS1:IS-1)
            ENDIF
            IS1=IS+1
         ENDIF
         IS=IS+1
         GO TO 80
      ENDIF
  90  NLPAT=0
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZREAD(IV,N,IPC,IFORM)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/ LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +, LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +, IZRECL,IMODEC,IMODEH
      COMMON /RZCOUT/IP1,IR1,IR2,IROUT,IRLOUT,IOPTRR
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     + KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     + KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     + KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     + KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      COMMON /MZIOC/ NWFOAV,NWFOTT,NWFODN,NWFORE,IFOCON(3)
     +, MFOSAV(2), JFOEND,JFOREP,JFOCUR,MFO(200)
      DIMENSION IV(*)
      NL1=LREC-IP1+1
      IF(IPC.LE.NL1)THEN
         IRS=IR1
         IS1=IP1+IPC-1
      ELSE
         NBEF=(IPC-NL1-1)/LREC
         IRS=IR2+NBEF
         IS1 =IPC-NL1-NBEF*LREC
      ENDIF
      LRIN=LQ(KQSP+LTOP-7)
      IF(LRIN.EQ.0)THEN
         CALL MZBOOK(JQPDVS,LRIN,LTOP,-7,'RZIN',0,0,LREC+1,2,-1)
         IQ(KQSP+LRIN-5)=IQ(KQSP+LTOP-5)
         IQ(KQSP+LTOP+KIRIN)=0
         IRIN=0
      ELSE
         IRIN=IQ(KQSP+LTOP+KIRIN)
      ENDIF
      LROUT=LQ(KQSP+LTOP-6)
      IF(LROUT.EQ.0)THEN
         IROUT=0
      ELSE
         IROUT=IQ(KQSP+LTOP+KIROUT)
      ENDIF
      IF(IRS.NE.IRIN)THEN
         IF(IRS.NE.IROUT)THEN
            CALL RZIODO(LUN,LREC,IRS,IQ(KQSP+LRIN+1),1)
            IF(IQUEST(1).NE.0)GO TO 90
            IRIN=IRS
            IQ(KQSP+LTOP+KIRIN)=IRIN
         ENDIF
      ENDIF
      IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
         NWFOTT = N
         NWFODN = 0
         IF(IFORM.GT.0)THEN
            MFO(1) = IFORM
            MFO(2) = -1
            JFOEND = 2
         ENDIF
      ENDIF
      NLEFT=LREC-IS1+1
      IF(N.LE.NLEFT)THEN
         NP1=N
      ELSE
         NP1=NLEFT
      ENDIF
      IF(IRS.NE.IROUT)THEN
         IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
            NWFOAV=NP1
            CALL FZICV(IQ(KQSP+LRIN+IS1),IV)
            IF(NWFOAV.GT.0.OR.IFOCON(1).LT.0)GO TO 95
            IF(NWFOAV.LT.0)IDOUB1=IQ(KQSP+LRIN+IS1+NP1-1)
            IQUEST(1)=0
         ELSE
            CALL UCOPYI(IQ(KQSP+LRIN+IS1),IV,NP1)
         ENDIF
      ELSE
         IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
            NWFOAV=NP1
            CALL FZICV(IQ(KQSP+LROUT+IS1),IV)
            IF(NWFOAV.GT.0.OR.IFOCON(1).LT.0)GO TO 95
            IF(NWFOAV.LT.0)IDOUB1=IQ(KQSP+LROUT+IS1+NP1-1)
            IQUEST(1)=0
         ELSE
            CALL UCOPYI(IQ(KQSP+LROUT+IS1),IV,NP1)
         ENDIF
      ENDIF
      IF(NP1.LT.N)THEN
         NR=(N-NP1-1)/LREC+1
         IF(IRS.EQ.IR1)THEN
            IRS=IR2
         ELSE
            IRS=IRS+1
         ENDIF
         DO 60 I=1,NR
            IF(I.NE.NR)THEN
               IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
                  CALL RZIODO(LUN,LREC,IRS+I-1,IQ(KQSP+LRIN+1),1)
                  IF(IQUEST(1).NE.0)GO TO 90
                  IF(NWFOAV.LT.0)THEN
                     CALL UCOPY2(IQ(KQSP+LRIN+1),IQ(KQSP+LRIN+2),LREC)
                     IQ(KQSP+LRIN+1)=IDOUB1
                     NWFOAV=LREC+1
                     CALL FZICV(IQ(KQSP+LRIN+1),IV)
                     CALL UCOPY2(IQ(KQSP+LRIN+2),IQ(KQSP+LRIN+1),LREC)
                  ELSE
                     NWFOAV=LREC
                     CALL FZICV(IQ(KQSP+LRIN+1),IV)
                  ENDIF
                  IF(NWFOAV.GT.0.OR.IFOCON(1).LT.0)GO TO 95
                  IF(NWFOAV.LT.0)IDOUB1=IQ(KQSP+LRIN+LREC)
                  IQUEST(1)=0
               ELSE
                  print*,'>>>>>> RZIODO'
***               CALL RZIODO(LUN,LREC,IRS+I-1,V(NP1+1),1)
                  IF(IQUEST(1).NE.0)GO TO 90
               ENDIF
               NP1=NP1+LREC
            ELSE
               NL=N-NP1
               IRIN=IRS+I-1
               IF(IRIN.NE.IROUT)THEN
                  CALL RZIODO(LUN,LREC,IRIN,IQ(KQSP+LRIN+1),1)
                  IF(IQUEST(1).NE.0)GO TO 90
                  IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
                     IF(NWFOAV.LT.0)THEN
                       CALL UCOPY2(IQ(KQSP+LRIN+1),IQ(KQSP+LRIN+2),LREC)
                       IQ(KQSP+LRIN+1)=IDOUB1
                       NWFOAV=NL+1
                       CALL FZICV(IQ(KQSP+LRIN+1),IV)
                       CALL UCOPY2(IQ(KQSP+LRIN+2),IQ(KQSP+LRIN+1),LREC)
                     ELSE
                       NWFOAV=NL
                       CALL FZICV(IQ(KQSP+LRIN+1),IV)
                     ENDIF
                     IF(NWFOAV.GT.0.OR.IFOCON(1).LT.0)GO TO 95
                     IF(NWFOAV.LT.0)IDOUB1=IQ(KQSP+LRIN+NL)
                     IQUEST(1)=0
                  ELSE
                     CALL UCOPYI(IQ(KQSP+LRIN+1),IV(NP1+1),NL)
                  ENDIF
                  IQ(KQSP+LTOP+KIRIN)=IRIN
               ELSE
                 IF(IMODEX.GT.0.AND.IFORM.NE.1)THEN
                   IF(NWFOAV.LT.0)THEN
                     CALL UCOPY2(IQ(KQSP+LROUT+1),IQ(KQSP+LROUT+2),LREC)
                     IQ(KQSP+LROUT+1)=IDOUB1
                     NWFOAV=NL+1
                     CALL FZICV(IQ(KQSP+LROUT+1),IV)
                     CALL UCOPY2(IQ(KQSP+LROUT+2),IQ(KQSP+LROUT+1),LREC)
                   ELSE
                     NWFOAV=NL
                     CALL FZICV(IQ(KQSP+LROUT+1),IV)
                   ENDIF
                   IF(NWFOAV.GT.0.OR.IFOCON(1).LT.0)GO TO 95
                   IF(NWFOAV.LT.0)IDOUB1=IQ(KQSP+LROUT+NL)
                   IQUEST(1)=0
                 ELSE
                     CALL UCOPYI(IQ(KQSP+LROUT+1),IV(NP1+1),NL)
                  ENDIF
               ENDIF
            ENDIF
  60  CONTINUE
      ENDIF
  90  CONTINUE
      GO TO 99
  95  IQUEST(1) =4
      IQUEST(11)=NWFOTT
      IQUEST(12)=NWFORE
      IQUEST(13)=NWFOAV
      IQUEST(14)=NWFODN
      IF(JBYT(IQ(KQSP+LTOP),15,3)-3.GE.-2) WRITE(IQLOG,1000)
 1000 FORMAT(' RZREAD. Error during conversion into native format')
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZCHLN (IXST,LP)
      PARAMETER (IQBITW=32, IQBITC=8, IQCHAW=4)
      COMMON /ZMACH/ NQBITW,NQBITC,NQCHAW
     +, NQLNOR,NQLMAX,NQLPTH,NQRMAX,IQLPCT,IQNIL
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      DIMENSION IXST(9), LP(9)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IXSTOR = IXST(1)
      IQLN = LP(1)
      IF (IXSTOR.EQ.-7) GO TO 21
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR) CALL MZSDIV (IXSTOR,-7)
   21 IF (IQLN.LT.LQSTA(KQT+1)) GO TO 98
      IF (IQLN.GE.LQSTA(KQT+21)) GO TO 98
      IWD = LQ(KQS+IQLN)
      NST = JBYT (IWD,1,16) - 12
      IF (NST.LT.0) GO TO 41
      IQLS = IQLN + NST + 1
      IF (IQLS.GE.LQSTA(KQT+21)) GO TO 91
      IQNIO = JBYT (IQ(KQS+IQLS),19,4)
      IQID = IQ(KQS+IQLS-4)
      IQNL = IQ(KQS+IQLS-3)
      IQNS = IQ(KQS+IQLS-2)
      IQND = IQ(KQS+IQLS-1)
      IF ( JBYT(IQNL,IQBITW-3,4)
     + + JBYT(IQNS,IQBITW-3,4)
     + + JBYT(IQND,IQBITW-3,4) .NE.0) GO TO 91
      IQNX = IQLS + IQND + 9
      IF (IQNX.GT.LQSTA(KQT+21)) GO TO 91
      IF (IQNS.GT.IQNL) GO TO 91
      IF (NST.NE.IQNIO+IQNL) GO TO 91
      IQFOUL = 0
      RETURN
   41 NWD = JBYT (IWD,17,IQDROP-17)
      IQLS = IQLN - 8
      IQNX = IQLN + NWD
      IQND = -NWD
      IF (NWD.EQ.0) GO TO 91
      IF (NWD.NE.NST+12) GO TO 91
      NST = JBYT (IWD,IQDROP,IQBITW-IQDROP)
      IF (NST.NE.1) GO TO 91
      IQFOUL= 0
      RETURN
   91 IQFOUL = 7
      RETURN
   98 IQFOUL = -7
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZCHNB (LIX)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION LIX(9)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZCH, 4HNB  /
      K = LOCF (LIX(1)) - LQSTOR
      IF (K.LT.LQSTA(KQT+1)) RETURN
      IF (K.GE.LQEND(KQT+20)) RETURN
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      NQCASE = 1
      NQFATA = 2
      IQUEST(11) = K
      IQUEST(12) = LIX(1)
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZDROP (IXSTOR,LHEADP,CHOPT)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      DIMENSION LHEADP(9)
      CHARACTER *(*) CHOPT
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZDR, 4HOP  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      LHEAD = LHEADP(1)
      IF (LHEAD.EQ.0) RETURN
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR) CALL MZSDIV (IXSTOR,-7)
      CALL UOPTC (CHOPT,'LV',IQUEST)
      IFLAG = IQUEST(1)
      IF (IQUEST(2).NE.0) IFLAG=-1
      CALL MZCHLS (-7,LHEAD)
      IF (IQFOUL.NE.0) GO TO 91
      IF (NQLOGL.LT.2) GO TO 19
      WRITE (IQLOG,9018) JQSTOR,LHEAD,IQID,CHOPT
 9018 FORMAT (' MZDROP-  Store',I3,' L/ID=',I9,1X,A4,' Opt=',A)
   19 KHEAD = LQ(KQS+LHEAD+2)
   21 IF (IFLAG) 22, 31, 41
   22 NS = IQNS
      CALL MZFLAG (IXSTOR,LHEAD,IQDROP,'V')
      CALL VZEROI (LQ(KQS+LHEAD-NS),NS)
      GO TO 999
   31 CALL MZFLAG (IXSTOR,LHEAD,IQDROP,'.')
      LN = LQ(KQS+LHEAD)
      IF (LN.EQ.0) GO TO 88
      IF (LN.EQ.LHEAD) GO TO 88
      CALL MZCHLS (-7,LN)
      IF (IQFOUL.NE.0) GO TO 92
      IF (KHEAD.NE.0) LQ(KQS+KHEAD)=LN
      LQ(KQS+LN+2) = KHEAD
      GO TO 999
   41 CALL MZFLAG (IXSTOR,LHEAD,IQDROP,'L')
   88 IF (KHEAD.NE.0) LQ(KQS+KHEAD)=0
  999 NQTRAC = NQTRAC - 2
      RETURN
   92 NQCASE = 1
      NQFATA = 1
      IQUEST(12) = LN
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 1
      IQUEST(11) = LHEAD
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      FUNCTION MZDVAC (IXDIVP)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION IXDIVP(9)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZDV, 4HAC  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      JBYTET (MZ,IZW,IZP,NZB) = IAND (MZ,
     + ISHFT (ISHFT(IZW,33-IZP-NZB),-(32-NZB)) )
      IXIN = IXDIVP(1)
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      JST = JBYT (IXIN,27,6)
      IF (JST.EQ.JQSTOR) GO TO 31
      IF (JST-16.EQ.JQSTOR) GO TO 21
      CALL MZSDIV (IXIN,-7)
      IF (JST.LT.16) GO TO 31
   21 IXAC = JBYT (IXIN,1,20)
      IXGE = JBYT (IXIN,21,6)
      IF (IXGE.EQ.0) GO TO 59
      IF (IXGE.LT.16) GO TO 41
   29 CALL MZSDIV (IXIN,0)
   31 JDIV = JBYT (IXIN,1,26)
      IF (JDIV.GE.25) GO TO 29
      IXAC = 0
      IF (JDIV.GE.21) GO TO 33
      IXAC = MSBIT1 (IXAC,JDIV)
      GO TO 59
   33 IXGE = MSBIT1 (0, JDIV-20)
   41 JDIV = 1
   42 IF (JDIV.EQ.JQDVLL+1) JDIV=JQDVSY
      IF (JBYTET(IXGE,IQKIND(KQT+JDIV),21,4).EQ.0) GO TO 47
      IXAC = MSBIT1 (IXAC,JDIV)
   47 JDIV = JDIV + 1
      IF (JDIV.LT.21) GO TO 42
   59 MZDVAC = IXAC
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZGARB (IXGP,IXWP)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION IXGP(1), IXWP(9)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZGA, 4HRB  /
      IXGARB = IXGP(1)
      IXWIPE = IXWP(1)
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      JVLEV = 2
      MQDVGA = 0
      MQDVWI = 0
      IF (IXGARB.EQ.0) GO TO 16
      JVLEV = 1
      MQDVGA = MZDVAC (IXGARB)
      IF (IXWIPE.EQ.0) GO TO 19
      JSTO = JQSTOR
      MQDVWI = MZDVAC (IXWIPE)
      IF (JSTO.NE.JQSTOR) GO TO 91
      GO TO 19
   16 MQDVWI = MZDVAC (IXWIPE)
   19 IF (MQDVGA+MQDVWI.EQ.0) GO TO 999
      NQRESV = 0
      JQSTMV = -1
      IF (NQLOGL.LT.1) GO TO 24
      IF (MQDVGA.NE.0) GO TO 22
      IF (NQLOGL.LT.2) GO TO 24
   22 WRITE (IQLOG,9022) JQSTOR,MQDVGA,MQDVWI
 9022 FORMAT (' MZGARB-  User Garb.C./Wipe for store',I3,', Divs',
     F2(2X,Z6))
      IQVREM(1,JVLEV) = IQVID(1)
      IQVREM(2,JVLEV) = IQVID(2)
   24 CALL MZTABM
      CALL MZTABR
      CALL MZTABX
      CALL MZTABF
      IF (NQNOOP.NE.0) GO TO 999
      CALL MZGSTA (NQDGAU(KQT+1))
      CALL MZRELX
      CALL MZMOVE
      IF (IQPART.NE.0) GO TO 24
  999 NQTRAC = NQTRAC - 2
      RETURN
   91 NQCASE = 1
      NQFATA = 2
      IQUEST(11) = JSTO
      IQUEST(12) = JQSTOR
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZGAR1
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZGA, 4HR1  /
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IQVREM(1,1) = IQVID(1)
      IQVREM(2,1) = IQVID(2)
      MQDVGA = 0
      MQDVWI = 0
      IF (JQDIVI.LT.3) GO TO 24
      MQDVGA = MSBIT1 (0,JQDIVI)
      JQDVM2 = JQDIVI - JQMODE
      IF (JQDVM2.EQ.JQDVSY-1) JQDVM2=JQDVLL
      JQDVM1 = 2
      JQSTMV = JQSTOR
      IQTNMV = 0
      IF (JQSHAR.EQ.0) GO TO 29
      MQDVGA = MSBIT1 (MQDVGA,JQSHAR)
      GO TO 29
   24 MQDVGA = 3
      JQSTMV = -1
   29 NQDVMV = 0
      NRESAV = NQRESV
      IF (NQLOGL.GE.1) WRITE (IQLOG,9028) MQTRAC(NQTRAC-3),
     + MQTRAC(NQTRAC-2),JQSTOR,JQDIVI,NQRESV
 9028 FORMAT (' MZGAR1-  Auto Garbage Collection called from ',2A4,
     F' for Store/Div',2I3,' Free',I7)
      CALL MZTABM
      CALL MZTABR
      NQRESV = NQRESV + NQFREE
      IF (NQRESV.GE.0) GO TO 51
      IF (IQPART.NE.0) GO TO 51
      IF (JQDIVI.LT.3) GO TO 72
      NRESV1 = LQSTA(KQT+2) - LQEND(KQT+1) - NQMINR
      NRESV1 = MIN (NRESV1,LQEND(KQT+2)-LQ2END)
      IF (JQMODE.NE.0) GO TO 34
      IF (JQSHAR.NE.0) THEN
          NPOSSH = NQDMAX(KQT+JQDIVI) + NQDMAX(KQT+JQDIVN)
     + -(LQEND(KQT+JQDIVN) - LQSTA(KQT+JQDIVI))
          GO TO 36
        ELSE
          NPOSSH = LQSTA(KQT+JQDIVI) + NQDMAX(KQT+JQDIVI)
     + - LQSTA(KQT+JQDIVN)
          GO TO 36
        ENDIF
   34 IF (JQSHAR.NE.0) THEN
          NPOSSH = NQDMAX(KQT+JQDIVI) + NQDMAX(KQT+JQDIVN)
     + -(LQEND(KQT+JQDIVI) - LQSTA(KQT+JQDIVN))
        ELSE
          NPOSSH = LQEND(KQT+JQDIVN)
     + - (LQEND(KQT+JQDIVI) - NQDMAX(KQT+JQDIVI))
        ENDIF
   36 NSH = (LQEND(KQT+JQDIVI)-LQSTA(KQT+JQDIVI)) / 8
      NSH = MAX (NSH,24) - NQRESV
      NSH = MIN (NSH, NPOSSH, NRESV1)
      IF (NSH+NQRESV.LT.0) GO TO 72
      NQRESV = NQRESV + NSH
      NQDVMV = - NSH
      CALL MZTABS
   51 NWIN = NQRESV - NRESAV
      IF (NQLOGL.GE.1) WRITE (IQLOG,9051) NWIN,NQDVMV
 9051 FORMAT (10X,'Wins',I7,' words, Shift by',I7)
      CALL MZTABX
      CALL MZTABF
      IF (NQNOOP) 68, 53, 67
   53 CALL MZGSTA (NQDGAF(KQT+1))
      CALL MZRELX
   67 CALL MZMOVE
   68 IF (NQRESV.LT.0) GO TO 71
  999 NQTRAC = NQTRAC - 2
      RETURN
   71 IF (IQPART.NE.0) GO TO 29
   72 IQUEST(11) = NQRESV
      IQUEST(12) = JQSTOR
      IQUEST(13) = JQDIVI
      IF (NQLOGL.GE.1) WRITE (IQLOG,9072) NQRESV
 9072 FORMAT (10X,'Not enough space, Free',I7)
      IF (NQPERM.NE.0) GO TO 999
      IF (JQKIND.NE.1) GO TO 91
      print*,'>>>>>> CALL ZTELL (99,1)'
   91 NQCASE = 1
      NQFATA = 1
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZFORM (CHID,CHFORM,IXIOP)
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      COMMON /ZKRAKC/IQHOLK(120), IQKRAK(80), IQCETK(122)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      EQUIVALENCE (LQFORM,LQSYSS(5))
      EQUIVALENCE (NW,IQUEST(1))
      DIMENSION IXIOP(99)
      CHARACTER CHID*(*), CHFORM*(*)
      DIMENSION MMID(5), MMIX(5), MMIO(5)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZFO, 4HRM  /
      DATA MMID / 4HQID , 2, 2, 10, 5 /
      DATA MMIX / 4HQIOX, 0, 0, 7, 2 /
      DATA MMIO / 4HQIOD, 0, 0, 50, 1 /
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     + IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     + ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      N = MIN (4, LEN(CHID))
      CALL UCTOH (CHID,IDH,4,N)
      IF (LQFORM.EQ.0) GO TO 75
   12 IQCETK(121) = IDH
      LIOD = LQ(KQSP+LQFORM-2)
      IXIOD = IQ(KQSP+LIOD+1)
      CALL MZIOCH (IQ(KQSP+LIOD+IXIOD+1),16,CHFORM)
      NW = NW + 1
      NWIO = IXIOD + NW
      IQ(KQSP+LIOD+1) = NWIO
      NFRIO = IQ(KQSP+LIOD-1) - NWIO
      LID = LQFORM
      IF (IDH.LT.0) LID=LQ(KQSP+LID)
      LIX = LQ(KQSP+LID-1)
      NWID = IQ(KQSP+LID+1) + 1
      IQ(KQSP+LID+1) = NWID
      IQ(KQSP+LID+NWID+3) = IDH
      IQ(KQSP+LIX+NWID) = IXIOD
      NFRID = IQ(KQSP+LID-1) - NWID - 3
      IQUEST(2) = 64*NW + 2
      IQUEST(2) = MSBYT (IXIOD,IQUEST(2),17,15)
      IXIOP(1) = IQUEST(2)
      IF (NFRID.EQ.0) GO TO 71
   28 IF (NFRIO.LT.16) GO TO 73
   29 CONTINUE
  999 NQTRAC = NQTRAC - 2
      RETURN
   71 CALL MZPUSH (JQPDVS,LID,0,20,'I')
      LIX = LQ(KQSP+LID-1)
      CALL MZPUSH (JQPDVS,LIX,0,20,'I')
      GO TO 28
   73 LIOD = LQ(KQSP+LQFORM-2)
      CALL MZPUSH (JQPDVS,LIOD,0,60,'I')
      GO TO 29
   75 CONTINUE
      DO 76 J=1,2
      CALL MZLIFT (JQPDVS,L,LQFORM,1,MMID,0)
      CALL MZLIFT (JQPDVS,LIX,L,-1,MMIX,0)
   76 CONTINUE
      CALL MZLIFT (JQPDVS,L,LQFORM,-2,MMIO,0)
      IQ(KQSP+L+1) = 1
      GO TO 12
      END

*-------------------------------------------------------------------------------

      FUNCTION MZFDIV (IXST,LIXP)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION IXST(9), LIXP(9)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IXSTOR = IXST(1)
      LIX = LIXP(1)
      IF (IXSTOR.NE.-7) THEN
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR) CALL MZSDIV (IXSTOR,-7)
          JDIVI = 2
        ELSE
          JDIVI = JQDIVI
        ENDIF
      IF (JDIVI.EQ.0) GO TO 21
      IF (LIX.LT.LQSTA(KQT+JDIVI)) GO TO 21
      IF (LIX.LT.LQEND(KQT+JDIVI)) GO TO 99
   21 JDIVI = 1
      IF (LIX.LT.LQEND(KQT+JQDVLL)) GO TO 24
      IF (LIX.GE.LQEND(KQT+20)) GO TO 91
      JDIVI = JQDVSY
   24 IF (LIX.LT.LQEND(KQT+JDIVI)) GO TO 26
      JDIVI = JDIVI + 1
      GO TO 24
   26 IF (LIX.GE.LQSTA(KQT+JDIVI)) GO TO 99
   91 JDIVI = 0
   99 MZFDIV = JDIVI
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZFGAP
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NGAPV(7), JDIVV(7), JSTOV(7), JPV(7)
      EQUIVALENCE (NGAPV(1),IQUEST(11)), (JDIVV(1),IQUEST(21))
      EQUIVALENCE (JSTOV(1),IQUEST(31)), (JPV(1), IQUEST(41))
      DIMENSION NQGAPV(2)
      EQUIVALENCE (NQGAPV(1),NQGAPN)
      CALL VZEROI (IQGAP,20)
   15 DO 17 J=1,6
   17 NGAPV(J) = 0
      IF (JQSTMV.LT.0) GO TO 19
      KT = NQOFFT(JQSTMV+1)
      JDVSH1 = JQDVM1
      JDVSH2 = JQDVM2
      IF (NQDVMV.GT.0) GO TO 19
      IF (JDVSH1.EQ.IQTABV(KT+9)) JDVSH1=IQTABV(KT+8)+1
      JDVSH1 = JDVSH1 - 1
      JDVSH2 = JDVSH2 - 1
   19 MINGN = 0
      MINGV = 0
      JMINGN = 1
      JMINGV = 5
      JSTO = -1
   21 JSTO = JSTO + 1
      IF (JSTO.GT.NQSTOR) GO TO 61
      IF (NQALLO(JSTO+1).LT.0) GO TO 21
      KT = NQOFFT(JSTO+1)
      JDVN = 1
   31 JDIV = JDVN
      IF (JDIV.EQ.21) GO TO 21
      JDVN = JDIV + 1
      IF (JDIV.EQ.IQTABV(KT+8)) JDVN=IQTABV(KT+9)
      NWGAP = LQSTA(KT+JDVN) - LQEND(KT+JDIV)
      IF (NWGAP.LT.164) GO TO 31
      IF (JSTO.NE.JQSTMV) GO TO 41
      IF (JDIV.LT.JDVSH1) GO TO 41
      IF (JDIV.GT.JDVSH2) GO TO 41
      IF (NWGAP.LE.MINGV) GO TO 31
      NGAPV(JMINGV) = NWGAP
      JDIVV(JMINGV) = JDIV
      JSTOV(JMINGV) = JSTO
      JMINGV = 5
      MINGV = NGAPV(5)
      IF (MINGV.LE.NGAPV(6)) GO TO 31
      JMINGV = 6
      MINGV = NGAPV(6)
      GO TO 31
   41 IF (NWGAP.LE.MINGN) GO TO 31
      NGAPV(JMINGN) = NWGAP
      JDIVV(JMINGN) = JDIV
      JSTOV(JMINGN) = JSTO
      JMINGN = 1
      MINGN = NGAPV(1)
      DO 44 J=2,4
      IF (MINGN.LE.NGAPV(J)) GO TO 44
      JMINGN = J
      MINGN = NGAPV(J)
   44 CONTINUE
      GO TO 31
   61 DO 62 J=1,6
   62 JPV(J) = J
      JG = 1
   65 JF = JPV(JG)
      JN = JPV(JG+1)
      IF (NGAPV(JF).LT.NGAPV(JN)) GO TO 67
      IF (JG.EQ.3) GO TO 71
   66 JG = JG + 1
      GO TO 65
   67 JPV(JG) = JN
      JPV(JG+1) = JF
      IF (JG.EQ.1) GO TO 66
      JG = JG - 1
      GO TO 65
   71 JG = 4
   75 JF = JPV(JG)
      JN = JPV(JG+1)
      IF (NGAPV(JF).LT.NGAPV(JN)) GO TO 77
      IF (JG.EQ.5) GO TO 81
   76 JG = JG + 1
      GO TO 75
   77 JPV(JG) = JN
      JPV(JG+1) = JF
      IF (JG.EQ.3) GO TO 76
      JG = JG - 1
      GO TO 75
   81 NQGAPN = 0
      NQGAP = 0
      JSEL = 1
      DO 87 JG=1,4
      JU = JPV(JG)
      NWGAP= NGAPV(JU)
      IF (NWGAP.EQ.0) GO TO 87
      JDIV = JDIVV(JU)
      JSTO = JSTOV(JU)
      KT = NQOFFT(JSTO+1)
      KS = NQOFFS(JSTO+1)
      IQGAP(1,JG) = NWGAP
      IQGAP(2,JG) = KS+ LQEND(KT+JDIV)
      IQGAP(3,JG) = JDIV
      IQGAP(4,JG) = JSTO
      IF (JU.GE.5) JSEL=2
      NQGAPV(JSEL) = JG
   87 CONTINUE
      NQGAP = MAX (NQGAPN,NQGAP)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABC
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      EQUIVALENCE (LS,IQLS), (LNX,IQNX)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZTA, 4HBC  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      NQFRTC = 0
      NQLIVE = 0
      N = 0
      LNX = LQMTC1
      LQTE = LQTC1
      NEW = IQTVAL
      LQ(LQTE) = LNX
      LQ(LQTE+2) = 0
   21 MODE = NEW
   22 LN = LNX
      IF (LN.GE.LQMTC2) GO TO 41
      N = N + 1
      CALL MZCHLN (-7,LN)
      IF (IQFOUL.NE.0) GO TO 91
      NEW = JBIT (IQ(KQS+LS),IQTBIT)
      IF (NEW.EQ.MODE) GO TO 22
      IF (NEW.EQ.IQTVAL) GO TO 36
      NQLIVE = NQLIVE + N - 1
      LQ(LQTE+1) = LN
      LQ(LQTE+3) = 1
      LQTE = LQTE + 4
      GO TO 21
   36 NQFRTC = NQFRTC + (LN - LQ(LQTE-3))
      LQ(LQTE) = LN
      LQ(LQTE+2) = 0
      N = 1
      IF (LQTE.LT.LQTC2) GO TO 21
      CALL MZTABH
      IF (IQPART.EQ.0) GO TO 21
      IQPART = 1
      LN = LQMTC2
   41 IF (NEW.NE.IQTVAL) GO TO 43
      NQLIVE = NQLIVE + N
      LQ(LQTE+1) = LN
      LQ(LQTE+3) = 0
      GO TO 45
   43 NQFRTC = NQFRTC + (LN-LQ(LQTE-3))
      LQ(LQTE) = LN
      LQ(LQTE+1) = LN
      LQ(LQTE+2) = 0
      LQ(LQTE+3) = 0
   45 LQTE = LQTE + 4
  999 NQTRAC = NQTRAC - 2
      RETURN
   91 NQCASE = 1
      NQFATA = 3
      IQUEST(11) = LN
      IQUEST(12) = LQMTC1
      IQUEST(13) = LQMTC2
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABF
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZTA, 4HBF  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LMT = LQMTA
      NCOLL = 0
      NGARB = 0
      NQNOOP = 0
      LFIXLO = NQLINK + 1
   21 JDIV = LQ(LMT)
      IACT = LQ(LMT+1)
      IF (IACT.EQ.4) GO TO 26
      IF (IACT.GE.2) GO TO 28
      IF (IACT.GE.0) LFIXLO=LQEND(KQT+JDIV)
      LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 21
      NQNOOP = -7
      IF (NQDVMV.EQ.0) GO TO 81
      NQNOOP = 7
      GO TO 81
   26 IF (LQ(LMT+9).NE.4) GO TO 28
      IF (LMT+8.GE.LQMTE) GO TO 28
      LMT = LMT + 8
      GO TO 26
   28 LFIR = LMT
      LQTA = LQRTA + LQ(LMT+5)
   31 IACT = LQ(LMT+1)
      NSHF = LQ(LMT+2)
      LTU = LQRTA + LQ(LMT+5)
      IF (IACT.EQ.4) GO TO 71
      IF (IACT.EQ.3) GO TO 61
      IF (IACT.EQ.2) GO TO 41
      IF (IACT.LT.0) GO TO 79
      NCOLL = NCOLL + 1
      IF (NCOLL.NE.1) GO TO 79
      LCOLL = LMT
      GO TO 79
   41 IF (NCOLL+NGARB.LT.2) GO TO 49
   43 LCOLE = LMT - 8
      LT = LQ(LCOLE+5)
      LTF = LQ(LCOLL+5)
      N = LT - LTF
      NW = LQRTA + LTF+1 - LQTA
      CALL UCOPY2 (LQ(LQTA),LQ(LQTA+N),NW)
      LQTA = LQTA + N
      NCOLL = 0
      IF (IACT.EQ.4) GO TO 71
      IF (IACT.EQ.3) GO TO 61
   49 LQ(LTU+2) = NSHF
      GO TO 77
   61 IF (NCOLL+NGARB.GE.2) GO TO 43
      JDIV = LQ(LMT)
      LT = LTU
      LTE = LQRTA + LQ(LMT+6)
      MODE = JBIT (IQMODE(KQT+JDIV),1)
      IF (MODE.NE.0) GO TO 65
      NCUM = NSHF
      GO TO 66
   65 NCUM = LQ(LMT+7) + NSHF
   66 LQ(LT+2) = NCUM
      NCUM = NCUM - (LQ(LT+4)-LQ(LT+1))
      LT = LT + 4
      IF (LT.LT.LTE) GO TO 66
      NGARB = -64
      GO TO 77
   71 IF (NCOLL+NGARB.GE.2) GO TO 43
   77 NCOLL = 0
   79 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 31
      LQ(LQTA-1) = LFIXLO
      IF (NCOLL.EQ.0) GO TO 81
      LQTE = LQRTA + LQ(LCOLL+5)
   81 CONTINUE
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABH
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZTA, 4HBH  /
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (JQGAPM.NE.0) GO TO 41
      CALL MZFGAP
      NW = LQMTE+1 - LQMTA
      JQGAPM = NQGAPN
      IF (JQGAPM.LT.2) GO TO 26
   23 LNEW = IQGAP(2,JQGAPM)
      NSH = LNEW - LQMTA
      CALL UCOPYI (LQ(LQMTA),LQ(LNEW),NW)
      LQMTA = LNEW
      LQMTB = LQMTB + NSH
      LQMTE = LQMTE + NSH
      LQTC2 = LQTC2 + 161
      LQRTE = LQRTE + 161
      IQGAP(1,JQGAPM) = IQGAP(1,JQGAPM) - NW
      IQGAP(2,JQGAPM) = IQGAP(2,JQGAPM) + NW
  999 NQTRAC = NQTRAC - 2
      RETURN
   26 IF (IQTNMV.EQ.0) JQGAPM=NQGAP
      IF (JQGAPM.NE.0) GO TO 23
      IF (IQTNMV.LT.0) GO TO 31
   29 IQPART = 7
      GO TO 999
   31 JQGAPM = NQGAP
      IF (JQGAPM.EQ.0) GO TO 29
      IQPART = -7
      GO TO 23
   36 IF (IQTNMV.GE.0) GO TO 29
      IF (JQGAPR.GT.NQGAPN) GO TO 29
      JQGAPR = NQGAP
      IF (JQGAPR.EQ.0) GO TO 29
      IF (IQGAP(1,NQGAP-1).GT.IQGAP(1,NQGAP)) JQGAPR=NQGAP-1
      NNEW = IQGAP(1,JQGAPR) - (LQRTE-LQRTA) - 10
      IF (NNEW.LT.16) GO TO 29
      IQPART = -7
      GO TO 44
   41 IF (JQGAPR.NE.0) GO TO 36
      IF (NQGAPN.EQ.0) GO TO 36
      NNEW = IQGAP(1,1) - NQWKTB
      IF (NNEW.LT.16) GO TO 36
      JQGAPR = 1
   44 LNEW = IQGAP(2,JQGAPR)
      NSH = LNEW - LQRTA
      NW = LQTE+4 - LQRTA
      CALL UCOPYI (LQ(LQRTA),LQ(LNEW),NW)
      GO TO 999
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABM
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZTA, 4HBM  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LQMTBR = 0
      IQTBIT = IQDROP
      IQTVAL = 0
      NQFREE = 0
      IQPART = 0
      MQDVAC = 0
      IQFLIO = 0
      IF (JQSTMV.LT.0) THEN
          IQTNMV = 0
          JQDVM1 = 0
          JQDVM2 = 0
          NQDVMV = 0
        ENDIF
      JQGAPM = 0
      JQGAPR = 0
      LQMTE = LQWKTB + NQWKTB - 1
      LQMTA = LQMTE - 160
      LQMTB = LQMTA
      LQRTE = LQMTA - 10
      LQTC2 = LQRTE
      LQRTA = LQWKTB
      LQTA = LQRTA + 1
      LQTE = LQTA
      LQTC1 = LQTA
      LMT = LQMTA
      JDIV = 1
   32 LQ(LMT) = JDIV
      LQ(LMT+1) = 0
      LQ(LMT+2) = 0
      LQ(LMT+3) = LQSTA(KQT+JDIV)
      LQ(LMT+4) = LQEND(KQT+JDIV)
      LQ(LMT+5) = 0
      LQ(LMT+6) = 0
      LQ(LMT+7) = 0
      NW = LQ(LMT+4) - LQ(LMT+3)
      IF (NW.EQ.0) GO TO 37
      NQDSIZ(KQT+JDIV) = MAX (NQDSIZ(KQT+JDIV),NW)
      IF (JBIT(MQDVWI,JDIV).NE.0) GO TO 41
      IF (JBIT(MQDVGA,JDIV).NE.0) GO TO 44
      GO TO 48
   37 LQ(LMT+1) = -1
      GO TO 48
   41 IF (JDIV.EQ.JQDVSY) GO TO 48
      LQ(LMT+1) = 4
      GO TO 45
   44 LQ(LMT+1) = 3
   45 MQDVAC = MSBIT1 (MQDVAC,JDIV)
   48 LMT = LMT + 8
      JDIV = JDIV + 1
      IF (JDIV.EQ.JQDVLL+1) JDIV=JQDVSY
      IF (JDIV.LT.21) GO TO 32
      LQMTE = LMT
      LQMTLU = LMT
      LQ(LQMTE) = 21
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABR
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      EQUIVALENCE (LMT,LQMTB)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZTA, 4HBR  /
      MSBIT0 (IZW,IZP) = IAND (IZW, NOT(ISHFT(1,IZP-1)))
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (LQMTBR.NE.0) GO TO 81
      LQTA = LQRTA + 2
      LQTE = LQTA
      LMT = LQMTA
      LQ(LQTA-1) = NQLINK + 1
   41 JDIV = LQ(LMT)
      LQ(LMT+5) = LQTE - LQRTA
   42 LQ(LQTE) = LQ(LMT+3)
      LQ(LQTE+1) = LQ(LMT+4)
      LQ(LQTE+2) = 0
      LQ(LQTE+3) = 0
      IACT = LQ(LMT+1)
      IF (IACT.EQ.3) GO TO 61
      IF (IACT.EQ.-1) GO TO 78
      IF (IACT.EQ.4) GO TO 56
      IF (LQ(LMT+6).EQ.-3) GO TO 45
      LQTE = LQTE + 4
      GO TO 78
   45 LQ(LMT+6) = 0
      LQ(LMT+1) = 3
      MQDVAC = MSBIT1 (MQDVAC,JDIV)
      GO TO 42
   56 LQ(LMT+7) = LQ(LMT+4) - LQ(LMT+3)
      GO TO 78
   61 IF (IQPART.NE.0) GO TO 66
      LQTC1 = LQTE
      LQTC2 = LQRTE - (LQMTE-LMT)/2
      IF (LQTC1.GE.LQTC2) GO TO 65
      LQMTC1 = LQ(LMT+3)
      LQMTC2 = LQ(LMT+4)
      CALL MZTABC
      NQFREE = NQFREE + NQFRTC
      IF (NQLIVE.EQ.0) GO TO 64
      IF (IQPART.NE.0) LQMTBR=LMT
      IF (NQFRTC.EQ.0) GO TO 67
      LQ(LMT+6) = LQTE - LQRTA
      LQ(LMT+7) = NQFRTC
      GO TO 78
   64 LQTE = LQTC1
      LQ(LMT+1) = 4
      GO TO 42
   65 LQMTBR = LMT
      IQPART = 7
   66 LQ(LMT+6) = -3
      LQTE = LQTE + 4
   67 LQ(LMT+1) = 2
      IF (LQ(LMT+2).EQ.0) THEN
          LQ(LMT+1) = 0
          MQDVAC = MSBIT0 (MQDVAC,JDIV)
        ENDIF
   78 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 41
      JDIV = LQ(LMT)
      LQ(LQTE) = LQSTA(KQT+JDIV)
  999 NQTRAC = NQTRAC - 2
      RETURN
   81 LMT = LQMTBR
      LQMTBR = 0
      IQPART = 0
      JDIV = LQ(LMT)
      MQDVAC = MSBIT1 (MQDVAC,JDIV)
      WRITE (IQLOG, 9882)
 9882 FORMAT (1X/' MZTABR!! !!!!****  re-entry with LQMTBR non-zero',
     F'****!!!!'/1X)
      JWAY = LQ(LMT+6)
      IF (JWAY.EQ.-3) THEN
          LQTE = LQ(LMT+5)
          GO TO 45
        ENDIF
      LQTE = JWAY - 4
      LQMTC1 = LQ(LQTE)
      LQMTC2 = LQ(LQTE+1)
      LQTC1 = LQTE
      LQTC2 = LQRTE - (LQMTE-LMT)/2
      CALL MZTABC
      LQ(LMT+6) = LQTE - LQRTA
      LQ(LMT+7) = LQ(LMT+7) + NQFRTC
      GO TO 78
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABS
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      LMT = LQMTA
   21 LMT = LMT + 8
      JDIV = LQ(LMT)
      IF (JDIV.LT.JQDVM1) GO TO 21
      IF (JDIV.GT.JQDVM2) RETURN
      LQ(LMT+2) = LQ(LMT+2) + NQDVMV
      IF (LQ(LMT+1).LT.0) GO TO 21
      IF (LQ(LMT+1).GE.2) GO TO 21
      LQ(LMT+1) = 2
      MQDVAC = MSBIT1 (MQDVAC,JDIV)
      GO TO 21
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZTABX
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      JBYTET (MZ,IZW,IZP,NZB) = IAND (MZ,
     + ISHFT (ISHFT(IZW,33-IZP-NZB),-(32-NZB)) )
      MERGE = 0
      LMT = LQMTA
   22 IF (LQ(LMT+1).LT.2) GO TO 27
      JDIV = LQ(LMT)
      MERGE = IOR (MERGE, IQKIND(KQT+JDIV))
   27 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 22
      LMT = LQMTA
   32 IF (LQ(LMT+1)) 38, 33, 37
   33 JDIV = LQ(LMT)
      IF (JBYTET(IQRTO(KQT+JDIV),MERGE,1,26).EQ.0) GO TO 38
      IF (JBYTET(IQRNO(KQT+JDIV),MERGE,1,26).EQ.0) GO TO 38
      LQ(LMT+1) = 1
   37 LQMTLU = LMT + 8
   38 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 32
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZIOCH (IODVEC,NWIOMP,CHFORM)
      COMMON /ZKRAKC/IQHOLK(120), IQKRAK(80), IQCETK(122)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /QUEST/ IQUEST(100)
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /ZBCD/ IQNUM2(11),IQLETT(26),IQNUM(10), IQPLUS,IQMINS
     +, IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +, IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +, IQUNDE,IQCLSQ,IQAND, IQAT, IQQUES,IQOPSQ,IQGREA
     +, IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC, IQLOWL(26)
     +, IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV, IQILEG
     +, NQHOL0,NQHOLL(95)
      DIMENSION IODVEC(99), NWIOMP(9)
      CHARACTER CHFORM*(*)
      EQUIVALENCE (NGR,IQUEST(1)), (NGRU,IQUEST(2))
      DIMENSION MU(99), MCE(99)
      EQUIVALENCE (MU(1),IQHOLK(1)), (MCE(1),IQCETK(1))
      DIMENSION NBITVA(4), NBITVB(4), NBITVC(7)
      DIMENSION MXVALA(4), MXVALB(4), MXVALC(7)
      DIMENSION ITAB(48), INV(10)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZIO, 4HCH  /
      DATA ITAB / 47
     +, -1, 12, -1, 15, -1, 14, -1, 16, 13, -1, -1, -1, -1
     +, -1, -1, -1, -1, -1, 18, -1, -1, -1, -1, -1, -1, -1
     +, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 11, 10
     +, 19, -1, -1, -1, -1, -2, -2, -2 /
      DATA INV / 39, 38, 2, 9, 6, 4, 8, 24, 19, 40 /
      DATA NBITVA / 32, 16, 10, 8 /
      DATA NBITVB / 29, 14, 9, 7 /
      DATA NBITVC / 26, 11, 6, 4, 2, 1, 1 /
      DATA MXVALA / 0, 65536, 1024, 256 /
      DATA MXVALB / 0, 16384, 512, 128 /
      DATA MXVALC / 0, 2048, 64, 16, 4, 2, 2 /
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     + IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     + ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      NWIOMX = NWIOMP(1)
      NCH = LEN (CHFORM)
      IF (NCH.GE.121) GO TO 90
      CALL UCTOH1 (CHFORM,IQHOLK,NCH)
      CALL IZBCDT (NCH,ITAB)
      NCH = IQUEST(1)
      IF (IQUEST(2).NE.0) GO TO 91
      IF (IQUEST(1).EQ.0) GO TO 91
      JPOSR = -1
      JPOSIN = -1
      IVAL = 0
      JCH = 0
      JU = 0
   21 NVAL = 0
   22 JCH = JCH + 1
      NUM = MCE(JCH)
      IF (NUM.GE.10) GO TO 24
      NVAL = 10*NVAL + NUM
      IF (JCH.LT.NCH) GO TO 22
      GO TO 92
   24 IF (NUM.GE.12) GO TO 26
      IF (NVAL.NE.0) GO TO 92
      IF (NUM.EQ.11) THEN
          NVAL = -1
          JPOSIN = JU
          IF (JPOSR.GE.0) GO TO 93
        ENDIF
      IF (JCH.EQ.NCH) GO TO 92
      JCH = JCH + 1
      NUM = MCE(JCH)
      IF (NUM.LT.12) GO TO 92
      IF (NUM.GE.19) GO TO 92
      GO TO 27
   26 IF (NUM.EQ.19) GO TO 29
      IF (NUM.EQ.18) GO TO 92
      IF (NVAL.EQ.0) GO TO 92
      IF (NUM.EQ.15) THEN
          IF (NVAL.NE.2*(NVAL/2)) GO TO 92
        ENDIF
      IVAL = 7
   27 MU(JU+1) = NUM - 11
      MU(JU+2) = NVAL
      JU = JU + 2
      IF (JCH.EQ.NCH) GO TO 31
      IF (JPOSIN.LT.0) GO TO 21
      GO TO 94
   29 IF (NVAL.NE.0) GO TO 92
      IF (JPOSR.GE.0) GO TO 95
      IF (JCH.EQ.NCH) GO TO 92
      JPOSR = JU
      GO TO 21
   31 NU = JU
      NSECA = NU/2
      JU = 2
      IOWD = 65
      NWIO = 0
      JFL12 = 1
      IF (JPOSR.GE.0) THEN
          IF (JPOSR+2.NE.NU) GO TO 41
          IF (MU(NU-1).NE.7) GO TO 41
        ELSE
          IF (MU(NU).EQ.0) JFL12=2
        ENDIF
   32 NSECA = NSECA - 1
      IQUEST(12) = MU(NU-1)
      IF (NSECA.EQ.0) THEN
          IF (JFL12.EQ.1) GO TO 82
          IQUEST(12) = MSBIT1 (IQUEST(12),4)
          GO TO 82
        ENDIF
      IF (NSECA.GE.2) GO TO 33
      IF (MU(2).GE.64) GO TO 34
      IF (JFL12.EQ.2) IQUEST(12)= MSBIT1(IQUEST(12),4)
      IQUEST(12) = MSBYT (MU(1),IQUEST(12),5,3)
      IQUEST(12) = MSBYT (MU(2),IQUEST(12),8,6)
      GO TO 82
   33 IF (IVAL+NSECA.EQ.2) GO TO 38
   34 IQUEST(12) = MSBYT (JFL12,IQUEST(12),14,2)
      IQUEST(12) = MSBYT (MU(1),IQUEST(12),5,3)
      IQUEST(13) = MU(2)
      JBTF = 8
      IF (NSECA.GE.4) GO TO 36
      IOWD = 2177
      NWIO = 1
      IF (NSECA.EQ.1) GO TO 82
      NGR = NSECA
      CALL MZIOCF (0,MXVALA)
      IF (NGR.NE.NGRU) GO TO 36
      NBT = NBITVA(NGRU)
      GO TO 71
   36 IQUEST(12) = MSBIT1 (IQUEST(12),4)
      NGR = MIN (NSECA,3)
      CALL MZIOCF (0,MXVALB)
      NBT = NBITVB(NGRU)
      GO TO 70
   38 IQUEST(12) = 16*IQUEST(12)
      IQUEST(12) = MSBYT (MU(1),IQUEST(12), 8,3)
      IQUEST(12) = MSBYT (MU(3),IQUEST(12),11,3)
      IQUEST(12) = MSBYT (JFL12,IQUEST(12),14,2)
      GO TO 82
   41 NSECL = JPOSR/2
      IF (NSECA.GE.3) GO TO 44
      IQUEST(12) = MU(NU-1)
      IQUEST(12) = MSBYT (3,IQUEST(12),14,2)
      IF (NSECA.EQ.2) GO TO 42
      IF (MU(2).EQ.0) GO TO 82
      GO TO 32
   42 IF (MU(4).NE.0) GO TO 44
      IF (MU(2).GE.64) GO TO 44
      IQUEST(12) = MSBYT (MU(1),IQUEST(12),5,3)
      IQUEST(12) = MSBYT (MU(2),IQUEST(12),8,6)
      IF (NSECL.EQ.1) GO TO 82
      IQUEST(12) = MSBIT1 (IQUEST(12),4)
      GO TO 82
   44 IF (NSECL.EQ.0) GO TO 51
      IF (NSECL.GE.3) GO TO 61
      IF (NSECA.GE.5) GO TO 61
      IF (IVAL+NSECA.EQ.3) GO TO 48
      NGR = NSECA
      CALL MZIOCF (0,MXVALA)
      IF (NGR.NE.NGRU) GO TO 61
      IQUEST(12) = MU(1)
      IQUEST(13) = MU(2)
      IF (NSECL.EQ.2) IQUEST(12)=IQUEST(12)+8
      IQUEST(12) = MSBIT1 (IQUEST(12),16)
      JBTF = 5
      NBT = NBITVA(NGRU)
      IOWD = 2177
      NWIO = 1
      GO TO 71
   48 IQUEST(12) = 8*(2*MU(1)+NSECL-1)
      IQUEST(12) = MSBYT (MU(3),IQUEST(12), 8,3)
      IQUEST(12) = MSBYT (MU(5),IQUEST(12),11,3)
      IQUEST(12) = MSBIT1 (IQUEST(12),16)
      GO TO 82
   51 IF (IVAL+NSECA.EQ.3) GO TO 58
      IQUEST(12) = MU(1)
      IQUEST(13) = MU(2)
      IQUEST(12) = MSBYT (5,IQUEST(12),14,3)
      JBTF = 5
      IF (NSECA.GE.5) GO TO 55
      NGR = NSECA
      CALL MZIOCF (0,MXVALA)
      IF (NGR.NE.NGRU) GO TO 55
      NBT = NBITVA(NGRU)
      IOWD = 2177
      NWIO = 1
      GO TO 71
   55 IQUEST(12) = MSBIT1 (IQUEST(12),4)
      NGR = MIN (NSECA,4)
      CALL MZIOCF (0,MXVALB)
      NBT = NBITVB(NGRU)
      GO TO 70
   58 IQUEST(12) = 16*MU(1)
      IQUEST(12) = MSBYT (MU(3),IQUEST(12), 8,3)
      IQUEST(12) = MSBYT (MU(5),IQUEST(12),11,3)
      IQUEST(12) = MSBYT (5,IQUEST(12),14,3)
      GO TO 82
   61 IQUEST(12) = NSECL
      IQUEST(13) = MU(2)
      IF (NSECL.GE.16) GO TO 96
      IQUEST(12) = MSBYT (MU(1),IQUEST(12),5,3)
      IQUEST(12) = MSBYT (6,IQUEST(12),14,3)
      JBTF = 8
      NGR = 3
      CALL MZIOCF (0,MXVALB)
      NBT = NBITVB(NGRU)
   70 IF (NGRU.EQ.1) GO TO 73
   71 JBTC = 1
      DO 72 JL=2,NGRU
      IQUEST(12) = MSBYT (MU(JU+1),IQUEST(12),JBTF,3)
      JBTF = JBTF + 3
      JBTC = JBTC + NBT
      IQUEST(13) = MSBYT (MU(JU+2),IQUEST(13),JBTC,NBT)
   72 JU = JU + 2
      IF (NGRU.EQ.NSECA) GO TO 82
   73 NSECD = NGRU
      JWIO = 13
   74 JWIO = JWIO + 1
      IQUEST(JWIO) = MU(JU+1)
      JBT = 4
      NGRU = 1
      NGR = MIN (7,NSECA-NSECD)
      IF (NGR.EQ.1) GO TO 77
      CALL MZIOCF (JU,MXVALC)
      IF (NGRU.EQ.1) GO TO 77
      JUST = JU
      DO 76 JL=2,NGRU
      JU = JU + 2
      IQUEST(JWIO) = MSBYT (MU(JU+1),IQUEST(JWIO),JBT,3)
   76 JBT = JBT + 3
      JU = JUST
   77 IQUEST(JWIO-1) = MSBYT (NGRU,IQUEST(JWIO-1),30,3)
      NBT = NBITVC(NGRU)
      DO 79 JL=1,NGRU
      IQUEST(JWIO) = MSBYT (MU(JU+2),IQUEST(JWIO),JBT,NBT)
      JBT = JBT + NBT
   79 JU = JU + 2
      NSECD = NSECD + NGRU
      IF (NSECD.LT.NSECA) GO TO 74
      NWIO = JWIO - 12
      IF (NWIO.GE.NWIOMX) GO TO 97
      IF (NWIO.GE.16) GO TO 97
      IOWD = 64*(32*NWIO+NWIO+1) + 1
   82 IOWD = MSBYT (IQUEST(12),IOWD,17,16)
      IQUEST(12) = IOWD
      IQUEST(1) = NWIO
      CALL UCOPYI (IQUEST(12),IODVEC,NWIO+1)
      IQCETK(121) = IQBLAN
      IF (NQLOGM.GE.1) WRITE (IQLOG,9088) NWIO,CHFORM
 9088 FORMAT (' MZIOCH-',I5,' extra I/O words for Format ',A)
  999 NQTRAC = NQTRAC - 2
      RETURN
   90 NQFATA = 2
      IQUEST(12) = NCH
      GO TO 99
   91 NQCASE = 1
      NQFATA = 3
      IQUEST(12) = IQUEST(1)
      IQUEST(13) = IQUEST(2)
      IF (IQUEST(1).EQ.0) GO TO 99
      GO TO 98
   97 NQCASE = 7
      IQUEST(12) = NWIOMX
      IQUEST(13) = NWIO + 1
      GO TO 98
   96 NQCASE = 6
      IQUEST(12) = NSECA
      IQUEST(13) = NSECL
      GO TO 98
   95 NQCASE = 1
   94 NQCASE = NQCASE + 1
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 2
      print*, '>>>>>> MZIOCH: BAD SYNTAX'
      IQUEST(12) = JCH
      IQUEST(13) = 0
   98 DO 88 JCH=1,NCH
      JCET = MCE(JCH)
      IF (JCET.LT.10) THEN
          MCE(JCH)=IQNUM(JCET+1)
        ELSE
          JCET = INV(JCET-9)
          MCE(JCH) = IQLETT(JCET)
        ENDIF
   88 CONTINUE
      NQFATA = (NCH-1)/4 + 4
   99 IQUEST(11) = IQCETK(121)
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZFLAG (IXSTOR,LHEADP,KBITP,CHOPT)
      COMMON /ZLIMIT/LQLIML,LQLIMH
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      PARAMETER (NQWKTT=2560)
      COMMON /MZCWK/ IQWKTB(NQWKTT), IQWKFZ(NQWKTT)
      DIMENSION KBITP(9),LHEADP(9)
      CHARACTER *(*) CHOPT
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZFL, 4HAG  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MSBIT0 (IZW,IZP) = IAND (IZW, NOT(ISHFT(1,IZP-1)))
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MSBIT (MZ,IZW,IZP) = IOR (IAND (IZW, NOT(ISHFT(1,IZP-1)))
     + ,ISHFT(IAND(MZ,1),IZP-1))
      LHEAD = LHEADP(1)
      IF (LHEAD.EQ.0) RETURN
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR) CALL MZSDIV (IXSTOR,-7)
      CALL MZCHLS (-7,LHEAD)
      IF (IQFOUL.NE.0) GO TO 92
      LQLIML = LQSTA(KQT+21)
      LQLIMH = 0
      IQTBIT = KBITP(1)
      CALL UOPTC (CHOPT,'ZLV',IQUEST)
      IQTVAL = 1 - IQUEST(1)
      IOPTS = 1 - IQUEST(3)
      IOPTH = IQUEST(2)
      LEV = LQWKTB + 3
      LEVE = LEV + NQWKTB - 10
      LQ(LEV-2) = 0
      LQ(LEV-1) = 0
      LQ(LEV) = LHEAD
      LCUR = LHEAD
      LX = LHEAD - 1 + IOPTH
      LAST = LHEAD - IQNS
      IQ(KQS+LCUR) = MSBIT1 (IQ(KQS+LCUR),IQSYSX)
      GO TO 24
   20 LAST = LCUR - IQ(KQS+LCUR-2)
      IQ(KQS+LNEW) = MSBIT0 (IQ(KQS+LNEW),IQSYSX)
      IQ(KQS+LNEW) = MSBIT (IQTVAL,IQ(KQS+LNEW),IQTBIT)
      LQLIML = MIN (LQLIML,LNEW)
      LQLIMH = MAX (LQLIMH,LNEW)
   24 IF (LX.LT.LAST) GO TO 41
      LNEW = LQ(KQS+LX)
      LX = LX - 1
      IF (LNEW.EQ.0) GO TO 24
      CALL MZCHLS (-7,LNEW)
      IF (IQFOUL.NE.0) GO TO 94
      IF (JBIT(IQ(KQS+LNEW),IQSYSX).NE.0) GO TO 24
      LQ(LEV+1) = LX
      LQ(LEV+2) = LCUR
      LEV = LEV + 3
      IF (LEV.GE.LEVE) GO TO 91
      LQ(LEV) = LNEW
   32 LCUR = LNEW
      IQ(KQS+LCUR) = MSBIT1 (IQ(KQS+LCUR),IQSYSX)
      LNEW = LQ(KQS+LCUR)
      IF (LNEW.EQ.0) GO TO 36
      CALL MZCHLS (-7,LNEW)
      IF (IQFOUL.NE.0) GO TO 93
      IF (JBIT(IQ(KQS+LNEW),IQSYSX).NE.0) GO TO 36
      IF (LQ(KQS+LNEW+2).NE.LCUR) GO TO 95
      GO TO 32
   36 CONTINUE
      LAST = LCUR - IQNS
      LX = LCUR - 1
      GO TO 24
   41 LNEW = LCUR
      IF (LCUR.EQ.LQ(LEV)) GO TO 46
      LCUR = LQ(KQS+LCUR+2)
      LX = LCUR - 1
      GO TO 20
   46 LEV = LEV - 3
      LX = LQ(LEV+1)
      LCUR = LQ(LEV+2)
      IF (LCUR.NE.0) GO TO 20
   61 IQ(KQS+LHEAD) = MSBIT0 (IQ(KQS+LHEAD),IQSYSX)
      IF (IOPTS.EQ.0) GO TO 999
      IQ(KQS+LHEAD) = MSBIT (IQTVAL,IQ(KQS+LHEAD),IQTBIT)
      LQLIML = MIN (LQLIML,LHEAD)
      LQLIMH = MAX (LQLIMH,LHEAD)
  999 NQTRAC = NQTRAC - 2
      RETURN
   95 NQCASE = 2
      NQFATA = 1
      IQUEST(14) = LQ(KQS+LNEW+2)
      GO TO 93
   94 NQCASE = 1
      NQFATA = 1
      IQUEST(14) = LX+1 - LCUR
   93 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 2
      IQUEST(12) = LNEW
      IQUEST(13) = LCUR
   92 NQCASE = NQCASE + 1
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 1
      IQUEST(11) = LHEAD
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZGSTA (IGARB)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION IGARB(20)
      LMT = LQMTA
   22 IACT = LQ(LMT+1)
      IF (IACT.LT.3) GO TO 28
      JDIV = LQ(LMT)
      IF (IACT.EQ.3) GO TO 26
      NQDWIP(KQT+JDIV) = NQDWIP(KQT+JDIV) + 1
      GO TO 28
   26 IGARB(JDIV) = IGARB(JDIV) + 1
   28 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 22
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZIOCF (JUP,MXVAL)
      COMMON /ZKRAKC/IQHOLK(120), IQKRAK(80), IQCETK(122)
      COMMON /QUEST/ IQUEST(100)
      DIMENSION MU(99)
      EQUIVALENCE (MU(1),IQHOLK(1))
      EQUIVALENCE (NGR,IQUEST(1)), (NGRU,IQUEST(2))
      DIMENSION JUP(9), MXVAL(9)
      JU = JUP(1)
      MXC = MU(JU+2)
      DO 24 JL=2,NGR
      JU = JU + 2
      MXC = MAX (MU(JU+2),MXC)
      IF (MXC.GE.MXVAL(JL)) GO TO 29
   24 CONTINUE
      NGRU = NGR
      RETURN
   29 NGRU = JL - 1
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZIOCR (IOW)
      COMMON /QUEST/ IQUEST(100)
      COMMON /MZIOC/ NWFOAV,NWFOTT,NWFODN,NWFORE,IFOCON(3)
     +, MFOSAV(2), JFOEND,JFOREP,JFOCUR,MFO(200)
      EQUIVALENCE (JIO,IQUEST(1))
      DIMENSION IOW(9)
      DIMENSION NBITVA(4), NBITVB(4), NBITVC(7)
      DATA NBITVA / 32, 16, 10, 8 /
      DATA NBITVB / 29, 14, 9, 7 /
      DATA NBITVC / 26, 11, 6, 4, 2, 1, 1 /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      NWFODN = 0
      JFOCUR = 0
      JTYPR = IOW(1)
      IOW1 = JBYT (JTYPR,17,16)
      IF (IOW1.NE.0) GO TO 21
      IF (JTYPR.GE.8) GO TO 21
      MFO(1) = JTYPR
      MFO(2) = -1
      JFOEND = 2
      JFOREP = 2
      RETURN
   21 JFOEND = 0
      JFOREP = 0
      JIO = 1
      JTYPR = JBYT (IOW1,1,3)
      JFLAG = JBIT (IOW1,4)
      JCLASS = JBYT (IOW1,14,3)
      JFL12 = 0
      GO TO ( 101, 201, 301, 401, 501, 601, 991), JCLASS
      JFL12 = JFLAG + 1
      JTYP = JBYT (IOW1,5,3)
      IF (JTYP.NE.0) THEN
          MFO(1) = JTYP
          MFO(2) = JBYT (IOW1,8,6)
          JFOEND = 2
        ENDIF
   24 IF (JTYPR.EQ.7) GO TO 28
      MFO(JFOEND+1) = JTYPR
      MFO(JFOEND+2) = JFL12 - 2
      JFOEND = JFOEND + 2
      JFOREP = JFOEND
      RETURN
   28 JFOREP = JFOEND
      MFO(JFOEND+1) = 7
      MFO(JFOEND+2) = 0
      JFOEND = JFOEND + 2
      RETURN
  101 CONTINUE
  201 JFL12 = JCLASS
      IF (JTYPR.NE.0) GO TO 821
      JTYPR = JBYT (IOW1,5,3)
      JBT = 8
      GO TO 831
  301 JTYP = JBYT (IOW1,5,3)
      IF (JTYP.NE.0) THEN
          MFO(1) = JTYP
          MFO(2) = JBYT (IOW1,8,6)
          JFOEND = 2
          IF (JFLAG.EQ.0) JFOREP = 2
        ENDIF
      MFO(JFOEND+1) = JTYPR
      MFO(JFOEND+2) = 0
      JFOEND = JFOEND + 2
      RETURN
  401 JFOREP = 2*(JFLAG+1)
      JFLAG = 0
  501 IF (JTYPR.EQ.0) GO TO 830
      MFO(1) = JTYPR
      JFOEND = 2
      GO TO 821
  601 JFOREP = 2*JBYT(IOW1,1,4)
      JFLAG = 1
  821 JIO = 2
      DO 822 JBT=5,11,3
      JTYP = JBYT (IOW1,JBT,3)
      IF (JTYP.EQ.0) GO TO 823
      MFO(JFOEND+1) = JTYP
  822 JFOEND = JFOEND + 2
  823 NGRU = JFOEND/2
      IF (JFLAG.EQ.0) THEN
          NBT = NBITVA(NGRU)
        ELSE
          NBT = NBITVB(NGRU)
        ENDIF
      JFOEND = 0
      JBT = 1
      IOWN = IOW(2)
      DO 824 JL=1,NGRU
      MFO(JFOEND+2) = JBYT(IOWN,JBT,NBT)
      JFOEND = JFOEND + 2
  824 JBT = JBT + NBT
      IF (JFLAG.EQ.0) GO TO 839
  825 NGRU = JBYT(IOWN,30,3)
      IF (NGRU.EQ.0) GO TO 839
      JIO = JIO + 1
      IF (JIO.EQ.17) GO TO 991
      IOWN = IOW(JIO)
      JBTT = 1
      JBTC = 3*NGRU + 1
      NBT = NBITVC(NGRU)
      DO 826 JL=1,NGRU
      MFO(JFOEND+1) = JBYT (IOWN,JBTT,3)
      MFO(JFOEND+2) = JBYT (IOWN,JBTC,NBT)
      JBTT = JBTT + 3
      JBTC = JBTC + NBT
  826 JFOEND = JFOEND + 2
      GO TO 825
  830 JBT = 5
  831 DO 834 JL=JBT,11,3
      JTYP = JBYT (IOW1,JL,3)
      IF (JTYP.EQ.0) GO TO 839
      MFO(JFOEND+1) = JTYP
      MFO(JFOEND+2) = 0
  834 JFOEND = JFOEND + 2
  839 IF (JFL12.NE.0) GO TO 24
      RETURN
  991 IQUEST(1) = -1
      MFO(1) = 0
      MFO(2) = -1
      JFOEND = 2
      END

*-------------------------------------------------------------------------------

      FUNCTION MZIXCO (IXAA,IXBB,IXCC,IXDD)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION IXAA(9), IXBB(9), IXCC(9), IXDD(9), IXV(4)
      EQUIVALENCE (IXV(1),IQUEST(11))
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZIX, 4HCO  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     + IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     + ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      MBYTOR (MZ,IZW,IZP,NZB) = IOR (IZW,
     + ISHFT (ISHFT(MZ,32-NZB),-(33-IZP-NZB)))
      IXV(1) = IXAA(1)
      IXV(2) = IXBB(1)
      IXV(3) = IXCC(1)
      IXV(4) = IXDD(1)
      IXCOMP = 0
      DO 49 JL=1,4
      IXIN = IXV(JL)
      IF (IXIN.EQ.0) GO TO 49
      JDV = JBYT (IXIN,1,26)
      JST = JBYT (IXIN,27,6)
      IF (JST.LT.16) GO TO 31
      JST = JST - 16
      IF (JST.GT.NQSTOR) GO TO 91
      IF (JDV.GE.16777216) GO TO 92
      IF (JL.NE.1) GO TO 24
      IXCOMP = IXIN
      JSTORU = JST
      GO TO 49
   24 IF (JST.NE.JSTORU) GO TO 93
      IXCOMP = MBYTOR (JDV,IXCOMP,1,26)
      GO TO 49
   31 IF (JST.GT.NQSTOR) GO TO 91
      IF (JDV.GE.25) GO TO 92
      IF (JDV.EQ.0) GO TO 92
      IF (JL.NE.1) GO TO 34
      IXCOMP = MSBYT (JST+16,IXCOMP,27,5)
      JSTORU = JST
      GO TO 47
   34 IF (JST.EQ.JSTORU) GO TO 47
      IF (JST.NE.0) GO TO 93
      IF (JDV.LT.3) GO TO 47
      IF (JDV.LT.21) GO TO 93
   47 IXCOMP = MSBIT1 (IXCOMP,JDV)
   49 CONTINUE
   59 MZIXCO = IXCOMP
      RETURN
   93 NQCASE = 1
   92 NQCASE = NQCASE + 1
   91 NQCASE = NQCASE + 1
      NQFATA = 7
      IQUEST(15) = JL
      IQUEST(16) = JST
      IQUEST(17) = JDV
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      MZIXCO = 0
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZMOVE
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZMO, 4HVE  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LMT = LQMTA
   23 JDIV = LQ(LMT)
      IACT = LQ(LMT+1)
      NSHF = LQ(LMT+2)
      IF (IACT.EQ.4) GO TO 26
      IF (IACT.NE.3) GO TO 31
      L = LQ(LMT+3)
      LT = LQ(LMT+5) + LQRTA
      N = LQ(LT+2)
      LQSTA(KQT+JDIV) = L + N
      L = LQ(LMT+4)
      LT = LQ(LMT+6) + LQRTA - 4
      N = LQ(LT+2)
      LQEND(KQT+JDIV) = L + N
      GO TO 36
   26 MODE = JBIT (IQMODE(KQT+JDIV),1)
      IF (MODE.NE.0) GO TO 28
      LQSTA(KQT+JDIV) = LQSTA(KQT+JDIV) + NSHF
      GO TO 29
   28 LQSTA(KQT+JDIV) = LQEND(KQT+JDIV) + NSHF
   29 LQEND(KQT+JDIV) = LQSTA(KQT+JDIV)
      GO TO 36
   31 IF (NSHF.EQ.0) GO TO 37
      LQSTA(KQT+JDIV) = LQSTA(KQT+JDIV) + NSHF
      LQEND(KQT+JDIV) = LQEND(KQT+JDIV) + NSHF
   36 CONTINUE
   37 LMT = LMT + 8
      IF (LMT.LT.LQMTE) GO TO 23
      IF (NQNOOP.NE.0) GO TO 999
      IF (LQTE.LE.LQTA) GO TO 999
      LTF = LQTA
   61 NREL = LQ(LTF+2)
      IF (NREL) 64, 68, 71
   64 LOLD = LQ(LTF)
      LNEW = LOLD + NREL
      NW = LQ(LTF+1) - LOLD
      IF (NW.EQ.0) GO TO 68
      CALL UCOPYI (LQ(KQS+LOLD),LQ(KQS+LNEW),NW)
   68 LTF = LTF + 4
      IF (LTF.NE.LQTE) GO TO 61
      GO TO 999
   71 LTFN = LTF
   72 LTFN = LTFN + 4
      IF (LTFN.EQ.LQTE) GO TO 76
      IF (LQ(LTFN+2).GT.0) GO TO 72
   76 LTR = LTFN
   81 LTR = LTR - 4
      LOLD = LQ(LTR)
      NW = LQ(LTR+1) - LOLD
      IF (NW.EQ.0) GO TO 88
      LNEW = LOLD + LQ(LTR+2)
      CALL UCOPY2 (LQ(KQS+LOLD),LQ(KQS+LNEW),NW)
   88 IF (LTR.NE.LTF) GO TO 81
      LTF = LTFN
      IF (LTF.NE.LQTE) GO TO 61
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZPUDX (LP,NWP)
      COMMON /ZBCD/ IQNUM2(11),IQLETT(26),IQNUM(10), IQPLUS,IQMINS
     +, IQSTAR,IQSLAS,IQOPEN,IQCLOS,IQDOLL,IQEQU, IQBLAN
     +, IQCOMA,IQDOT, IQNUMB,IQAPO, IQEXCL,IQCOLO,IQQUOT
     +, IQUNDE,IQCLSQ,IQAND, IQAT, IQQUES,IQOPSQ,IQGREA
     +, IQLESS,IQREVE,IQCIRC,IQSEMI,IQPERC, IQLOWL(26)
     +, IQCROP,IQVERT,IQCRCL,IQNOT, IQGRAV, IQILEG
     +, NQHOL0,NQHOLL(95)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      DIMENSION LP(9),NWP(9)
      MSBIT1 (IZW,IZP) = IOR (IZW, ISHFT(1,IZP-1))
      MSBYT (MZ,IZW,IZP,NZB) = IOR (
     + IAND (IZW, NOT(ISHFT (ISHFT(NOT(0),-(32-NZB)),IZP-1)))
     + ,ISHFT (ISHFT(MZ,32-NZB), -(33-IZP-NZB)) )
      L = LP(1)
      NW = NWP(1)
      ND = NW - 10
      N = MIN (10,NW)
      DO 12 J=0,N-1
   12 LQ(KQS+L+J) = 0
      IF (ND.GE.0) THEN
          LQ(KQS+L) = 12
          L = L + 9
          LQ(KQS+L-4) = IQLETT(4)
          LQ(KQS+L-1) = ND
        ELSE
          N = MSBYT (NW,N,17,6)
          LQ(KQS+L) = N
        ENDIF
      LQ(KQS+L) = MSBIT1 (LQ(KQS+L),IQDROP)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZRELB
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZRE, 4HLB  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LFIXLO = LQ(LQTA-1)
      LFIXRE = LQ(LQTA)
      LFIXHI = LQ(LQTE)
      JHIGO = (LQTE-LQTA) / 4
      NENTR = JHIGO - 1
      IF (NENTR.EQ.0) THEN
          LADTB1 = LQ(LQTA+1)
          NRLTB2 = LQ(LQTA+2)
          IFLTB3 = LQ(LQTA+3)
        ENDIF
      LMRNX = LQMTA
   12 LMR = LMRNX
      IF (LMR.GE.LQMTE) GO TO 999
      LMRNX = LMRNX + 8
      IACT = LQ(LMR+1)
      IF (IACT.LE.0) GO TO 12
      IF (IACT.EQ.4) GO TO 12
      LSTOP = LQ(LMR+4)
      IF (IACT.EQ.3) GO TO 14
      LN = LQ(LMR+3)
      LDEAD = LSTOP
      GO TO 19
   14 LSEC = LQRTA + LQ(LMR+5) - 4
   16 LSEC = LSEC + 4
      LNX = LQ(LSEC)
      LDEAD = LQ(LSEC+1)
   17 LN = LNX
      IF (LN.GE.LSTOP) GO TO 12
      IF (LN.EQ.LDEAD) GO TO 16
   19 CONTINUE
      CALL MZCHLN (-7,LN)
      IF (IQFOUL.NE.0) GO TO 91
      LNX = IQNX
      IF (IQND.LT.0) GO TO 17
      LS = IQLS
      LX = LS + 3
      L2 = LS - IQNS
      L1 = LS - IQNL
      NST = JBYT (LQ(KQS+LN),1,16) - 11
      IF (NST.LT.0) THEN
          LNX = LN + NST + 11
          GO TO 17
        ELSE
          LS = LN + NST
          LX = LS + 3
          L2 = LS - IQ(KQS+LS-2)
          L1 = LS - IQ(KQS+LS-3)
          LNX = LS + IQ(KQS+LS-1) + 9
        ENDIF
      IF (NENTR) 66, 46, 26
   24 LQ(KQS+L1)= 0
   25 L1 = L1 + 1
      IF (L1.EQ.LX) GO TO 17
   26 LFIRST= LQ(KQS+L1)
   27 LINK = LQ(KQS+L1)
      IF (LINK.EQ.0) GO TO 25
      IF (IQFLIO.EQ.0) THEN
          IF (LINK.LT.LFIXLO) GO TO 25
          IF (LINK.GE.LFIXHI) GO TO 25
          IF (LINK.LT.LFIXRE) GO TO 24
        ELSE
          IF (LINK.LT.LFIXRE) GO TO 24
          IF (LINK.GE.LFIXHI) GO TO 24
        ENDIF
      JLOW = 0
      JHI = JHIGO
   29 JEX = (JHI+JLOW) / 2
      IF (JEX.EQ.JLOW) GO TO 31
      IF (LINK.GE.LQ(LQTA+4*JEX)) GO TO 30
      JHI = JEX
      GO TO 29
   30 JLOW = JEX
      GO TO 29
   31 JTB = LQTA + 4*JLOW
      IF (LINK.GE.LQ(JTB+1)) GO TO 33
      LQ(KQS+L1) = LINK + LQ(JTB+2)
      GO TO 25
   33 IF (LQ(JTB+3)) 25, 24, 34
   34 IF (L1.LT.L2) GO TO 24
      IF (LS+1-L1) 36, 24, 35
   35 CONTINUE
      CALL MZCHLS (-7,LINK)
      IF (IQFOUL.NE.0) GO TO 92
      LINK = LQ(KQS+LINK)
      LQ(KQS+L1) = LINK
      IF (LINK.NE.LFIRST) GO TO 27
      GO TO 24
   36 LINK = LQ(KQS+LINK+2)
      LQ(KQS+L1) = LINK
      GO TO 27
   44 LQ(KQS+L1)= 0
   45 L1 = L1 + 1
      IF (L1.EQ.LX) GO TO 17
   46 LFIRST= LQ(KQS+L1)
   47 LINK = LQ(KQS+L1)
      IF (LINK.EQ.0) GO TO 45
      IF (IQFLIO.EQ.0) THEN
          IF (LINK.LT.LFIXLO) GO TO 45
          IF (LINK.GE.LFIXHI) GO TO 45
          IF (LINK.LT.LFIXRE) GO TO 44
          IF (LINK.GE.LADTB1) GO TO 53
        ELSE
          IF (LINK.LT.LFIXRE) GO TO 44
          IF (LINK.GE.LADTB1) GO TO 44
        ENDIF
      LQ(KQS+L1) = LINK + NRLTB2
      GO TO 45
   53 IF (IFLTB3) 45, 44, 54
   54 IF (L1.LT.L2) GO TO 44
      IF (LS+1-L1) 56, 44, 55
   55 CONTINUE
      CALL MZCHLS (-7,LINK)
      IF (IQFOUL.NE.0) GO TO 92
      LINK = LQ(KQS+LINK)
      LQ(KQS+L1) = LINK
      IF (LINK.NE.LFIRST) GO TO 47
      GO TO 44
   56 LINK = LQ(KQS+LINK+2)
      LQ(KQS+L1) = LINK
      GO TO 47
   64 LQ(KQS+L1)= 0
   65 L1 = L1 + 1
      IF (L1.EQ.LX) GO TO 17
   66 LINK = LQ(KQS+L1)
      IF (LINK.EQ.0) GO TO 65
      IF (LINK.LT.LFIXLO) GO TO 65
      IF (LINK.GE.LFIXHI) GO TO 65
      GO TO 64
   92 NQCASE = 1
      NQFATA = 2
      LN = LS
      IQUEST(12) = L1
      IQUEST(13) = LINK
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 1
      IQUEST(11) = LN
      IF (IQFLIO.NE.0) GO TO 98
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
   98 IQUEST(9) = NQCASE
      IQUEST(10)= NQFATA
      NQCASE = 0
      NQFATA = 0
      IQFLIO = -7
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZRELL (MDESV)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION MDESV(99)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZRE, 4HLL  /
      JBIT(IZW,IZP) = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LFIXLO = LQ(LQTA-1)
      LFIXRE = LQ(LQTA)
      LFIXHI = LQ(LQTE)
      JHIGO = (LQTE-LQTA) / 4
      NENTR = JHIGO - 1
      IF (NENTR.EQ.0) THEN
          LADTB1 = LQ(LQTA+1)
          NRLTB2 = LQ(LQTA+2)
          IFLTB3 = LQ(LQTA+3)
        ENDIF
      JDESMX = MDESV(1) - 4
      JDES = -4
      IF (MDESV(2).GE.MDESV(3)) JDES =1
   17 JDES = JDES + 5
      IF (JDES.GE.JDESMX) GO TO 999
      LOCAR = MDESV(JDES+1)
      LIX = LOCAR
      LOCARE = MDESV(JDES+2)
      MODAR = MDESV(JDES+3)
      IF (JBIT(MODAR,31).NE.0) THEN
          IF (LQ(KQS+LOCAR).EQ.0) GO TO 17
          LIX = LIX + 2
        ENDIF
      LIR = LOCAR + JBYT (MODAR,1,15)
      IF (NENTR) 66, 46, 26
   24 LQ(KQS+LIX)= 0
   25 LIX = LIX + 1
      IF (LIX.EQ.LOCARE) GO TO 17
   26 LFIRST= LQ(KQS+LIX)
   27 LINK = LQ(KQS+LIX)
      IF (LINK.EQ.0) GO TO 25
      IF (LINK.LT.LFIXLO) GO TO 25
      IF (LINK.GE.LFIXHI) GO TO 25
      IF (LINK.LT.LFIXRE) GO TO 24
      JLOW = 0
      JHI = JHIGO
   29 JEX = (JHI+JLOW) / 2
      IF (JEX.EQ.JLOW) GO TO 31
      IF (LINK.GE.LQ(LQTA+4*JEX)) GO TO 30
      JHI = JEX
      GO TO 29
   30 JLOW = JEX
      GO TO 29
   31 JTB = LQTA + 4*JLOW
      IF (LINK.GE.LQ(JTB+1)) GO TO 33
      LQ(KQS+LIX) = LINK + LQ(JTB+2)
      GO TO 25
   33 IF (LIX.GE.LIR) GO TO 24
      IF (LQ(JTB+3).LE.0) GO TO 24
      CALL MZCHLS (-7,LINK)
      IF (IQFOUL.NE.0) GO TO 91
      LINK = LQ(KQS+LINK)
      LQ(KQS+LIX) = LINK
      IF (LINK.NE.LFIRST) GO TO 27
      GO TO 24
   44 LQ(KQS+LIX)= 0
   45 LIX = LIX + 1
      IF (LIX.EQ.LOCARE) GO TO 17
   46 LFIRST= LQ(KQS+LIX)
   47 LINK = LQ(KQS+LIX)
      IF (LINK.EQ.0) GO TO 45
      IF (LINK.LT.LFIXLO) GO TO 45
      IF (LINK.GE.LFIXHI) GO TO 45
      IF (LINK.LT.LFIXRE) GO TO 44
      IF (LINK.GE.LADTB1) GO TO 53
      LQ(KQS+LIX) = LINK + NRLTB2
      GO TO 45
   53 IF (LIX.GE.LIR) GO TO 44
      IF (IFLTB3.LE.0) GO TO 44
      CALL MZCHLS (-7,LINK)
      IF (IQFOUL.NE.0) GO TO 91
      LINK = LQ(KQS+LINK)
      LQ(KQS+LIX) = LINK
      IF (LINK.NE.LFIRST) GO TO 47
      GO TO 44
   64 LQ(KQS+LIX)= 0
   65 LIX = LIX + 1
      IF (LIX.EQ.LOCARE) GO TO 17
   66 LINK = LQ(KQS+LIX)
      IF (LINK.EQ.0) GO TO 65
      IF (LINK.LT.LFIXLO) GO TO 65
      IF (LINK.GE.LFIXHI) GO TO 65
      GO TO 64
   91 NQCASE = 1
      NQFATA = 5
      IQUEST(11) = LOCAR + LQSTOR
      IQUEST(12) = LIX - LOCAR + 1
      IQUEST(13) = LINK
      IQUEST(14) = MDESV(JDES+4)
      IQUEST(15) = MDESV(JDES+5)
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZRELX
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCT/ MQDVGA,MQDVWI,JQSTMV,JQDVM1,JQDVM2,NQDVMV,IQFLIO
     +, MQDVAC,NQNOOP,IQPART,NQFREE, IQTBIT,IQTVAL
     +, IQTNMV,JQGAPM,JQGAPR,NQGAPN,NQGAP,IQGAP(5,4)
     +, LQTA,LQTE, LQRTA,LQTC1,LQTC2,LQRTE
     +, LQMTA,LQMTB,LQMTE,LQMTLU,LQMTBR
     +, LQMTC1,LQMTC2, NQFRTC,NQLIVE
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZRE, 4HLX  /
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      L = LQSYSS(KQT+1)
      IF (L.NE.0) THEN
          IQ(KQS+L+3) = IQ(KQS+L+2) + NQLINK
          CALL MZRELL (IQ(KQS+L+1))
        ENDIF
      CALL MZRELB
  999 NQTRAC = NQTRAC - 2
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZSDIV (IXDIVP,IFLAGP)
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +, NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      DIMENSION IXDIVP(9), IFLAGP(9)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HMZSD, 4HIV  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IXIN = IXDIVP(1)
      IFLAG = IFLAGP(1)
      JSTO = JBYT (IXIN,27,4)
      IF (JSTO.NE.JQSTOR) GO TO 41
      IF (IFLAG.LT.0) GO TO 48
   21 JDIV = JBYT (IXIN,1,26)
      JCOM = JBYT (IXIN,31,2)
      IF (JCOM-1) 22, 31, 91
   22 IF (JDIV.GE.25) GO TO 92
      IF (JDIV.GE.21) GO TO 24
      IF (JDIV.GT.JQDVLL) THEN
          IF (JDIV.LT.JQDVSY) GO TO 92
        ENDIF
      IF (JDIV.EQ.0) THEN
          IF (IFLAG.EQ.4) GO TO 94
        ENDIF
      JQDIVI = JDIV
      RETURN
   24 IF (JDIV.EQ.24) GO TO 26
      IF (IFLAG.GT.0) GO TO 93
      JQDIVI = 0
      RETURN
   26 JQDIVI = JQDVSY
      RETURN
   31 IF (IFLAG.GT.0) GO TO 93
      IF (JDIV.GE.16777216) GO TO 92
      JQDIVI = 0
      RETURN
   41 IF (JSTO.GT.NQSTOR) GO TO 91
      JQSTOR = JSTO
      JQDIVR = 0
      KQT = NQOFFT(JQSTOR+1)
      KQS = NQOFFS(JQSTOR+1)
      DO 44 J=1,12
   44 IQCUR(J) = IQTABV(KQT+J)
      NQLOGM = NQLOGL
      IF (IFLAG.GE.0) GO TO 21
   48 JQDIVI = 0
      RETURN
   94 NQCASE = 1
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
      NQFATA = 1
      IQUEST(14) = JDIV
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 3
      IQUEST(11) = IXIN
      IQUEST(12) = IFLAG
      IQUEST(13) = JSTO
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE ZSHUNT (IXSTOR,LSHP,LSUPP,JBIASP,IFLAGP)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN, NQUSED
      PARAMETER (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/ IQFENC(4), LQ(100)
                              DIMENSION IQ(92), Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/ NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +, LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +, MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +, NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/ JQSTOR,KQT,KQS, JQDIVI,JQDIVR
     +, JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +, LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +, JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/ LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +, JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +, LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +, LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +, IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +, NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +, NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +, IQDN1(20), IQDN2(20), KQFT, LQFSTA(21)
                                       DIMENSION IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /MZCN/ IQLN,IQLS,IQNIO,IQID,IQNL,IQNS,IQND, IQNX,IQFOUL
      DIMENSION LSHP(9),LSUPP(9),JBIASP(9),IFLAGP(9)
      DIMENSION NAMESR(2)
      DATA NAMESR / 4HZSHU, 4HNT  /
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      LSH = LSHP(1)
      IF (LSH.EQ.0) GO TO 999
      LSUP = LSUPP(1)
      JBIAS = JBIASP(1)
      IFLAG = IFLAGP(1)
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR) CALL MZSDIV (IXSTOR,-7)
      CALL MZCHLS (-7,LSH)
      IF (IQFOUL.NE.0) GO TO 91
      IF (NQLOGL.GE.2) THEN
          IF (JBIAS.GE.2) LSUP=0
          WRITE (IQLOG,9011) JQSTOR,LSH,LSUP,JBIAS,IFLAG,IQID
        ENDIF
 9011 FORMAT (' ZSHUNT-  Store',I3,' LSH/LSUP/JBIAS/IFLAG='
     F,2I9,1X,I6,1X,I3,' IDH= ',A4)
      KEX = LQ(KQS+LSH+2)
      LNEX = LQ(KQS+LSH)
      LPRE = 0
      IF (JBIAS-1) 21, 25, 28
   21 CONTINUE
      CALL MZCHLS (-7,LSUP)
      IF (IQFOUL.NE.0) GO TO 92
      IF (IQNS+JBIAS.LT.0) GO TO 93
      KIN = LSUP + JBIAS
      LNIN = LQ(KQS+KIN)
      LUP = LSUP
      IF (JBIAS.NE.0) GO TO 29
      LPRE = LUP
      LUP = LQ(KQS+LUP+1)
      GO TO 29
   25 LNIN = LSUP
      IF (LNIN.EQ.0) GO TO 26
      CALL MZCHLS (-7,LSUP)
      IF (IQFOUL.NE.0) GO TO 92
      KIN = LQ(KQS+LNIN+2)
      LUP = LQ(KQS+LNIN+1)
      GO TO 29
   26 KIN = LOCF(LSUPP(1)) - LQSTOR
      LUP = 0
      GO TO 29
   28 KIN = 0
      LNIN = 0
      LUP = 0
      IF (KEX.EQ.0) GO TO 51
   29 IF (KIN.EQ.KEX) GO TO 999
      L = MAX (LNIN,LPRE)
      IF (L.EQ.0) GO TO 51
      IF (L.GE.LQEND(KQT+20)) GO TO 94
      IF (L.GE.LQEND(KQT+JQDVLL)) GO TO 43
      JQDIVI = 2
      IF (L.GE.LQEND(KQT+2)) GO TO 44
      IF (L.GE.LQSTA(KQT+2)) GO TO 45
      JQDIVI = 1
      GO TO 45
   43 JQDIVI = JQDVSY - 1
   44 JQDIVI = JQDIVI + 1
      IF (L.GE.LQEND(KQT+JQDIVI)) GO TO 44
   45 IF (LSH.LT.LQSTA(KQT+JQDIVI)) GO TO 94
      IF (LSH.GE.LQEND(KQT+JQDIVI)) GO TO 94
   51 IF (LNEX.EQ.0) GO TO 58
      IF (IFLAG.EQ.0) GO TO 57
      L = LSH
   53 CALL MZCHLS (-7,LNEX)
      IF (IQFOUL.NE.0) GO TO 95
      L = LNEX
      LNEX = LQ(KQS+LNEX)
      IF (LNEX.NE.0) GO TO 53
      LNEX = LSH
   55 LEND = LNEX
      LQ(KQS+LEND+1) = LUP
      LNEX = LQ(KQS+LEND)
      IF (LNEX.NE.0) GO TO 55
      GO TO 71
   57 CONTINUE
      L = LSH
      CALL MZCHLS (-7,LNEX)
      IF (IQFOUL.NE.0) GO TO 95
   58 LEND = LSH
      LQ(KQS+LSH+1) = LUP
   71 IF (KEX .NE.0) LQ(KQS+KEX) = LNEX
      IF (LNEX.NE.0) LQ(KQS+LNEX+2) = KEX
      IF (KIN.NE.0) THEN
          LQ(KQS+KIN) = LSH
        ELSE
          LSUPP(1) = LSH
        ENDIF
      LQ(KQS+LSH+2) = KIN
      LQ(KQS+LEND) = LNIN
      IF (LNIN.NE.0) LQ(KQS+LNIN+2) = LEND
  999 NQTRAC = NQTRAC - 2
      RETURN
   95 NQCASE = 1
      NQFATA = 1
      IQUEST(16) = LNEX
   94 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 1
      IQUEST(15) = L
   93 NQCASE = NQCASE + 1
   92 NQCASE = NQCASE + 1
   91 NQCASE = NQCASE + 1
      NQFATA = NQFATA + 4
      IQUEST(11) = LSH
      IQUEST(12) = LSUP
      IQUEST(13) = JBIAS
      IQUEST(14) = IFLAG
      IQUEST(9) = NAMESR(1)
      IQUEST(10)= NAMESR(2)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE ZHTOI (HOLL,INTV,NP)
      PARAMETER (NQTCET=256)
      COMMON /ZCETA/ IQCETA(256),IQTCET(256)
      INTEGER INTV(99), HOLL(99)
      DO 39 JWH=1,NP
      MWH = HOLL(JWH)
      INTW = 0
      DO 29 JL=1,4
      INTW = ISHFT (INTW,-6)
      JV = IAND (MWH,255)
      IF (JV.EQ.32) THEN
      IF (JL.NE.1) GO TO 29
      ENDIF
      JV = IQTCET(JV+1)
      INTW = IOR (INTW,ISHFT(JV,18))
   29 MWH = ISHFT (MWH,-8)
   39 INTV(JWH) = INTW
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZSCAN(CHPATH,UROUT)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      PARAMETER (NLPATM=100)
      COMMON /RZDIRN/NLCDIR,NLNDIR,NLPAT
      COMMON /RZDIRC/CHCDIR(NLPATM),CHNDIR(NLPATM),CHPAT(NLPATM)
      CHARACTER*16   CHNDIR,    CHCDIR,    CHPAT
      COMMON /RZCH/  CHWOLD,CHL
      CHARACTER*255  CHWOLD,CHL
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     +           KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     +           KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     +           KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     +           KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
      CHARACTER *(*) CHPATH
      EXTERNAL UROUT
      DIMENSION ISD(NLPATM),NSD(NLPATM),IHDIR(4)
      IQUEST(1)=0
      IF(LQRS.EQ.0)GO TO 99
      IF(LCDIR.EQ.0)GO TO 99
      CALL RZCDIR(CHWOLD,'R')
      CALL RZCDIR(CHPATH,' ')
      IF(IQUEST(1).NE.0) GOTO 99
      CALL RZPAFF(CHPAT,NLPAT,CHL)
      NLPAT0=NLPAT
      ITIME=0
  10  CONTINUE
      IF(ITIME.NE.0)THEN
         CALL RZPAFF(CHPAT,NLPAT,CHL)
      IF(IQUEST(1).NE.0)THEN
         NLPAT=NLPAT-1
         GO TO 20
      ENDIF
         CALL RZCDIR(CHL,' ')
      ENDIF
      IF(IQUEST(1).NE.0)THEN
         NLPAT=NLPAT-1
         GO TO 20
      ENDIF
      ISD(NLPAT)=0
      NSD(NLPAT)=IQ(KQSP+LCDIR+KNSD)
      CALL UROUT(CHL)
  20  ISD(NLPAT)=ISD(NLPAT)+1
      IF(ISD(NLPAT).LE.NSD(NLPAT))THEN
         NLPAT=NLPAT+1
         LS=IQ(KQSP+LCDIR+KLS)
         IH=LS+7*(ISD(NLPAT-1)-1)
         CALL ZITOH(IQ(KQSP+LCDIR+IH),IHDIR,4)
         CALL UHTOC(IHDIR,4,CHPAT(NLPAT),16)
         ITIME=ITIME+1
         GO TO 10
      ELSE
         NLPAT=NLPAT-1
         IF(NLPAT.GE.NLPAT0)THEN
            LUP=LQ(KQSP+LCDIR+1)
            CALL MZDROP(JQPDVS,LCDIR,' ')
            LCDIR=LUP
            GO TO 20
         ENDIF
      ENDIF
  90  CALL RZCDIR(CHWOLD,' ')
  99  RETURN
      END

*-------------------------------------------------------------------------------

      SUBROUTINE MZWIPE (IXWP)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /ZVFAUT/IQVID(2),IQVSTA,IQVLOG,IQVTHR(2),IQVREM(2,6)
      DIMENSION    IXWP(9)
      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HMZWI, 4HPE   /
      IXWIPE = IXWP(1)
      IF (IXWIPE.EQ.0)  IXWIPE=21
      CALL MZGARB (0,IXWIPE)
      END

*-------------------------------------------------------------------------------

      SUBROUTINE RZEND(CHDIR)
      COMMON /ZUNIT/ IQREAD,IQPRNT,IQPR2,IQLOG,IQPNCH,IQTTIN,IQTYPE
      COMMON /ZUNITZ/IQDLUN,IQFLUN,IQHLUN,  NQUSED
      COMMON /ZSTATE/QVERSN,NQPHAS,IQDBUG,NQDCUT,NQWCUT,NQERR
     +,              NQLOGD,NQLOGM,NQLOCK,NQDEVZ,NQOPTS(6)
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
      COMMON /QUEST/ IQUEST(100)
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
      COMMON /MZCC/  LQPSTO,NQPFEN,NQPSTR,NQPREF,NQPLK,NQPMIN,LQP2E
     +,              JQPDVL,JQPDVS,NQPLOG,NQPNAM(6)
     +,              LQSYSS(10), LQSYSR(10), IQTDUM(22)
     +,              LQSTA(21), LQEND(20), NQDMAX(20),IQMODE(20)
     +,              IQKIND(20),IQRCU(20), IQRTO(20), IQRNO(20)
     +,              NQDINI(20),NQDWIP(20),NQDGAU(20),NQDGAF(20)
     +,              NQDPSH(20),NQDRED(20),NQDSIZ(20)
     +,              IQDN1(20), IQDN2(20),      KQFT, LQFSTA(21)
                                       DIMENSION    IQTABV(16)
                                       EQUIVALENCE (IQTABV(1),LQPSTO)
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
      EQUIVALENCE (LQRS,LQSYSS(7))
      CHARACTER  CHDIR*(*)
      DIMENSION IHDIR(4)
      LOGICAL RZSAME
      JBIT(IZW,IZP)     = IAND(ISHFT(IZW,-(IZP-1)),1)
      JBYT(IZW,IZP,NZB) = ISHFT(ISHFT(IZW,33-IZP-NZB),-(32-NZB))
      IQUEST(1)=0
      IF(LQRS.EQ.0)GO TO 99
      CALL RZSAVE
      NCHD=LEN(CHDIR)
      IF(NCHD.GT.16)NCHD=16
      CALL VBLANK(IHDIR,4)
      CALL UCTOH(CHDIR,IHDIR,4,NCHD)
      CALL ZHTOI(IHDIR,IHDIR,4)
      LRZ=LQRS
  10  IF(LRZ.NE.0)THEN
         IF(.NOT.RZSAME(IHDIR,IQ(KQSP+LRZ+1),4))THEN
            LRZ=LQ(KQSP+LRZ)
            GO TO 10
         ENDIF
      LTOP=LRZ
      LOGLV = JBYT(IQ(KQSP+LTOP),15,3)-3
      IF(LOGLV.GE.0) WRITE(IQLOG,9019) CHDIR
 9019 FORMAT(' RZEND. called for ',A)
         IF(JBIT(IQ(KQSP+LTOP),3).NE.0)THEN
            LCDIR=LTOP
         print*,'>>>>>> RZFREE'
*            CALL RZFREE('RZFILE')
         ENDIF
         CALL MZDROP(JQPDVS,LTOP,' ')
         LTOP = 0
         LCDIR= 0
      ELSEIF(NQLOGD.GE.-2)THEN
         WRITE(IQLOG,1000) CHDIR
 1000    FORMAT(' RZEND. Unknown directory ',A)
      ENDIF
  99  RETURN
      END

*-------------------------------------------------------------------------------
