* @(#)root/main:$Name$:$Id$ */
* Author: Valery Fine(fine@vxcern.cern.ch)   02/06/97 */

************************************************************************
*   This subroutine conatined the wrong number of parameters and
*   has not been fixed by CERNLIB yet.
*   For that we include it in the ROOT distribution.
************************************************************************
* $Id: tzvers.F,v 1.2 1996/04/18 16:14:56 mclareni Exp $
*
* $Log: tzvers.F,v $
* Revision 1.2  1996/04/18 16:14:56  mclareni
* Incorporate changes from J.Zoll for version 3.77
*
* Revision 1.1.1.1  1996/03/06 10:47:28  mclareni
* Zebra
*
*
      SUBROUTINE TZVERS (IXSTOR,LBK,CHIDH,IDNP,ISELP,IFLAG)

C-    Find title bank with IDH (and IDN if non-zero)
C-    and a validity range spanning ISELP

*
* $Id: mqsys.inc,v 1.1.1.1 1996/03/06 10:46:54 mclareni Exp $
*
* $Log: mqsys.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:54  mclareni
* Zebra
*
*
*

*
* mqsys.inc
*

*
* $Id: mzbits.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: mzbits.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*
*

*
*
* mzbits.inc
*
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
*
* $Id: quest.inc,v 1.1.1.1 1996/03/06 10:46:52 mclareni Exp $
*
* $Log: quest.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:52  mclareni
* Zebra
*
*
*
*
* quest.inc
*
      COMMON /QUEST/ IQUEST(100)
*
* $Id: zebq.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: zebq.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*

*
*
* zebq.inc
*
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
*
* $Id: mzca.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: mzca.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*
*
*     Process Master parameters
*
* mzca.inc
*

      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM,NQFATA,NQCASE
     +,              NQTRAC,MQTRAC(48)
                                       EQUIVALENCE (KQSP,NQOFFS(1))

*    Process Master parameters

*
* $Id: mzcb.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: mzcb.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra

*
*                 Current Store and Division
*
* mzcb.inc
*
      COMMON /MZCB/  JQSTOR,KQT,KQS,  JQDIVI,JQDIVR
     +,              JQKIND,JQMODE,JQDIVN,JQSHAR,JQSHR1,JQSHR2,NQRESV
     +,              LQSTOR,NQFEND,NQSTRU,NQREF,NQLINK,NQMINR,LQ2END
     +,              JQDVLL,JQDVSY,NQLOGL,NQSNAM(6)
                                       DIMENSION    IQCUR(16)
                                       EQUIVALENCE (IQCUR(1),LQSTOR)
*
* $Id: mzcc.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: mzcc.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*

*
*                 Store Control Table (matrix)
*
* mzcc.inc
*
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

*
* $Id: eqlqt.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: eqlqt.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*

*
*          TZ - 1 structural link (only)
*
* eqlqt.inc
*
                   DIMENSION    LQT(9)
                   EQUIVALENCE (LQT(1),LQSYSS(3))
C--------------    END CDE                             -----------------
      INTEGER      CHIDH(9), IDNP(9), ISELP(9)

      DIMENSION    NAMESR(2)
      DATA  NAMESR / 4HTZVE, 4HRS   /

*
* $Id: q_jbyt.inc,v 1.1 1996/04/18 16:15:07 mclareni Exp $
*
* $Log: q_jbyt.inc,v $
* Revision 1.1  1996/04/18 16:15:07  mclareni
* Incorporate changes from J.Zoll for version 3.77
*
*

*
* q_jbyt.inc
*
      JBYT (IZW,IZP,NZB) = ISHFT (ISHFT(IZW,33-IZP-NZB), -32+NZB)
*
* $Id: qstore.inc,v 1.1.1.1 1996/03/06 10:46:53 mclareni Exp $
*
* $Log: qstore.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:53  mclareni
* Zebra
*
*

*
*
* qstore.inc
*
      IF (JBYT(IXSTOR,27,6).NE.JQSTOR)  CALL MZSDIV (IXSTOR,-7)
*  -------------   for ZEBRA MZ   ------------------------------

      ISEL = ISELP(1)
      IDN  = IDNP(1)
C*        CALL UCTOH (CHIDH,IDH,4,4)
        CALL UCOPY (CHIDH,IDH,1)

      L = LQT(KQT+1)
      GO TO 12

   11 L = LQ(KQS+L)
   12 IF (L.EQ.0)                  GO TO 21
      IF (IQ(KQS+L-4).NE.IDH)         GO TO 11

      IF (IDN.NE.0)  THEN
          IF (IDN.NE.IQ(KQS+L-5))     GO TO 11
        ENDIF

      IF (ISEL.LT.IQ(KQS+L+1))        GO TO 11
      IF (ISEL.GT.IQ(KQS+L+2))        GO TO 11

      LBK = L
      RETURN

C--       bank not found

   21 IF (IFLAG.EQ.0)  THEN
          LBK = 0
          RETURN
        ENDIF
*
* $Id: qtrace.inc,v 1.1.1.1 1996/03/06 10:46:54 mclareni Exp $
*
* $Log: qtrace.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:54  mclareni
* Zebra

*
*        with check on capacity MQTRAC
*
* qtrace.inc
*
      MQTRAC(NQTRAC+1) = NAMESR(1)
      MQTRAC(NQTRAC+2) = NAMESR(2)
      NQTRAC = NQTRAC + 2
      IF (NQTRAC.GE.41)      CALL ZFATAL
      IQUEST(2)= IDH
      IQUEST(3)= IDN
      IQUEST(4)= ISEL
      IQUEST(5)= LQT(KQT+1)
      K = IFLAG
      IF (K.LT.100)  K=62
      CALL ZTELL (K,1)
      END
*      ==================================================

*
* $Id: qcardl.inc,v 1.1.1.1 1996/03/06 10:46:55 mclareni Exp $
*
* $Log: qcardl.inc,v $
* Revision 1.1.1.1  1996/03/06 10:46:55  mclareni
* Zebra
*
*

*
*
* qcardl.inc
*
