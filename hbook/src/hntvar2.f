*CMZ :  2.21/05 08/02/99  11.10.43  by  Rene Brun
*CMZ :  0.90/10 09/12/96  17.08.32  by  Rene Brun
*-- Author :    Rene Brun   09/12/96
      SUBROUTINE HNTVAR2(ID1,IVAR,CHTAG,CHFULL,BLOCK,NSUB,ITYPE,ISIZE
     +                  ,NBITS,IELEM)
*.==========>
*.
*.           Returns the tag, block, type, size and array length of the
*.           variable with index IVAR in N-tuple ID1.
*.           N-tuple must already be in memory.
*.
*.           This routine is a modification of the HBOOK routine HNTVAR.
*.
*..=========> ( R.Brun, A.A.Rademakers )
*
*KEEP,HCNTPAR.
      INTEGER   ZBITS,  ZNDIM,  ZNOENT, ZNPRIM, ZNRZB,  ZIFCON,
     +          ZIFNAM, ZIFCHA, ZIFINT, ZIFREA, ZNWTIT, ZITIT1,
     +          ZNCHRZ, ZDESC,  ZLNAME, ZNAME,  ZARIND, ZRANGE, ZNADDR,
     +          ZIBLOK, ZNBLOK, ZLCONT, ZIFBIT, ZIBANK, ZIFTMP, ZITMP,
     +          ZID,    ZNTMP,  ZNTMP1, ZLINK
      PARAMETER(ZBITS=1,   ZNDIM=2,   ZNOENT=3,  ZNPRIM=4,  ZLCONT=6,
     +          ZNRZB=5,   ZIFCON=7,  ZIFNAM=4,  ZIFCHA=5,  ZIFINT=6,
     +          ZIFREA=7,  ZNWTIT=8,  ZITIT1=9,  ZNCHRZ=13, ZIFBIT=8,
     +          ZDESC=1,   ZLNAME=2,  ZNAME=3,   ZRANGE=4,  ZNADDR=12,
     +          ZARIND=11, ZIBLOK=8,  ZNBLOK=10, ZIBANK=9,  ZIFTMP=11,
     +          ZID=12,    ZITMP=10,  ZNTMP=6,   ZNTMP1=3,  ZLINK=6)
*
*KEEP,HCFLAG.
      INTEGER       ID    ,IDBADD,LID   ,IDLAST,IDHOLD,NBIT  ,NBITCH,
     +       NCHAR ,NRHIST,IERR  ,NV
      COMMON/HCFLAG/ID    ,IDBADD,LID   ,IDLAST,IDHOLD,NBIT  ,NBITCH,
     +       NCHAR ,NRHIST,IERR  ,NV
*
*KEEP,HCBOOK.
      INTEGER     NWPAW,IXPAWC,IHDIV,IXHIGZ,IXKU,        LMAIN
      REAL                                       FENC   ,      HCV
*
* VERY IMPORTANT. The dimension of /pawc/must be the same as
*                 in THbookFile. Otherwise FATAL on Windows
      COMMON/PAWC/NWPAW,IXPAWC,IHDIV,IXHIGZ,IXKU,FENC(5),LMAIN,
     +HCV(2000000-11)
      INTEGER   IQ        ,LQ
      REAL            Q
      DIMENSION IQ(2),Q(2),LQ(8000)
      EQUIVALENCE (LQ(1),LMAIN),(IQ(1),LQ(9)),(Q(1),IQ(1))
      INTEGER       HVERSN,IHWORK,LHBOOK,LHPLOT,LGTIT,LHWORK,
     +LCDIR,LSDIR,LIDS,LTAB,LCID,LCONT,LSCAT,LPROX,LPROY,LSLIX,
     +LSLIY,LBANX,LBANY,LPRX,LPRY,LFIX,LLID,LR1,LR2,LNAME,LCHAR,LINT,
     +LREAL,LBLOK,LLBLK,LBUFM,LBUF,LTMPM,LTMP,LTMP1,LHPLIP,LHDUM,
     +LHFIT,LFUNC,LHFCO,LHFNA,LCIDN
      COMMON/HCBOOK/HVERSN,IHWORK,LHBOOK,LHPLOT,LGTIT,LHWORK,
     +LCDIR,LSDIR,LIDS,LTAB,LCID,LCONT,LSCAT,LPROX,LPROY,LSLIX,
     +LSLIY,LBANX,LBANY,LPRX,LPRY,LFIX,LLID,LR1,LR2,LNAME,LCHAR,LINT,
     +LREAL,LBLOK,LLBLK,LBUFM,LBUF,LTMPM,LTMP,LTMP1,LHPLIP,LHDUM(9),
     +LHFIT,LFUNC,LHFCO,LHFNA,LCIDN
*
      INTEGER   KNCX   ,KXMIN  ,KXMAX  ,KMIN1  ,KMAX1 ,KNORM  , KTIT1,
     +          KNCY   ,KYMIN  ,KYMAX  ,KMIN2  ,KMAX2 ,KSCAL2 , KTIT2,
     +          KNBIT  ,KNOENT ,KSTAT1 ,KNSDIR  ,KNRH ,
     +          KCON1  ,KCON2  ,KBITS  ,KNTOT
      PARAMETER(KNCX=3,KXMIN=4,KXMAX=5,KMIN1=7,KMAX1=8,KNORM=9,KTIT1=10,
     +          KNCY=7,KYMIN=8,KYMAX=9,KMIN2=6,KMAX2=10,KSCAL2=11,
     +          KTIT2=12,KNBIT=1,KNOENT=2,KSTAT1=3,KNSDIR=5,KNRH=6,
     +          KCON1=9,KCON2=3,KBITS=1,KNTOT=2)
*
*KEEP,HCBITS.
      INTEGER           I1,   I2,   I3,   I4,   I5,   I6,   I7,   I8,
     +                  I9,   I10,  I11,  I12,  I13,  I14,  I15,  I16,
     +I17,  I18,  I19,  I20,  I21,  I22,  I23,  I24,  I25,  I26,  I27,
     +I28,  I29,  I30,  I31,  I32,  I33,  I34,  I35,  I123, I230
      COMMON / HCBITS  / I1,   I2,   I3,   I4,   I5,   I6,   I7,   I8,
     +                  I9,   I10,  I11,  I12,  I13,  I14,  I15,  I16,
     +I17,  I18,  I19,  I20,  I21,  I22,  I23,  I24,  I25,  I26,  I27,
     +I28,  I29,  I30,  I31,  I32,  I33,  I34,  I35,  I123, I230
*
*KEND.
*
      CHARACTER*(*)  CHTAG, CHFULL, BLOCK
      CHARACTER*80 VAR
      CHARACTER*32   NAME, SUBS
      LOGICAL        LDUM
*
      ID    = ID1
      IDPOS = LOCATI(IQ(LTAB+1),IQ(LCDIR+KNRH),ID)
      IF (IDPOS .LE. 0) THEN
         CALL HBUG('Unknown N-tuple','HNTVAR',ID1)
         RETURN
      ENDIF
      LCID  = LQ(LTAB-IDPOS)
*
      CHTAG = ' '
      NAME  = ' '
      BLOCK = ' '
      NSUB  = 0
      ITYPE = 0
      ISIZE = 0
      IELEM = 0
*
      ICNT  = 0
*
*
      IF (IVAR .GT. IQ(LCID+ZNDIM)) RETURN
*
      LBLOK = LQ(LCID-1)
      LCHAR = LQ(LCID-2)
      LINT  = LQ(LCID-3)
      LREAL = LQ(LCID-4)
*
*-- loop over all blocks
*
  5   LNAME = LQ(LBLOK-1)
*
      IOFF = 0
      NDIM = IQ(LBLOK+ZNDIM)
*
      DO 10 I = 1, NDIM
         ICNT = ICNT + 1
         IF (ICNT .EQ. IVAR) THEN
*
            CALL HNDESC(IOFF, NSUB, ITYPE, ISIZE, NBITS, LDUM)
*	write(6,*) "NBITS = " , NBITS
*
            LL = IQ(LNAME+IOFF+ZLNAME)
            LV = IQ(LNAME+IOFF+ZNAME)
            CALL UHTOC(IQ(LCHAR+LV), 4, NAME, LL)
            CALL UHTOC(IQ(LBLOK+ZIBLOK), 4, BLOCK, 8)
*
            IELEM = 1
            IF (NSUB .GT. 0) THEN
               VAR = NAME(1:LL)//'['
               DO 25 J = NSUB,1,-1
                  LP = IQ(LINT+IQ(LNAME+IOFF+ZARIND)+(J-1))
                  IF (LP .LT. 0) THEN
                     IE = -LP
                     CALL HITOC(IE, SUBS, LL, IERR)
                  ELSE
                     LL = IQ(LNAME+LP-1+ZLNAME)
                     LV = IQ(LNAME+LP-1+ZNAME)
                     CALL UHTOC(IQ(LCHAR+LV), 4, SUBS, LL)
                     LL1 = IQ(LNAME+LP-1+ZRANGE)
                     IE  = IQ(LINT+LL1+1)
                  ENDIF
                  IELEM = IELEM*IE
                  IF (J .EQ. NSUB) THEN
                     VAR = VAR(1:LENOCC(VAR))//SUBS(1:LL)
                  ELSE
                     VAR = VAR(1:LENOCC(VAR))//']['//SUBS(1:LL)
                  ENDIF
   25          CONTINUE
*
               VAR = VAR(1:LENOCC(VAR))//']'
            ELSE
               VAR = NAME(1:LL)
            ENDIF
            CHTAG  = NAME
            CHFULL = VAR
            RETURN
*
         ENDIF
*
         IOFF = IOFF + ZNADDR
  10  CONTINUE
*
      LBLOK = LQ(LBLOK)
      IF (LBLOK .NE. 0) GOTO 5
*
      END

      subroutine hntvar3(id,last,chvar)
      character *80 allvars
      common/callvars/allvars(100)
      common/calloff/ioffset(100)
      character *(*) chvar
      integer id,ivar,last
      save ivar
      data ivar/0/
      if (ivar.ne.0) then
         if (allvars(ivar).ne.chvar) then
            ivar = ivar+1
            allvars(ivar) = chvar
            ioffset(ivar) = 0
         endif
      else
         ivar = ivar+1
         allvars(ivar) = chvar
         ioffset(ivar) = 0
      endif
      ier = 0
      if (last.ne.0) then
         call hgnt1(id,'*',allvars,ioffset,-ivar,1,ier)
         allvars(1) = ' '
         ivar = 0
      endif
      end
