*CMZ :  2.22/00 08/04/99  18.41.30  by  Rene Brun
*CMZ :  2.21/05 08/02/99  12.19.15  by  Rene Brun
*CMZ :  2.00/11 22/08/98  15.25.26  by  Rene Brun
*CMZ :  2.00/10 28/07/98  13.32.42  by  Rene Brun
*-- Author :
      Program g2root
*
**********************************************************************
*
*      Program to convert an existing GEANT geometry/RZ file
*      into a ROOT macro (C++ file).
*
*  To use this conversion program (in $ROOTSYS/bin),
*        g2root [-f <map_names>] <geant_rzfile> <macro_name> [lrecl]
*  run g2root without parameters to see the usage info.
*
*  for example
*        g2root na49.geom na49.C
*  will convert the GEANT RZ file na49.geom into a ROOT macro na49.C
*
*  The default value for lrecl is 1024. The parameter lrecl must be specified
*  if the record length of the Zebra file is greater than 1024.
*
*  You can use <map_names> file to rename generated TNode's.
*  See an example of that file in the commented section below.
*
*
*  To generate the Geometry structure within Root, do:
*    Root > .x na49.C
*    Root > na49.Draw()
*    Root > c1.x3d()    (this invokes the 3-d Root viewver)
*    Root > TFile gna49("na49.root","NEW")  //open a new root file
*    Root > na49.Write()                    //Write the na49 geometry structure
*    Root > gna49.Write()                   //Write all keys (in this case only one)
*
*    IMPORTANT NOTE
*    To be compiled, this program requires a Fortran compiler supporting
*    recursive calls.
*
*  Author: Rene Brun
*  modified by Nikolay I. Root <nroot@inp.nsk.su> to support map_names
*
**********************************************************************
* The following lines starting with the 2 characters ** are an example
* of (map_names> file.
* To make a valid <map_names> file, remove these 2 characters.
*
**#
**#      formal definitions :
**#
**#  'comments' - the things that are ignored by the parser.
**#               The lines starting with '#' or ' '  - comments
**#  'map'      - first two words of 'non-comments'
**#               ==> the trailing part of this lines - also 'comments'
**
**# next lines are the 'map' examples :
**
**CMD2   Detector     Names translation map for CMD2 detector
**VALL   Internals    Internal part of detector.
**VTBE   TubeBe
**VTAL   TubeAL
**VTML   DCInner
**DC0_   DCOuter
**DRIF   DCCell
**
**   first part of map ('key') consists of exactly 4 characters,
**   including trailing blanks. This part of map is a key of
**   GEANT3 volumes/detectors.
**   The second part - 'name' - is just a sequence of non-blank chars.
**   'name' is used as a replacement of 'key' in g2root output file.
**
**   'alias'    - is a map with 'key' and 'name'
**                having one 'stab character' - '%'.
**
**DCC%   DCSector%  - example of alias.
**
**    For above example - any 'keys' that have first 3 chars 'DCC'
**    would be matched by this alias.
**    The last char of 'key'  - 'stab' - used as substitution for
**    '%' in 'name' (like gmake rules, but not so complicated).
**
**    Keys are matched against aliases only if they do not
**    match any explicit 'map'. First found alias, that match
**    the key - is used for substitution.
**
**  ==> The order of aliases is important for matching !
**
**  The last alias may be like this :
**
**%      Block%       Match any key !
**
**%      %     <- assumed by default.
*
**********************************************************************

      parameter (nwpaw=2000000)
      common/pawc/paw(nwpaw)

      character *80 gname
      character *80 fname
      character *8 chtop
      character *8 crecl
      integer npar, lrecl

      call hlimit(nwpaw)

      npar = iargc()
      if (npar.eq.0) then
         print *, 
     +       'Invoke g2root [-f map_name] geant_name macro_name [lrecl]'
         go to 90
      endif
      narg = 1
      call getarg(narg,gname)
      narg = narg + 1
      npar = npar - 1
      if(gname .eq. "-f") then
         if(npar .eq. 0) then
            print *,'Invoke g2root [-f map_name] geant_name macro_name'
            print *,'  Parse error: you need specify <map_name>'
            go to 90
         endif
         call getarg(narg,gname)
         call create_map(gname)
         narg = narg + 1
         npar = npar - 1
         if(npar .eq. 0) then
            print *,'Invoke g2root [-f map_name] geant_name macro_name'
            print *,'  Parse error: you need specify <geant_name>'
            go to 90
         endif
         call getarg(narg,gname)
         narg = narg + 1
         npar = npar - 1
      endif
      if (npar.ge.1) then
         call getarg(narg,fname)
         narg = narg + 1
      else
         idot=index(gname,'.')
         fname = gname(1:idot-1)//'.C'
      endif

      lrecl = 1024
      if (npar.ge.2) then
         call getarg(narg,crecl)
         read (crecl,'(I6)') lrecl
      endif
      call rzopen(1,chtop,gname,'W',lrecl,istat)
      if (istat.ne.0) then
         print *,'Cannot open file'
         go to 90
      endif
      call rzfile(1,chtop,' ')
      call rzldir(' ',' ')

      call g2rin

      call toroot(fname)
      lg = lenocc(gname)
      lf = lenocc(fname)
      print 1000,gname(1:lg),fname(1:lf)
 1000 format(' GEANT file: ',a, ' converted to ROOT macro: ',a)
  90  continue
      end
      Subroutine toroot(fname)
*
**********************************************************************
*
*      Rotation matrices (structure JROTM) are saved into a linked
*      list of class objects TRotMatrix.
*      The JVOLUM structure generates the following lists:
*        - the TMaterial list (material definition only).
*        - the TRotmatrix list (Rotation matrices definition only).
*        - the TShape list (volume definition only).
*        - the TNode list assembling all detector elements
*          the TNode list is a real tree.
*          The Node list contains two variants TNode and TNodeDiv
*          corresponding to the GSPOS and GSDIV mechanisms.
*
*  Author: Rene Brun
**********************************************************************
*
*KEEP,HCBOOK.
      parameter (nwpaw=2000000)
      common/pawc/paw(nwpaw)

      INTEGER   IQ(2), LQ(8000)
      REAL            Q(2)
      EQUIVALENCE (LQ(1),paw(11)),(IQ(1),paw(19)),(Q(1),paw(19))

      INTEGER       JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      COMMON/GCLINK/JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
*KEEP,GCNUM.
      COMMON/GCNUM/NMATE ,NVOLUM,NROTM,NTMED,NTMULT,NTRACK,NPART
     +            ,NSTMAX,NVERTX,NHEAD,NBIT
*

      common/cnodes/nnodes

      parameter (MAXPOS=250000)
*      parameter (MAXPOS=50000)
      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS)

      CHARACTER*4 KSHAP(30),klshap(30)

      character*20 matname
      character*16 cname,mname
      character*(*) fname
      character*256 line
      character*128 creals
      character*16 astring,ptrname
      dimension pmixt(3)
      logical map_found

      DATA KSHAP/'BRIK','TRD1','TRD2','TRAP','TUBE','TUBS','CONE',
     +           'CONS','SPHE','PARA','PGON','PCON','ELTU','HYPE',
     + 13*'    ','GTRA','CTUB','    '/
*________________________________________________________________________
*

      open(unit=51,file=fname,status='unknown')
*
* Create new Geometry object
* ==========================
      nch=lenocc(fname)
      idot=index(fname,'.')
      nct=idot-1
      write(51,1111)fname(1:nct)
1111  format('void ',a,'()',/,'{',/,
     +'//',/,
     +'//  This file has been generated automatically via the root',/,
     +'//  utility g2root from an interactive version of GEANT',/,
     +'//   (see ROOT class TGeometry header for an example of use)',/,
     +'//',/,
     +'TMaterial *mat;',/,
     +'TMixture  *mix;',/,
     +'TRotMatrix *rot;',/,
     +'TNode *Node, *Node1;')

      do 1 i=1,30
         klshap(i) = kshap(i)
         call cutol(klshap(i))
   1  continue
      do 2 i=1,MAXPOS
         nodepos(i) = 0
         nodediv(i) = 0
   2  continue
      nodepos(1) = 1
      nodediv(1) = 1

      write(51,490)fname(1:nct),fname(1:nct),fname(1:nch)
 490  format(/,'TGeometry *',a,' = new TGeometry("',a,'","',a,'");',/)
      IF(JVOLUM.NE.0 ) NVOLUM = IQ(JVOLUM-2)
      IF(JMATE.NE.0 )  NMATE  = IQ(JMATE-2)
      IF(JROTM.NE.0 )  NROTM  = IQ(JROTM-2)
* Print Materials
* =======================
      write(51,3019)
 3019 format(/,
     +'//-----------List of Materials and Mixtures--------------',/)
      do 300 imat=1,nmate
         jma=lq(jmate-imat)
         if(jma.eq.0)go to 300
         nmixt=q(jma+11)
         call uhtoc(iq(jma+1),4,matname,20)
         ncn=lenocc(matname)
         call toint(imat,astring,nc)
         nm=abs(nmixt)
*-*             Case of a simple material
         if (nm.le.1)then
            call toreals(3,q(jma+6),creals,ncr)
            if(q(jma+6).lt.1.and.q(jma+7).lt.1)then
               creals=',0,0,0'
               ncr=lenocc(creals)
            endif
            line=' '
            write(line,3000)astring(1:nc),matname(1:ncn)
     +        ,creals(1:ncr)
 3000       format('mat = new TMaterial("mat',a,'","',a,'"',a,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
*-*             Case of a mixture
         else
            call toint(nm,creals,ncm)
            jmixt=lq(jma-5)
            if(nmixt.gt.0)then
               mname=creals(1:ncm)
            else
               mname='-'//creals(1:ncm)
               ncm=ncm+1
            endif
            line=' '
            write(line,3010)astring(1:nc),matname(1:ncn)
     +          ,mname(1:ncm)
 3010       format('mix = new TMixture("mix',a,'","',a,'",',a,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
            do 292 im=1,nm
               call toint(im-1,astring,nc)
               pmixt(1) = q(jmixt+im)
               pmixt(2) = q(jmixt+nm+im)
               pmixt(3) = q(jmixt+2*nm+im)
               call toreals(3,pmixt,creals,ncr)
               line=' '
               write(line,3020)astring(1:nc),creals(1:ncr)
 3020          format('  mix->DefineElement(',a,a,');')
               nch = lenocc(line)
               write(51,'(a)')line(1:nch)
 292        continue
         endif
 300  continue
* Print Rotation matrices
* =======================
      write(51,3021)
 3021 format(/,'//-----------List of Rotation matrices--------------',/)
      do 100 irot=1,nrotm
         jr=lq(jrotm-irot)
         if(jr.eq.0)go to 100
         call toint(irot,astring,nc)
         call toreals(6,q(jr+11),creals,ncr)
         line=' '
         ptrname = 'rot'//astring(1:nc)
         nch = nc+3
         write(line,1000)ptrname(1:nch),ptrname(1:nch),ptrname(1:nch),
     +     creals(1:ncr)
 1000    format('TRotMatrix *',a,
     +        ' = new TRotMatrix("',a,'","',a,'"',a,');')
         nch = lenocc(line)
         write(51,'(a)')line(1:nch)
 100  continue

*  Print volume definition (ROOT shapes)
*  =======================
      write(51,3022)
 3022 format(/,'//-----------List of Volumes--------------',/)
      Print *,' nvolum= ',nvolum, ' jvolum=',jvolum
C
C  ???? Convert GEANT3 keys to legal C++ names ?????
C
      do 50 ivo = 1,nvolum
         if (lq(jvolum-ivo).eq.0)go to 50        ! <--  That's
         cname=' '                                   !    'real'
*         write(cname,'(a4)')iq(jvolum+ivo)          !     trick !!!
         call uhtoc(iq(jvolum+ivo),4,cname,4)    ! <--
         do 29 i=1,4
            if (ichar(cname(i:i)).eq.0)cname(i:i) = ' '
  29     continue
         n1=lenocc(cname)
         if(n1.lt.4)then
            do 30 i=n1+1,4
               cname(i:i)='_'
  30        continue
         endif
         do 32 i=1,4
            if(cname(i:i).eq.' ')cname(i:i)='_'
            if(cname(i:i).eq.'+')cname(i:i)='p'
            if(cname(i:i).eq.'-')cname(i:i)='m'
            if(cname(i:i).eq.'*')cname(i:i)='s'
            if(cname(i:i).eq.'/')cname(i:i)='h'
            if(cname(i:i).eq.'.')cname(i:i)='d'
            if(cname(i:i).eq.'''')cname(i:i)='q'
            if(cname(i:i).eq.';')cname(i:i)='s'
            if(cname(i:i).eq.':')cname(i:i)='c'
            if(cname(i:i).eq.',')cname(i:i)='v'
            if(cname(i:i).eq.'<')cname(i:i)='l'
            if(cname(i:i).eq.'>')cname(i:i)='g'
            if(cname(i:i).eq.'!')cname(i:i)='e'
            if(cname(i:i).eq.'@')cname(i:i)='a'
            if(cname(i:i).eq.'#')cname(i:i)='d'
            if(cname(i:i).eq.'$')cname(i:i)='d'
            if(cname(i:i).eq.'%')cname(i:i)='p'
            if(cname(i:i).eq.'^')cname(i:i)='e'
            if(cname(i:i).eq.'&')cname(i:i)='a'
            if(cname(i:i).eq.'(')cname(i:i)='l'
            if(cname(i:i).eq.')')cname(i:i)='g'
            if(cname(i:i).eq.'[')cname(i:i)='l'
            if(cname(i:i).eq.']')cname(i:i)='g'
            if(cname(i:i).eq.'{')cname(i:i)='l'
            if(cname(i:i).eq.'}')cname(i:i)='g'
            if(cname(i:i).eq.'=')cname(i:i)='e'
            if(cname(i:i).eq.'~')cname(i:i)='t'
            if(cname(i:i).eq.'|')cname(i:i)='b'
  32     continue
         call uctoh(cname,iq(jvolum+ivo),4,4)
  50  continue
C----------------------------------------------
      do 200 ivo = 1,nvolum
         jv=lq(jvolum-ivo)
         if (jv.eq.0)go to 200
         cname=' '
         if(.not.map_found(iq(jvolum+ivo),cname)) then
            write(cname,'(a4)')iq(jvolum+ivo)
         endif
         call volume(cname,q(jv+1))
 200  continue

* Print volume positioning (ROOT nodes)
* ========================
      write(51,3023)
 3023 format(/,'//-----------List of Nodes--------------',/)

      nnodes = 1
      nlevel = 0
      if (nvolum.gt.0) call node(1,1)

      write(51,2222)
2222  format('}')
      close(51)
      end
*_______________________________________________________________________
      subroutine volume(cname,qjv)
*KEEP,HCBOOK.
      parameter (nwpaw=2000000)
      common/pawc/paw(nwpaw)

      INTEGER   IQ(2), LQ(8000)
      REAL            Q(2)
      EQUIVALENCE (LQ(1),paw(11)),(IQ(1),paw(19)),(Q(1),paw(19))

      INTEGER       JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      COMMON/GCLINK/JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      character *(*) cname
      character*16 astring,cmater
      character*128 creals
      character*256 line
      real qjv(100)
      double precision RADDEG
      dimension dummypars(100)
      DIMENSION NPARS(30)
      CHARACTER*4 KSHAP(30)
      data dummypars/100*0./

      DATA KSHAP/'BRIK','TRD1','TRD2','TRAP','TUBE','TUBS','CONE',
     +           'CONS','SPHE','PARA','PGON','PCON','ELTU','HYPE',
     + 13*'    ','GTRA','CTUB','    '/
      DATA NPARS/3,4,5,11,3,5,5,7,6,6,10,9,3,4,13*0,12,11,0/
*________________________________________________________________________
*

      RADDEG = 57.2957795130823209
      n1=lenocc(cname)
**      print *, 'VOLUME n1=',n1,' cname=',cname(1:n1)
      ishape = qjv(2)
      numed  = qjv(4)
      jtm    = lq(jtmed-numed)
      nmat   = q(jtm+6)
      jma    = lq(jmate-nmat)
      nmixt  = q(jma+11)
      call toint(nmat,astring,nc)
      if(abs(nmixt).eq.1)then
         cmater='mat'//astring(1:nc)
      else
         cmater='mix'//astring(1:nc)
      endif
      ncmat = lenocc(cmater)
      nin   = qjv(3)
      npar  = qjv(5)
      npar0 = npars(ishape)
**TRAP
      jpar = 6
      if (ishape.eq.4) then
         ph = 0.
         if (qjv(jpar+2).ne.0.)ph=atan2(qjv(jpar+3),qjv(jpar+2))*RADDEG
         tt = sqrt(qjv(jpar+2)**2+qjv(jpar+3)**2)
         qjv(jpar+2) = atan(tt)*RADDEG
         if (ph.lt.0.0) ph = ph+360.0
         qjv(jpar+3) = ph
         qjv(jpar+7) = atan(qjv(jpar+7))*RADDEG
         if (qjv(jpar+7).gt.90.0) qjv(jpar+7) = qjv(jpar+7)-180.0
         qjv(jpar+11)= atan(qjv(jpar+11))*RADDEG
         if (qjv(jpar+11).gt.90.0) qjv(jpar+11) = qjv(jpar+11)-180.0
      endif
**PARA
      if (ishape.eq.10) then
         ph = 0.
         if (qjv(jpar+5).ne.0.)ph=atan2(qjv(jpar+6),qjv(jpar+5))*RADDEG
         tt = sqrt(qjv(jpar+5)**2+qjv(jpar+6)**2)
         qjv(jpar+4) = atan(qjv(jpar+4))*RADDEG
         if (qjv(jpar+4).gt.90.0) qjv(jpar+4) = qjv(jpar+4)-180.0
         qjv(jpar+5) = atan(tt)*RADDEG
         if (ph.lt.0.0) ph = ph+360.0
         qjv(jpar+6) = ph
      endif
      if(ishape.eq.11)npar0=4
      if(ishape.eq.12)npar0=3
*      print 2351, cname(1:n1),kshap(ishape)
* 2351      format('Volume:',a, ' shape=',a)
      if (npar.le.0) then
**       print 2352, cname(1:n1),kshap(ishape)
** 2352      format('Warning, volume with 0 parameters:',a, ' shape=',a)
       return
      endif
      if(npar0.le.0)then
         call toreals(npar0,dummypars(1),creals,ncr)
      else
         call toreals(npar0,qjv(7),creals,ncr)
      endif
      line=' '
**      print 2000, kshap(ishape),cname(1:n1),kshap(ishape)
**     +          ,cname(1:n1),cname(1:n1),cmater(1:ncmat),creals(1:ncr)
      write(line,2000)kshap(ishape),cname(1:n1),kshap(ishape)
     +          ,cname(1:n1),cname(1:n1),cmater(1:ncmat),creals(1:ncr)
      nch = lenocc(line)
      write(51,'(a)')line(1:nch)
      if(ishape.eq.11)then
         ndz=qjv(10)
         do iz=1,ndz
            call toreals(3,qjv(11+(iz-1)*3),creals,ncr)
            line=' '
            call toint(iz-1,astring,nci)
            write(line,2010)cname(1:n1),astring(1:nci),creals(1:ncr)
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
         enddo
      endif
      if(ishape.eq.12)then
         ndz=qjv(9)
         do iz=1,ndz
            call toreals(3,qjv(10+(iz-1)*3),creals,ncr)
            line=' '
            call toint(iz-1,astring,nci)
            write(line,2010)cname(1:n1),astring(1:nci),creals(1:ncr)
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
         enddo
      endif
*   Any attributes set ?
      lseen  = qjv(npar+8)
      lstyle = qjv(npar+9)
      lwidth = qjv(npar+10)
      lcolor = qjv(npar+11)
      lfill  = qjv(npar+12)
*      if(ivo.eq.1)lseen=0
      if(lseen.ne.1)then
         call toint(lseen,creals,ncr)
         write(51,195)cname(1:n1),creals(1:ncr)
195        format(2x,a,'->SetVisibility(',a,');')
      endif
      if(lstyle.ne.1)then
         call toint(lstyle,creals,ncr)
         write(51,196)cname(1:n1),creals(1:ncr)
196        format(2x,a,'->SetLineStyle(',a,');')
      endif
      if(lwidth.ne.1)then
         call toint(lwidth,creals,ncr)
         write(51,197)cname(1:n1),creals(1:ncr)
197        format(2x,a,'->SetLineWidth(',a,');')
      endif
      if(lcolor.ne.1)then
         call toint(lcolor,creals,ncr)
         write(51,198)cname(1:n1),creals(1:ncr)
198        format(2x,a,'->SetLineColor(',a,');')
      endif
      if(lfill.ne.0)then
         call toint(lfill,creals,ncr)
         write(51,199)cname(1:n1),creals(1:ncr)
199        format(2x,a,'->SetFillStyle(',a,');')
      endif
2000  format(
     + 'T',a,' *',a,' = new T',a,'("',a,'","',a,'","',a,'"',a,');')
2010  format(2x,a,'->DefineSection(',a,a,');')

      end

      Subroutine node2(ivo,nuserm)
      call node(ivo,nuserm)
      End

*_______________________________________________________________________
       Subroutine node(ivo,nuserm)
*
*             Process one node (volume with contents)
*KEEP,HCBOOK.
      parameter (nwpaw=2000000)
      common/pawc/paw(nwpaw)

      INTEGER   IQ(2), LQ(8000)
      REAL            Q(2)
      EQUIVALENCE (LQ(1),paw(11)),(IQ(1),paw(19)),(Q(1),paw(19))

      INTEGER       JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      COMMON/GCLINK/JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      common/cnodes/nnodes
      common/clevel/nodeold(20),nlevel

      parameter (MAXPOS=250000)
      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS)

      dimension qjv(1000)
      character*16 cnode,cname,mname,anode
      character*256 line
      character*128 creals
      character*16 astring
      character*12 matrix
      character *16 cblank
      Logical map_found
      data cblank/' '/

*---------------------------------------------------------------------
      cblank = ' '
      nlevel = nlevel + 1
      nodeold(nlevel) = nnodes
      jv=lq(jvolum-ivo)
      ishape = q(jv+2)
      nin = q(jv+3)
      mname=' '
      if(.not.map_found(iq(jvolum+ivo),mname)) then
         write(mname,'(a4)')iq(jvolum+ivo)
      endif
      n2=lenocc(mname)
      call toint(nuserm,astring,nci)
*- If top volume, create the top node
      if(ivo.eq.1)then
         write(51,500)mname(1:n2),mname(1:n2),mname(1:n2)
500      format('Node1 = new TNode("',a,'1","',a,'1","',a,'");')
      endif
*-  Generate subnodes of this node (if any)
**      print 2346, iq(jvolum+ivo), nin
** 2346 format('Proc node:',a4,' nin=',i4)
      if(nin.eq.0)then
         nlevel=nlevel-1
         return
      endif
      call cdnode(nodeold(nlevel))
      if(nin.gt.0)then
*                    Volume has positioned contents
         do 300 in=1,nin
            jin=lq(jv-in)
            ivom=q(jin+2)
            nuser  = q(jin+3)
**            print *,'in=',in,' nuser=',nuser
            jinvom = lq(jvolum-ivom)
            npar   = q(jinvom+5)
            ninvom = q(jinvom+3)
            cname=' '
            if(.not.map_found(iq(jvolum+ivom),cname)) then
               write(cname,'(a4)')iq(jvolum+ivom)
            endif
            n1=lenocc(cname)
            if (npar.gt.0) then
               jpar = jinvom+6
            else
               jpar = jin+9
               npar = q(jin+9)
               call ucopy(q(jinvom+1),qjv(1),6)
               qjv(5) = npar
               call ucopy(q(jin+10),qjv(7),npar)
               call ucopy(q(jinvom+7),qjv(7+npar),6)
               call toint(in,astring,nci)
               mname=cname(1:n1)//astring(1:nci)
               cname = mname
               n1 = lenocc(cname)
               call volume(cname,qjv)
*               print 4566, cname(1:n1),ninvom
* 4566          format(' Positioning volume with 0 params:',a,
*     +             ' ninvom=',i5)
            endif
            if(ninvom.gt.0)then
               nnodes = nnodes+1
               if (nnodes.gt.MAXPOS) then
                  print *,'Too many nodes =',nnodes
                  go to 300
               endif
               call toint(nnodes,anode,ncd)
               cnode = 'Node'//anode(1:ncd)
               if (nodepos(nnodes).eq.0) then
                  write(51,4444)cblank(1:nlevel),anode(1:ncd)
 4444             format(a,'TNode *Node',a,';')
                  nodepos(nnodes) = 1
               endif
            else
               cnode = 'Node'
*               print 4567,iq(jvolum+ivom)
* 4567          format('Node divided:',a4)
            endif
            nd=lenocc(cnode)
            irot=q(jin+4)
            if(irot.eq.0)then
               matrix='0'
               ncmatrix=1
            else
               call toint(irot,astring,nci)
               matrix='rot'//astring(1:nci)
               ncmatrix=nci+3
            endif
            call toint(nuser,astring,nci)
            call toreals(3,q(jin+5),creals,ncr)
**            print *,' cname=',cname(1:n1), ' astring=',astring(1:nci)
            mname=cname(1:n1)//astring(1:nci)
            n2=lenocc(mname)
            line=' '
            write(line,3000)cblank(1:nlevel),cnode(1:nd)
     +         ,mname(1:n2),mname(1:n2),cname(1:n1)
     +         ,creals(1:ncr),matrix(1:ncmatrix)
 3000       format(a,a,' = new TNode("',a,'","',a,'",'
     +       ,a,a,',',a,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
            npar=q(jv+5)
            if(ninvom.gt.0) then
               call node2(ivom,nuser)
            endif
 300     continue
      else
         nnodes = nnodes+1
         call toint(nnodes,anode,ncd)
         cnode = 'Nodiv'//anode(1:ncd)
         nd=lenocc(cnode)
         if (nodediv(nnodes).eq.0) then
             write(51,4445)cblank(1:nlevel),anode(1:ncd)
 4445        format(a,'TNodeDiv *Nodiv',a,';')
             nodediv(nnodes) = 1
          endif
         jin=lq(jv-1)
         ivod=q(jin+2)
         cname=' '
*         if(.not.map_found(iq(jvolum+ivod),cname)) then
            write(cname,'(a4)')iq(jvolum+ivod)
*         endif
         n1=lenocc(cname)
         if(cname(n1:n1).eq.'+')cname(n1:)='plus'
         if(cname(n1:n1).eq.'-')cname(n1:)='minus'
         n1=lenocc(cname)
         cname=cname(1:n1)//'_0'
         n2=lenocc(cname)
         iaxis=q(jin+1)
         call toint(iaxis,astring,nci)
         call toreals(3,q(jin+3),creals,ncr)
         line = ' '
         write(line,995)cblank(1:nlevel),
     +          cnode(1:nd),cname(1:n2),cname(1:n2),
     +          cname(1:n1),astring(1:nci), creals(1:ncr)
 995     format(a,
     +   a,' = new TNodeDiv("',a,'","',a,'","',a,'",',a,a,');')
         nch = lenocc(line)
         write(51,'(a)')line(1:nch)

         call node2(ivod,0)

      endif

      nlevel = nlevel - 1
      if (nlevel.gt.0)call cdnode(nodeold(nlevel))
      end
      subroutine cdnode(node)
      common/clevel/nodeold(20),nlevel
      character*16 anode
      character*16 cblank
      data cblank/' '/
      call toint(node,anode,ncd)
      if (nlevel.gt.1)then
         write(51,1000)cblank(1:nlevel-1),anode(1:ncd)
1000     format(a,'Node',a,'->cd();')
      else
         write(51,1001)anode(1:ncd)
1001     format('Node',a,'->cd();')
      endif
      end
      subroutine toint(i,s,nc)
      character*16 s1,s
      s1=' '
      write(s1,'(i7)')i
      do j=1,16
         if(s1(j:j).ne.' ') then
            j1 = j
            go to 10
         endif
      enddo
  10  continue
      do j=16,1,-1
         if (s1(j:j).ne.' ') then
            s=s1(j1:j)
            nc=j-j1+1
            return
         endif
      enddo
      end
      subroutine toreals(n,r,s,nc)
      character*(*) s
      character*14 s1
      dimension r(200)
      s=' '
      nc=0
      do 10 i=1,n
         call toreal(r(i),s1,nch)
         if(nc+nch.gt.128)then
*            print *,'n=',n,' nc=',nc,' nch=',nch
            return
         endif
         s(nc+1:)=','
         s(nc+2:)=s1
         nc=nc+nch+1
  10  continue
      end
      subroutine toreal(r,s,nc)
      character*14 s
      character*14 s1
      s=' '
      if(r.eq.0)then
         s1='0'
         jbeg=1
         jend=1
      else
         write(s1,'(g14.7)')r
         do k=1,14
            if(s1(k:k).ne.' ') then
               jbeg=k
               goto 10
            endif
         enddo
         jbeg=1
 10      continue
         do k=14,jbeg,-1
            if(s1(k:k).ne.' '.and.s1(k:k).ne.'0') then
               jend=k
               goto 20
            endif
         enddo
         jend=14
 20      continue
         if(s1(jend:jend).eq.'.') jend=jend-1
      endif
      nc=jend-jbeg+1
      if(nc.le.0) then
         print *, 'Should never happen'
      endif
      s(1:nc)=s1(jbeg:jend)
      read(s(1:nc),*)t
      if(abs(t-r).gt.5e-7*abs(t+r)) print *, s(1:nc), t, r
      end
      subroutine toreal_old(r,s,nc)
      character*16 s1,s
      if(r.eq.0)then
         s='0'
         nc=1
         return
      endif
      s1=' '
      write(s1,'(f14.7)')r
      j1=1
      do j=1,16
         if(s1(j:j).ne.' ') then
            j1=j
            go to 10
         endif
      enddo
  10  continue
      j2=j1+7
      if(j2.gt.16)j2=16
      do 20 j=j2,j1+1,-1
         if (s1(j:j).eq.' ')go to 20
         if (s1(j:j).ne.'0') then
            if(s1(j:j).eq.'.')then
               s=s1(j1:j-1)
               nc=j-j1
               go to 30
            endif
            s=s1(j1:j)
            nc=j-j1+1
            go to 30
         endif
  20  continue
  30  continue
      if(nc.eq.1.and.s(1:1).eq.'-')then
         s='0'
      endif
      if(nc.eq.0)then
         nc=1
         s='0'
      endif
      end
C--------------------------------------------------------------
      subroutine create_map(fname)
      character*80 fname

      Parameter (max_MAPS = 1000)
      integer nmap,nalias,id(max_MAPS),ialias(max_MAPS)
      character*16 names(max_MAPS)
      common /maps_pool/ nmap,nalias,id,ialias,names

      character*80 line
      character*1 comment,blank
      character*4 chid
      comment='#'
      blank=' '
      nmap = 0
      open(52,file=fname,status='old',ERR=100)
 10   read(52,300,ERR=100,END=100) line
 300  format(a)
      if(line(1:1) .eq. comment) goto 10
      if(line(1:1) .eq. blank)   goto 10
      nmap = nmap + 1
      if(nmap .gt. max_MAPS) then
         nmap = nmap - 1
         print *, 'Warning: Number of names exceed maximum:',nmap
         print *, 'Warning: Rest of file are ignored'
         goto 100
      endif
      chid=line(1:4)
      ialias(nmap) = index(chid,'%')
      if(ialias(nmap).ne.0) then
         nalias = nalias + 1
      endif
      call uctoh(chid,id(nmap),4,4)
*
* Find substitution word
*
      l1=5
      ll=lenocc(line)
      do while(line(l1:l1).le.blank .and. l1.le.ll)
         l1=l1+1
      enddo
      if(l1.gt.ll) then
         nmap = nmap - 1
         goto 10
      endif
      l2=l1
      do while(line(l2:l2).gt.blank .and. l2.le.ll)
         l2=l2+1
      enddo
      if(l2.gt.ll) l2=ll
*************************************************************
      names(nmap) = line(l1:l2)
      if(ialias(nmap).ne.0) then
*     Check that 'name' also has '%'
         if(index(names(nmap),'%').eq.0) then
            nmap = nmap - 1
            nalias = nalias - 1
            print *, '* Error in line  ',line(1:l2)
            print *, '* Both words should have stab symbol "%"'
            print *, '*==> Line ignored ***'
         endif
      endif
*      print 200,ialias(nmap),id(nmap),names(nmap)
* 200  format('Stab index:',i4,' ID = ',a4,'  ==> ',a)
      goto 10
 100  continue
      close(52)
      end
C-------------------------------------------------------------------
      logical function map_found(idv,name)
      integer idv
      character*16 name

      Parameter (max_MAPS = 1000)
      integer nmap,nalias,id(max_MAPS),ialias(max_MAPS)
      character*16 names(max_MAPS)
      common /maps_pool/ nmap,nalias,id,ialias,names

      character*4 chid,chidv,stab
      integer i,j,l
      map_found = .FALSE.
* First step: Search only non-alias
      do i = 1,nmap
         if(ialias(i).eq.0 .and. id(i).eq.idv) then
            name = names(i)
            map_found = .TRUE.
            return
         endif
      enddo
      if(nalias.eq.0) return
* Second step: Search only alias
      call uhtoc(idv,4,chidv,4)
      do i = 1,nmap
         if(ialias(i).eq.0) goto 100
         call uhtoc(id(i),4,chid,4)
         l = lenocc(chid)
         if(ialias(i).eq.1) then
* case 1  :  %aaa
            if(l.eq.1) then       ! single '%' match any name
               stab = chidv
               goto 50            ! accept
            endif
            if(chid(2:l) .ne. chidv(4-l+2:4)) goto 100
            stab = chidv(1:4-l+1)
         elseif(ialias(i).eq.l) then
* case 2  :  aaa%
            if(chid(1:l-1) .ne. chidv(1:l-1)) goto 100
            stab = chidv(l:4)
         else
* case 3  :  a%aa
            j=ialias(i)    ! index of '%'
            if(chid(j+1:l) .ne. chidv(4-(l-j-1):4)) goto 100
            if(chid(1:j-1) .ne. chidv(1:j-1))       goto 100
            stab = chidv(j:4-(l-j-1)-1)
         endif
 50      continue
*         print *, chidv,' matched to ',chid,' with stab ',stab
         name = names(i)
         j = index(name,'%')
         l = lenocc(name)
         if(j.eq.1 .and. l.eq.1) then
            name = stab
         elseif(j.eq.1) then
            name = stab//name(2:l)
         elseif(j.eq.l) then
            name = name(1:l-1)//stab
         else
            name = name(1:j-1)//stab//name(j+1:l)
         endif
         map_found = .TRUE.
*         print *, names(i),' ==> ',name
         return
 100     continue
      enddo
      end
C----------------------------------------------------------------------
      SUBROUTINE G2RIN
C.
C.    ******************************************************************
C.    *                                                                *
C.    *       Routine to read GEANT object(s) fromin the RZ file       *
C.    *         at the Current Working Directory (See RZCDIR)          *
C.    *       The data structures from disk are read in memory         *
C.    *           (VOLU,ROTM,TMED,MATE,SETS,PART,SCAN)                 *
C.    *                                                                *
C.    *     This routine is s short-cut of the GEANT routine GRIN      *
C.    *       Author    R.Brun  *********                              *
C.    *                                                                *
C.    ******************************************************************
C.
*KEEP,HCBOOK.
      parameter (nwpaw=2000000)
      common/pawc/paw(nwpaw)

      INTEGER   IQ(2), LQ(8000)
      REAL            Q(2)
      EQUIVALENCE (LQ(1),paw(11)),(IQ(1),paw(19)),(Q(1),paw(19))

      INTEGER       JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      COMMON/GCLINK/JDIGI ,JDRAW ,JHEAD ,JHITS ,JKINE ,JMATE ,JPART
     +      ,JROTM ,JRUNG ,JSET  ,JSTAK ,JGSTAT,JTMED ,JTRACK,JVERTX
     +      ,JVOLUM,JXYZ  ,JGPAR ,JGPAR2,JSKLT
C
      COMMON/QUEST/IQUEST(100)
      PARAMETER (NLINIT=9,NMKEY=22)
      DIMENSION JNAMES(20),KEYS(2),LINIT(NLINIT)
      DIMENSION LINK(NMKEY)
      EQUIVALENCE (JNAMES(1),JDIGI)
      CHARACTER*4 CKEY,KNAMES(NMKEY)

      DATA KNAMES/'DIGI','DRAW','HEAD','HITS','KINE','MATE','PART',
     +     'ROTM','RUNG','SETS','STAK','STAT','TMED','NULL','VERT',
     +     'VOLU','JXYZ','NULL','NULL','NULL','SCAN','NULL'/
      DATA LINIT/2,6,7,8,9,10,13,16,21/
C.
C.    ------------------------------------------------------------------
C.
      print *,' In g2rin, iquest(13)=',iquest(13)
      IQUEST(1)=0
      IDIV=2
      KVOL=JVOLUM
      CALL VZERO(JNAMES,20)
C
C                 Create a permanent link area for master pointers
C
      CALL MZLINK(0,'/GCLINK/',JDIGI,JSKLT,JDIGI)
*
      DO 20 J=1, NLINIT
         LINK(J)=LINIT(J)
   20 CONTINUE
      NLINK=NLINIT
*
      IKEY=0
  130 CONTINUE
      IKEY=IKEY+1
      CALL RZINK(IKEY,0,'S')
      print *,' after rzink, ikey=',ikey,'iquest(1)=',iquest(1)
      IF(IQUEST(1).NE.0) THEN
         print *,' nkeys=',iquest(7),' nwkey=',iquest(8)
         if (iquest(1).ne.11) then
            IQUEST(1)=0
            GOTO 150
         endif
      ENDIF
      INDKEY = IQUEST(21)
      CALL UHTOC(INDKEY,4,CKEY,4)
      print *,'trying to read:',ckey
      DO 140 I=1,NLINK
         NKEY=ABS(LINK(I))
         IF(CKEY.EQ.KNAMES(NKEY))THEN
            KEYS(1)=IQUEST(21)
            KEYS(2)=IQUEST(22)
            IF(NKEY.LE.20)THEN
               IF(JNAMES(NKEY).NE.0)THEN
                  CALL MZDROP(IDIV,JNAMES(NKEY),'L')
                  JNAMES(NKEY)=0
               ENDIF
               CALL RZIN(IDIV,JNAMES(NKEY),1,KEYS,0,' ')
            ENDIF
         ENDIF
  140 CONTINUE
      GOTO 130
*
  150 NIN=0
*
  999 END
      SUBROUTINE RZINK(KEYU,ICYCLE,CHOPT)
*
************************************************************************
*
*         To find and decode KEYU,ICYCLE
* Input:
*   KEYU    Keyword vector of the information to be read
*   ICYCLE  Cycle number of the key to be read
*           ICYCLE > highest cycle number means read the highest cycle
*           ICYCLE = 0 means read the lowest cycle
*   CHOPT   Character variable specifying the options selected.
*           data structure
*             default
*                   Same as 'D' below
*             'A'   Read continuation of the previously read data structure
*                   with identifier KEYU,ICYCLE
*                   Given that option implies that the record was written with
*                   the same option by a call to RZOUT.
*             'C'   Provide   information   about   the   cycle   numbers
*                   associated with KEY.
*                   The  total number  of  cycles  and the  cycle  number
*                   identifiers of the 19 highest  cycles are returned in
*                   IQUEST(50) and IQUEST(51..89) respectively
*             'D'   Read the  Data structure  with the  (key,cycle)  pair
*                   specified.
*             'N'   Read the neighbouring. keys (i.e. those preceding and
*                   following KEY).
*                   The  key-vectors of  the previous  and  next key  are
*                   available   respectively   as   IQUEST(31..35)    and
*                   IQUEST(41..45), see below.
*             'R'   Read data into existing bank at LSUP,JBIAS
*             'S'   KEYU(1) contains the key serial number
*                   IQUEST(20)= serial number of the key in directory
*                   IQUEST(21..20+NWKEY)=KEY(1....NWKEY)
*
* Called by RZIN,RZVIN
*
*  Author  : R.Brun DD/US/PD
*  Written : 09.05.86
*  Last mod: 11.09.89
*          : 04.03.94 S.Banerjee (Change in cycle structure)
*          : 23.03.95 J.Shiers - check on K/C blocks is on KEY(1)
*
************************************************************************
*
*
* rzcl.inc
*
*
* mzbits.inc
*
      PARAMETER      (IQDROP=25, IQMARK=26, IQCRIT=27, IQSYSX=28)
*
* quest.inc
*
      COMMON /QUEST/ IQUEST(100)
*
* zebq.inc
*
      COMMON /ZEBQ/  IQFENC(4), LQ(100)
                              DIMENSION    IQ(92),        Q(92)
                              EQUIVALENCE (IQ(1),LQ(9)), (Q(1),IQ(1))
**#include "zebra/mzca.inc"
      COMMON /MZCA/  NQSTOR,NQOFFT(16),NQOFFS(16),NQALLO(16), NQIAM
     +,              LQATAB,LQASTO,LQBTIS, LQWKTB,NQWKTB,LQWKFZ
     +,              MQKEYS(3),NQINIT,NQTSYS,NQM99,NQPERM
     +,              NQFATA,NQCASE,NQTRAC
                                       EQUIVALENCE (KQSP,NQOFFS(1))
      COMMON /MZCA2/ MQTRAC(44)
                     CHARACTER  MQTRAC*8
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
* rzclxx.inc
*
      COMMON /RZCL/  LTOP,LRZ0,LCDIR,LRIN,LROUT,LFREE,LUSED,LPURG
     +,              LTEMP,LCORD,LFROM
*
*          RZ - 1 structural link
*
* eqlqr.inc
*
                   EQUIVALENCE (LQRS,LQSYSS(7))
C
*
* rzclun.inc
*
      COMMON /RZCLUN/LUN,LREC,ISAVE,IMODEX,IRELAT,NHPWD,IHPWD(2)
     +,              IZRECL,IMODEC,IMODEH
C
*
* rzcout.inc
*
      COMMON /RZCOUT/IP1,IR1,IR2,IROUT,IRLOUT,IOPTRR
C
*
* rzk.inc
*
      PARAMETER (KUP=5,KPW1=7,KNCH=9,KDATEC=10,KDATEM=11,KQUOTA=12,
     +           KRUSED=13,KWUSED=14,KMEGA=15,KRZVER=16,KIRIN=17,
     +           KIROUT=18,KRLOUT=19,KIP1=20,KNFREE=22,KNSD=23,KLD=24,
     +           KLB=25,KLS=26,KLK=27,KLF=28,KLC=29,KLE=30,KNKEYS=31,
     +           KNWKEY=32,KKDES=33,KNSIZE=253,KEX=6,KNMAX=100)
C
*
* rzckey.inc
*
      COMMON/RZCKEY/IHEAD(3),KEY(100),KEY2(100),KEYDUM(50)
C
* rzcycle.inc
*
*
*     Pointers to cycle content
*
*     KLCYCL : length of cycle block (4,7)
*     KPPCYC : pointer to previous cycle
*     KFRCYC : first record number
*     KSRCYC : secord record number
*     KFLCYC : creation date/time and other stuff
*     KORCYC : offset in first record to data
*     KCNCYC : cycle number
*     KNWCYC : number of words in d/s
*     KKYCYC : key number to which this cycle belongs (only for version 1)
*     KVSCYC : version of RZ cycles structure (0, 1)
*
      INTEGER        KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     +               KCNCYC, KNWCYC, KKYCYC, KVSCYC
      COMMON/RZCYCLE/KLCYCL, KPPCYC, KFRCYC, KSRCYC, KFLCYC, KORCYC,
     +               KCNCYC, KNWCYC, KKYCYC, KVSCYC

      CHARACTER*(*) CHOPT
      DIMENSION KEYU(*)
      EQUIVALENCE (IOPTA,IQUEST(91)), (IOPTC,IQUEST(92))
     +,    (IOPTD,IQUEST(93)), (IOPTN,IQUEST(94)), (IOPTR,IQUEST(95))
     +,    (IOPTS,IQUEST(96))
*
*-----------------------------------------------------------------------
*
*#include "zebra/q_jbyt.inc"
*
      IQUEST(1)=0
      CALL UOPTC(CHOPT,'ACDNRS',IQUEST(91))
*
*           Search KEY and CYCLE
*
      LK=IQ(KQSP+LCDIR+KLK)
      NKEYS=IQ(KQSP+LCDIR+KNKEYS)
      NWKEY=IQ(KQSP+LCDIR+KNWKEY)
      IQUEST(7)=NKEYS
      IQUEST(8)=NWKEY
      IF(NKEYS.EQ.0)GO TO 90
*
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
   5     CONTINUE
      ENDIF
      DO 30 I=IK1,IK2
         LKC=LK+(NWKEY+1)*(I-1)
         IF(IOPTS.EQ.0)THEN
            DO 10 K=1,NWKEY
               IF(IQ(KQSP+LCDIR+LKC+K).NE.KEY(K))GO TO 30
  10        CONTINUE
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
  15        CONTINUE
         ENDIF
         IQUEST(20)=I
         LCYC=IQ(KQSP+LCDIR+LKC)
*
**=================WARNING=======================================
**  The guy who introduced this change was probably drunk !!
** I had to comment this block (Rene brun)
**         IF (KVSCYC.NE.0) THEN
**           IF (IQ(KQSP+LCDIR+LCYC+KKYCYC).NE.I) THEN
**
**    Check should be on content of KEY(1)
**
**            IF (IQ(KQSP+LCDIR+LCYC+KKYCYC).NE.IQ(KQSP+LCDIR+LKC+1)) THEN
**               IQUEST(1) = 11
**               GO TO 99
**            ENDIF
**         ENDIF
**===============================================================
         NC=0
  20     NC=NC+1
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
*
*           Cycle has been found
*           Read record descriptor
*
  50  IF (KVSCYC.EQ.0) THEN
         IR1   = JBYT(IQ(KQSP+LCDIR+LCYC+KFRCYC),17,16)
         IR2   = JBYT(IQ(KQSP+LCDIR+LCYC+KSRCYC),17,16)
         IP1   = JBYT(IQ(KQSP+LCDIR+LCYC+KORCYC), 1,16)
         NW    = JBYT(IQ(KQSP+LCDIR+LCYC+KNWCYC), 1,20)
      ELSE
         IR1   = IQ(KQSP+LCDIR+LCYC+KFRCYC)
         IR2   = IQ(KQSP+LCDIR+LCYC+KSRCYC)
         IP1   = JBYT(IQ(KQSP+LCDIR+LCYC+KORCYC), 1,20)
         NW    = IQ(KQSP+LCDIR+LCYC+KNWCYC)
      ENDIF
      N1    = NW
      IQUEST(2)=1
      IF(IR2.NE.0)IQUEST(2)=(NW-N1-1)/LREC+2
      IQUEST(3)=IR1
      IQUEST(4)=IP1
      IQUEST(5)=IR2
      IQUEST(6)=ICY
      IQUEST(12)=NW
      IQUEST(14)=IQ(KQSP+LCDIR+LCYC+1)
      IQUEST(15)=LCYC
C
C           C option given
C
      IF(IOPTC.NE.0)THEN
         IQUEST(50)=0
         LC1=LCYC
  51     IQUEST(50)=IQUEST(50)+1
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
C
C           N option given. return neighbours
C
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
  52        CONTINUE
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
  53        CONTINUE
         ENDIF
      ENDIF
      GO TO 99
*
*           Error
*
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
  91        CONTINUE
         ENDIF
      ENDIF
*
  99  RETURN
      END
