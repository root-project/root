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
*  Author: Rene Brun
*  modified by Nikolay I. Root <nroot@inp.nsk.su> to support map_names
*  modified by Mihaela Gheata for the new geometry format
*
*  To use this conversion program (in $ROOTSYS/bin),
*        g2root [-f <map_names>] <geant_rzfile> <macro_name> [lrecl]
*  run g2root without parameters to see the usage info.
*
*  for example
*        g2root brahms.geom brahms.C
*  will convert the GEANT RZ file brahms.geom into a ROOT macro brahms.C
*
*  The default value for lrecl is 1024. The parameter lrecl must be specified
*  if the record length of the Zebra file is greater than 1024.
*
*
*  To generate the Geometry structure within Root, do:
*
*         root[0]> .x brahms.C;
*         root[0]> new TBrowser();
*
*  An interactive session
* ------------------------
*
*   Provided that a geometry was successfully built and closed , the manager class will
*  register itself to ROOT and the logical/physical structures will become immediately
*  browsable. The ROOT browser will display starting from the geometry folder : the list of
*  transformations and materials, the top volume and the top logical node. These last
*  two can be fully expanded, any intermediate volume/node in the browser being subject
*  of direct access context menu operations (right mouse button click). All user
*  utilities of classes TGeoManager, TGeoVolume and TGeoNode can be called via the
*  context menu.
*
* see http://root.cern.ch/root/htmldoc/gif/t_browser.jpg
*
*   --- Drawing the geometry
*
*    Any logical volume can be drawn via TGeoVolume::Draw() member function.
*  This can be direcly accessed from the context menu of the volume object
*  directly from the browser.
*    There are several drawing options that can be set with
*  TGeoManager::SetVisOption(Int_t opt) method :
*  opt=0 - only the content of the volume is drawn, N levels down (default N=3).
*     This is the default behavior. The number of levels to be drawn can be changed
*     via TGeoManager::SetVisLevel(Int_t level) method.
*
* see http://root.cern.ch/root/htmldoc/gif/t_frame0.jpg
*
*  opt=1 - the final leaves (e.g. daughters with no containment) of the branch
*     starting from volume are drawn down to the current number of levels.
*                                      WARNING : This mode is memory consuming
*     depending of the size of geometry, so drawing from top level within this mode
*     should be handled with care for expensive geometries. In future there will be
*     a limitation on the maximum number of nodes to be visualized.
*
* see http://root.cern.ch/root/htmldoc/gif/t_frame1.jpg
*
*  opt=2 - only the clicked volume is visualized. This is automatically set by
*     TGeoVolume::DrawOnly() method
*  opt=3 - only a given path is visualized. This is automatically set by
*     TGeoVolume::DrawPath(const char *path) method
*
*     The current view can be exploded in cartesian, cylindrical or spherical
*  coordinates :
*    TGeoManager::SetExplodedView(Int_t opt). Options may be :
*  - 0  - default (no bombing)
*  - 1  - cartesian coordinates. The bomb factor on each axis can be set with
*         TGeoManager::SetBombX(Double_t bomb) and corresponding Y and Z.
*  - 2  - bomb in cylindrical coordinates. Only the bomb factors on Z and R
*         are considered
*
* see http://root.cern.ch/root/htmldoc/gif/t_frameexpl.jpg
*
*  - 3  - bomb in radial spherical coordinate : TGeoManager::SetBombR()
*
*  Volumes themselves support different visualization settings :
*     - TGeoVolume::SetVisibility() : set volume visibility.
*     - TGeoVolume::VisibleDaughters() : set daughters visibility.
*  All these actions automatically updates the current view if any.
*
*   --- Checking the geometry
*
*   Several checking methods are accesible from the volume context menu. They
*  generally apply only to the visible parts of the drawn geometry in order to
*  ease geometry checking, and their implementation is in the TGeoChecker class
*  from the painting package.
*
*  1. Checking a given point.
*    Can be called from TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z).
*  This method is drawing the daughters of the volume containing the point one
*  level down, printing the path to the deepest physical node holding this point.
*  It also computes the closest distance to any boundary. The point will be drawn
*  in red.
*
* see http://root.cern.ch/root/htmldoc/gif/t_checkpoint.jpg
*
*   2. Shooting random points.
*    Can be called from TGeoVolume::RandomPoints() (context menu function) and
*  it will draw this volume with current visualization settings. Random points
*  are generated in the bounding box of the top drawn volume. The points are
*  classified and drawn with the color of their deepest container. Only points
*  in visible nodes will be drawn.
*
* see http://root.cern.ch/root/htmldoc/gif/t_random1.jpg
*
*
*   3. Raytracing.
*    Can be called from TGeoVolume::RandomRays() (context menu of volumes) and
*  will shoot rays from a given point in the local reference frame with random
*  directions. The intersections with displayed nodes will appear as segments
*  having the color of the touched node. Drawn geometry will be then made invisible
*  in order to enhance rays.
*
* see http://root.cern.ch/root/htmldoc/gif/t_random2.jpg
*
*    IMPORTANT NOTE
*    To be compiled, this program requires a Fortran compiler supporting
*    recursive calls.
*
**********************************************************************
* NOT YET MAPPED FROM OLD g2root
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

      parameter (nwpaw=4000000)
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
*      call rzldir(' ',' ')

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
      parameter (nwpaw=4000000)
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
      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS),nvflags(MAXPOS),
     +npflags(MAXPOS),nppflags(MAXPOS)

      CHARACTER*4 KSHAP(30),klshap(30)
      character*20 matname,medname
      character*16 cname,mname,pname, rname
      character*(*) fname
      character*256 line
      character*128 creals
      character*16 astring,astring2,ptrname
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
     +'//   (see ROOT class TGeoManager for an example of use)',/,
     +'//',/,
     +'gSystem->Load("libGeom");',/,
     +'TGeoRotation *rot;',/,
     +'TGeoNode *Node, *Node1;')

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
 490  format(/,'TGeoManager *',a,' = new TGeoManager("',a,'","',a,'");'
     +,/)
      IF(JVOLUM.NE.0 ) NVOLUM = IQ(JVOLUM-2)
      IF(JMATE.NE.0 )  NMATE  = IQ(JMATE-2)
      IF(JTMED.NE.0 )  NTMED  = IQ(JTMED-2)
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
            write(line,3000)astring(1:nc),matname(1:ncn),creals(1:ncr)
 3000       format('TGeoMaterial *mat',a,' = new TGeoMaterial("',a,
     +       '"',a,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
            write(line,3005) astring(1:nc),imat
 3005       format(4x,'mat',a,'->SetUniqueID(',i4,');')
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
            write(line,3010)astring(1:nc),matname(1:ncn),mname(1:ncm),
     +      q(jma+8)
 3010       format('TGeoMixture *mat',a,' = new TGeoMixture("',a,'",',a,
     +              ',',g14.6,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
            write(line,3011) astring(1:nc),imat
 3011       format(4x,'mat',a,'->SetUniqueID(',i4,');')
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
            do 292 im=1,nm
               call toint(im-1,astring2,nc2)
               pmixt(1) = q(jmixt+im)
               pmixt(2) = q(jmixt+nm+im)
               pmixt(3) = q(jmixt+2*nm+im)
               call toreals(3,pmixt,creals,ncr)
               line=' '
               write(line,3020)astring(1:nc),astring2(1:nc2),
     +         creals(1:ncr)
 3020          format(4x,'mat',a,'->DefineElement(',a,a,');')
               nch = lenocc(line)
               write(51,'(a)')line(1:nch)
 292        continue
         endif
 300  continue
* Print Tracking Media
* ====================
      write(51,3069)
 3069 format(/,
     +'//-----------List of Tracking Media--------------',/)
      do 350 itmed=1,ntmed
         jtm=lq(jtmed-itmed)
         if(jtm.eq.0)go to 350
         imat=q(jtm+6)
         call toint(imat,astring2,ncm)
         call uhtoc(iq(jtm+1),4,medname,20)
         ncn=lenocc(medname)
         call toint(itmed,astring,nc)
         call toreals(8,q(jtm+7),creals,ncr)
         line=' '
         write(line,3050)astring(1:nc),medname(1:ncn),astring(1:nc),
     +     astring2(1:ncm),creals(1:ncr)
 3050    format('TGeoMedium *med',a,' = new TGeoMedium("',a,'",',a,
     +     ',',a,a,');')
         nch = lenocc(line)
         write(51,'(a)')line(1:nch)
 350  continue
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
         write(line,1000)ptrname(1:nch),ptrname(1:nch),
     +     creals(1:ncr)
 1000    format('TGeoRotation *',a,
     +        ' = new TGeoRotation("',a,'"',a,');')
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
  50  continue
C----------------------------------------------
      do 77 ivo = 1,nvolum
 77   nvflags(ivo) = 0
      nlevel = 0
      call markdiv(1,1)

      do 200 ivo = 1,nvolum
         if (nvflags(ivo).eq.2) goto 200
         jv=lq(jvolum-ivo)
         if (jv.eq.0)go to 200
         cname=' '
         if(.not.map_found(iq(jvolum+ivo),cname)) then
            write(cname,'(a4)')iq(jvolum+ivo)
         endif
         call volume(cname,q(jv+1),0,0)
 200  continue

      do 88 ivo = 1,nvolum
         nvflags(ivo) = 0
         npflags(ivo) = 0
         nppflags(ivo) = 0
 88   continue
* Print volume positioning (ROOT nodes)
* ========================

      nnodes = 1
      nlevel = 0
      if (nvolum.gt.0) then
         call node(1,1,1)
         write(51,3023)
 3023    format(/,'//-----------List of Nodes--------------',/)
         do 89 ivo = 1,nvolum
 89      nvflags(ivo) = 0
         call node(1,1,0)
      endif

      write(51,2223)
      write(51,2222)
2222  format('}')
2223  format(' gGeoManager->CloseGeometry();')
      close(51)
      end

*_______________________________________________________________________
      Subroutine markdiv2(ivo,nuserm)
      call markdiv(ivo,nuserm)
      End

*_______________________________________________________________________
       Subroutine markdiv(ivo,nuserm)
*
*             Process one node (volume with contents)
*KEEP,HCBOOK.
      parameter (nwpaw=4000000)
      parameter (MAXPOS=250000)
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

      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS),nvflags(MAXPOS),
     +npflags(MAXPOS),nppflags(MAXPOS)

*      dimension qjv(1000)
      character*16 cname
*---------------------------------------------------------------------
      nlevel = nlevel + 1
      nodeold(nlevel) = nnodes
      jv=lq(jvolum-ivo)
      ishape = q(jv+2)
      nin = q(jv+3)
*-  Loop subnodes
      if(nin.eq.0)then
         nlevel=nlevel-1
         return
      endif
      call cdnode(nodeold(nlevel))
      if(nin.gt.0)then
            if (nvflags(ivo).ne.0) then
               goto 996
            endif
            nvflags(ivo)=1
*                    Volume has positioned contents
         do 300 in=1,nin
            jin=lq(jv-in)
            ivom=q(jin+2)
            nuser  = q(jin+3)
            jinvom = lq(jvolum-ivom)
            ninvom = q(jinvom+3)
            cname=' '
            write(cname,'(a4)')iq(jvolum+ivom)
            n1=lenocc(cname)
            if(ninvom.ge.0)then
               nnodes = nnodes+1
               if (nnodes.gt.MAXPOS) then
                  print *,'Too many nodes =',nnodes
                  go to 300
               endif
               if (nodepos(nnodes).eq.0) then
                  nodepos(nnodes) = 1
               endif
            endif
            if(ninvom.ne.0) then
               call markdiv2(ivom,nuser)
            endif
 300     continue
      else
         nnodes = nnodes+1
         if (nodediv(nnodes).eq.0) then
            nodediv(nnodes) = 1
         endif
         jin=lq(jv-1)
         ivod=q(jin+2)
         if (nvflags(ivod).gt.0) goto 996
         cname=' '
         write(cname,'(a4)')iq(jvolum+ivod)
         n1=lenocc(cname)
         Print 200, cname(1:n1)
 200     format('Division volume', a4)
         call markdiv2(ivod,0)
         nvflags(ivod) = 2
      endif

996   continue
      nlevel = nlevel - 1
      if (nlevel.gt.0)call cdnode(nodeold(nlevel))
      end
*_______________________________________________________________________
      subroutine volume(cname,qjv,iposp,ifirst)
*KEEP,HCBOOK.
      parameter (nwpaw=4000000)
      parameter (MAXPOS=250000)
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
      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS),nvflags(MAXPOS),
     +npflags(MAXPOS),nppflags(MAXPOS)
      character *(*) cname
      character*16 astring,cmater, pname, rname
      character*128 creals
      character*256 line
      real qjv(100)
      double precision RADDEG
      dimension dummypars(100)
      DIMENSION NPARS(30)
      CHARACTER*6 KSHAP(30)
      data dummypars/100*0./

      DATA KSHAP/'Box','Trd1','Trd2','Trap','Tube','Tubs','Cone',
     +           'Cons','Sphere','Para','Pgon','Pcon','Eltu','Hype',
     + 13*'      ','Gtra','Ctub','    '/
      DATA NPARS/3,4,5,11,3,5,5,7,6,6,10,9,3,4,13*0,12,11,0/
*________________________________________________________________________
*

      RADDEG = 57.2957795130823209
      n1=lenocc(cname)

**      print *, 'VOLUME n1=',n1,' cname=',cname(1:n1)
      ishape = qjv(2)
      numed  = qjv(4)
      jtm    = lq(jtmed-numed)
      call toint(numed,astring,nc)
      cmater='med'//astring(1:nc)
      ncmat = lenocc(cmater)
      nord  = qjv(1)
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
**HYPE
      if (ishape.eq.14) then
         hyrmin = qjv(jpar+1)
         hyrmax = qjv(jpar+2)
         hydz = qjv(jpar+3)
         hyst = qjv(jpar+4)
         dummypars(1) = hyrmin
         dummypars(2) = hyst
         dummypars(3) = hyrmax
         dummypars(4) = hyst
         dummypars(5) = hydz
         npar0 = -5
       endif

*      print 2351, cname(1:n1),kshap(ishape)
* 2351      format('Volume:',a, ' shape=',a)
      if (npar.le.0) then
**       print 2352, cname(1:n1),kshap(ishape)
** 2352      format('Warning, volume with 0 parameters:',a, ' shape=',a)
       return
      endif
      if(npar0.le.0)then
         call toreals(-npar0,dummypars(1),creals,ncr)
      else
         call toreals(npar0,qjv(7),creals,ncr)
      endif
      line=' '
      nshape = lenocc(kshap(ishape))
      call ptname(cname, pname)
      np = lenocc(pname)
      call realname(cname, rname)
      nrr = lenocc(rname)
      if (iposp.eq.0) then
         write(line,2000)pname(1:np),kshap(ishape)(1:nshape)
     +         ,rname(1:nrr),cmater(1:ncmat),creals(1:ncr)
      else
         if (ifirst.eq.1) then
            write(line,2001)pname(1:np),rname(1:nrr),cmater(1:ncmat)
            nch=lenocc(line)
            write(51,'(a)')line(1:nch)
         endif
         line=' '
         write(line,2002)pname(1:np),kshap(ishape)(1:nshape)
     +         ,rname(1:nrr),cmater(1:ncmat),creals(1:ncr)
         nch=lenocc(line)
         write(51,'(a)')line(1:nch)
      endif
2000  format(
     + 'TGeoVolume',' *',a,' = gGeoManager->Make',a,'("',a,'",'
     +,a,a,');')
2001  format('TGeoVolumeMulti *',a,' = gGeoManager->MakeVolumeMulti("'
     +,a,'", ',a,');')
2002  format(' ',a,'->AddVolume(gGeoManager->Make',a,'("',a,'",',
     +a,a,'));')
      nch = lenocc(line)
      if (iposp.eq.0) write(51,'(a)')line(1:nch)
      if(ishape.eq.11)then
         ndz=qjv(10)
         do iz=1,ndz
            call toreals(3,qjv(11+(iz-1)*3),creals,ncr)
            line=' '
            call toint(iz-1,astring,nci)
            if (iposp.eq.0) then
            write(line,2010)pname(1:np),astring(1:nci),creals(1:ncr)
            else
            write(line,2011)pname(1:np),astring(1:nci),creals(1:ncr)
            endif
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
            if (iposp.eq.0) then
            write(line,2010)pname(1:np),astring(1:nci),creals(1:ncr)
            else
            write(line,2011)pname(1:np),astring(1:nci),creals(1:ncr)
            endif
            nch = lenocc(line)
            write(51,'(a)')line(1:nch)
         enddo
      endif
2010  format(2x,'((TGeoPcon*)',a,'->GetShape())->DefineSection(',
     + a,a,');')
2011  format(2x,'((TGeoPcon*)',a,'->GetLastShape())->DefineSection(',
     + a,a,');')
*   Any attributes set ?
      lseen  = qjv(npar+8)
      lstyle = qjv(npar+9)
      lwidth = qjv(npar+10)
      lcolor = qjv(npar+11)
      lfill  = qjv(npar+12)
      if (lstyle.le.0) lstyle = 1
      if (lwidth.le.0) lwidth = 1
      if (lcolor.lt.0) lcolor = 1
      if (lfill.lt.0)  lfill  = 0
*      if(ivo.eq.1)lseen=0
*      if(nord.lt.0)then
*         print *,'ordering : ',-nord
*         call toint(-nord,creals,ncr)
*      endif
      if ((iposp.eq.0).or.(ifirst.eq.1)) then
      if(lseen.ne.1)then
         call toint(lseen,creals,ncr)
         write(51,195)pname(1:np),creals(1:ncr)
195        format(2x,a,'->SetVisibility(',a,');')
      endif
      if(lstyle.ne.1)then
         call toint(lstyle,creals,ncr)
         write(51,196)pname(1:np),creals(1:ncr)
196        format(2x,a,'->SetLineStyle(',a,');')
      endif
      if(lwidth.ne.1)then
         call toint(lwidth,creals,ncr)
         write(51,197)pname(1:np),creals(1:ncr)
197        format(2x,a,'->SetLineWidth(',a,');')
      endif
      if(lcolor.ne.1)then
         call toint(lcolor,creals,ncr)
         write(51,198)pname(1:np),creals(1:ncr)
198        format(2x,a,'->SetLineColor(',a,');')
      endif
      if(lfill.ne.0)then
         call toint(lfill,creals,ncr)
         write(51,199)pname(1:np),creals(1:ncr)
199        format(2x,a,'->SetFillStyle(',a,');')
      endif
      endif
      end

*_______________________________________________________________________
      Subroutine node2(ivo,nuserm,iposp)
      call node(ivo,nuserm,iposp)
      End

*_______________________________________________________________________
       Subroutine node(ivo,nuserm,iposp)
*
*             Process one node (volume with contents)
*KEEP,HCBOOK.
      parameter (nwpaw=4000000)
      parameter (MAXPOS=250000)
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

      common/cnpos/nodepos(MAXPOS),nodediv(MAXPOS),nvflags(MAXPOS),
     +npflags(MAXPOS),nppflags(MAXPOS)

      dimension qjv(1000)
      character*16 cnode,cname,mname,anode,mother,pname, rname
      integer nmother
      character*256 line
      character*128 creals
      character*16 astring,astring1
      character*16 matrix
      character*256 matrixs
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
      call ptname(mname, mother)
      nmother = lenocc(mother)
      if((ivo.eq.1).and.(iposp.eq.0))then
         write(51,510)mother(1:nmother)
*         write(51,510)
510      format('gGeoManager->SetTopVolume(',a,');')
      endif
*-  Generate subnodes of this node (if any)
**      print 2346, iq(jvolum+ivo), nin
2346  format('Processing node:',a4,' nin=',i4)
      if(nin.eq.0)then
         nlevel=nlevel-1
         return
      endif
      call cdnode(nodeold(nlevel))
*      print 520, mother(1:nmother), ivo
520   format('mother ',a,' index ',i9)
      if(nin.gt.0)then
            if (nvflags(ivo).ne.0) then
               goto 996
            endif
            nvflags(ivo)=1
*                    Volume has positioned contents
         do 300 in=1,nin
            ifirst = 0
            icurrent = 0
            imulti = 0
            nci1 = 0
            jin=lq(jv-in)
            ivom=q(jin+2)
            nuser  = q(jin+3)
            imany = q(jin+8)
*            print *,'in=',in,' nuser=',nuser
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
               if (iposp.eq.1) then
                  if (npflags(ivom).eq.0) then
                     ifirst = 1
                     npflags(ivom) = 1
                  else
                     npflags(ivom) = npflags(ivom)+1
                  endif
               else
                  icurrent = nppflags(ivom)
                  call toint(icurrent,astring1,nci1)
                  imulti = 1
                  nppflags(ivom) = nppflags(ivom)+1
               endif
               npar = q(jin+9)
               call ucopy(q(jinvom+1),qjv(1),6)
               qjv(5) = npar
               call ucopy(q(jin+10),qjv(7),npar)
               call ucopy(q(jinvom+7),qjv(7+npar),6)
               call toint(in,astring,nci)
               mname=cname(1:n1)//astring(1:nci)
               if (iposp.eq.1) call volume(cname,qjv,iposp,ifirst)
            endif
            if(ninvom.ge.0)then
               nnodes = nnodes+1
               if (nnodes.gt.MAXPOS) then
                  print *,'Too many nodes =',nnodes
                  go to 300
               endif
               call toint(nnodes,anode,ncd)
               cnode = 'Node'//anode(1:ncd)
               if (nodepos(nnodes).eq.0) then
*                  write(51,4444)cblank(1:nlevel),anode(1:ncd)
 4444             format(a,'TNode *Node',a,';')
                  nodepos(nnodes) = 1
               endif
            else
               cnode = 'Node'
            endif
            nd=lenocc(cnode)
            call toreals(3,q(jin+5),creals,ncr)
            itrans = 1
            if ((abs(q(jin+5)).lt.1E-30).and.
     +          (abs(q(jin+6)).lt.1E-30).and.
     +          (abs(q(jin+7)).lt.1E-30)) then
               itrans = 0
            endif
            irot=q(jin+4)
            matrixs=' '
            if(irot.eq.0)then
               matrix='0'
               ncmatrix=1
               if (itrans.eq.0) then
                  matrixs='gGeoIdentity'
               else
                  matrixs='new TGeoTranslation('//creals(2:ncr)//')'
               endif
            else
               call toint(irot,astring,nci)
               matrix='rot'//astring(1:nci)
               ncmatrix=nci+3
               if (itrans.eq.0) then
                  matrixs=matrix(1:ncmatrix)
               else
                  matrixs='new TGeoCombiTrans('//creals(2:ncr)//','//
     +               matrix(1:ncmatrix)//')'
               endif
            endif
            call toint(nuser,astring,nci)
**            print *,' cname=',cname(1:n1), ' astring=',astring(1:nci)
            mname=cname(1:n1)//astring(1:nci)
            n2=lenocc(mname)
            ncmats=lenocc(matrixs)
            line=' '

            call ptname(cname, pname)
            np = lenocc(pname)
            if (imany.eq.1) then
               if (imulti.eq.0) then
               write(line,3000)cblank(1:nlevel),mother(1:nmother),
     +               pname(1:np), astring(1:nci), matrixs(1:ncmats)
 3000          format(a,a,'->AddNode(',a,',',a,',',a,');')
               else
               write(line,3002)cblank(1:nlevel),mother(1:nmother),
     +               pname(1:np), astring1(1:nci1),astring(1:nci),
     +               matrixs(1:ncmats)
 3002          format(a,a,'->AddNode(',a,'->GetVolume(',a,'),',a
     +                ,',',a,');')
               endif
            else
               if (imulti.eq.0) then
               write(line,3001)cblank(1:nlevel),mother(1:nmother),
     +               pname(1:np), astring(1:nci), matrixs(1:ncmats)
 3001          format(a,a,'->AddNodeOverlap(',a,',',a,',',a,');')
               else
               write(line,3003)cblank(1:nlevel),mother(1:nmother),
     +               pname(1:np), astring1(1:nci1), astring(1:nci),
     +               matrixs(1:ncmats)
 3003          format(a,a,'->AddNodeOverlap(',a,'->GetVolume(',a,'),',a
     +                ,',',a,');')
               endif
            endif
            nch = lenocc(line)
            if (iposp.eq.0) write(51,'(a)')line(1:nch)
            npar=q(jv+5)
            if(ninvom.ne.0) then
               call node2(ivom,nuser,iposp)
            endif
 300     continue
      else
*         Print *,'===== DIVISION ====='
*         Print 4567,mother(1:nmother)
         nnodes = nnodes+1
         call toint(nnodes,anode,ncd)
         cnode = 'Nodiv'//anode(1:ncd)
         nd=lenocc(cnode)
         if (nodediv(nnodes).eq.0) then
             nodediv(nnodes) = 1
          endif
         jin=lq(jv-1)
         ivod=q(jin+2)
         if (nvflags(ivod).eq.1) goto 996
         cname=' '
*         if(.not.map_found(iq(jvolum+ivod),cname)) then
            write(cname,'(a4)')iq(jvolum+ivod)
*         endif
*         Print 4445,iq(jvolum+ivod)
 4445    format('daughter division', a4)
         n1=lenocc(cname)
*         if(cname(n1:n1).eq.'+')cname(n1:)='plus'
*         if(cname(n1:n1).eq.'-')cname(n1:)='minus'
*         n1=lenocc(cname)
*         cname=cname(1:n1)//'_0'
*         n2=lenocc(cname)
         iaxis=q(jin+1)
         call toint(iaxis,astring,nci)
         call toreals(3,q(jin+3),creals,ncr)
         line = ' '
         call ptname(cname, pname)
         np = lenocc(pname)
         call realname(cname, rname)
         nrr = lenocc(rname)
         write(line,995)cblank(1:nlevel),pname(1:np),mother(1:nmother),
     +      rname(1:nrr),astring(1:nci), creals(1:ncr)
 995    format(a,'TGeoVolume *',a,' = ',a,'->Divide("',a,'",',a,a,');')
         nch = lenocc(line)
         if (iposp.eq.0) write(51,'(a)')line(1:nch)

         call node2(ivod,0,iposp)
         nvflags(ivod) = 1
      endif

996   continue
      nlevel = nlevel - 1
      if (nlevel.gt.0)call cdnode(nodeold(nlevel))
      end

      subroutine realname(cname, pname)
      character *4 cname
      character *16 pname
      nind = 0
      pname = ' '
      write(pname,'(a4)')cname
      do i=1,4
          nind = nind+1
          pname(nind:nind)=cname(i:i)
          if(ichar(cname(i:i)).eq.0) then
             pname(nind:nind)=' '
             nind = nind+1
             pname(nind:nind)=' '
          endif
          if(ichar(cname(i:i)).eq.92) then
             pname(nind:nind)=char(92)
             nind = nind+1
             pname(nind:nind)=char(92)
          endif
          if(ichar(cname(i:i)).eq.34) then
             pname(nind:nind)=char(92)
             nind = nind+1
             pname(nind:nind)=char(34)
          endif
      enddo
*------ supress blanks
2333  if (pname(lenocc(pname):lenocc(pname)).eq.' ') then
         pname = pname(1:lenocc(pname)-1)
         goto 2333
      endif
      end

      subroutine ptname(cname, pname)
      character *4 cname
      character *16 pname
      pname = ' '
      write(pname,'(a4)')cname
      do i=1,4
          if(ichar(cname(i:i)).eq.0) then
             pname(i:i)='_'
             pname(5:5)='_'
          endif
          if(ichar(cname(i:i)).eq.92) then
             pname(i:i)='a'
             pname(5:5)='a'
          endif
          if(cname(i:i).eq.'?') then
             pname(i:i)='b'
             pname(5:5)='b'
          endif
          if(cname(i:i).eq.'`') then
             pname(i:i)='c'
             pname(5:5)='c'
          endif
          if(cname(i:i).eq.' ') then
             pname(i:i)='_'
          endif
          if(cname(i:i).eq.'+') then
             pname(i:i)='d'
             pname(5:5)='d'
          endif
          if(cname(i:i).eq.'-') then
             pname(i:i)='e'
             pname(5:5)='e'
          endif
          if(cname(i:i).eq.'*') then
             pname(i:i)='f'
             pname(5:5)='f'
          endif
          if(cname(i:i).eq.'/') then
             pname(i:i)='g'
             pname(5:5)='g'
          endif
          if(cname(i:i).eq.'.') then
             pname(i:i)='h'
             pname(5:5)='h'
          endif
          if(cname(i:i).eq.'''') then
             pname(i:i)='i'
             pname(5:5)='i'
          endif
          if(cname(i:i).eq.';') then
             pname(i:i)='j'
             pname(5:5)='j'
          endif
          if(cname(i:i).eq.':') then
             pname(i:i)='k'
             pname(5:5)='k'
          endif
          if(cname(i:i).eq.',') then
             pname(i:i)='l'
             pname(5:5)='l'
          endif
          if(cname(i:i).eq.'<') then
             pname(i:i)='m'
             pname(5:5)='m'
          endif
          if(cname(i:i).eq.'>') then
             pname(i:i)='n'
             pname(5:5)='n'
          endif
          if(cname(i:i).eq.'!') then
             pname(i:i)='o'
             pname(5:5)='o'
          endif
          if(cname(i:i).eq.'@') then
             pname(i:i)='p'
             pname(5:5)='p'
          endif
          if(cname(i:i).eq.'#') then
             pname(i:i)='q'
             pname(5:5)='q'
          endif
          if(cname(i:i).eq.'$') then
             pname(i:i)='r'
             pname(5:5)='r'
          endif
          if(cname(i:i).eq.'%') then
             pname(i:i)='s'
             pname(5:5)='s'
          endif
          if(cname(i:i).eq.'^') then
             pname(i:i)='t'
             pname(5:5)='t'
          endif
          if(cname(i:i).eq.'&') then
             pname(i:i)='u'
             pname(5:5)='u'
          endif
          if(cname(i:i).eq.'(') then
             pname(i:i)='v'
             pname(5:5)='v'
          endif
          if(cname(i:i).eq.')') then
             pname(i:i)='x'
             pname(5:5)='x'
          endif
          if(cname(i:i).eq.'[') then
             pname(i:i)='y'
             pname(5:5)='y'
          endif
          if(cname(i:i).eq.']') then
             pname(i:i)='z'
             pname(5:5)='z'
          endif
          if(cname(i:i).eq.'{') then
             pname(i:i)='c'
             pname(5:5)='a'
          endif
          if(cname(i:i).eq.'}') then
             pname(i:i)='c'
             pname(5:5)='b'
          endif
          if(cname(i:i).eq.'=') then
             pname(i:i)='c'
             pname(5:5)='d'
          endif
          if(cname(i:i).eq.'~') then
             pname(i:i)='c'
             pname(5:5)='e'
          endif
          if(cname(i:i).eq.'|') then
             pname(i:i)='c'
             pname(5:5)='f'
          endif
      enddo
      if ((ichar(pname(1:1)).ge.48).and.
     + (ichar(pname(1:1)).le.57)) then
         pname='Z'//pname(1:lenocc(pname))
      endif
      end

      subroutine cdnode(node)
      common/clevel/nodeold(20),nlevel
      character*16 anode
      character*16 cblank
      data cblank/' '/
      call toint(node,anode,ncd)
      if (nlevel.gt.1)then
*         write(51,1000)cblank(1:nlevel-1),anode(1:ncd)
1000     format(a,'Node',a,'->cd();')
      else
*         write(51,1001)anode(1:ncd)
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
      parameter (nwpaw=4000000)
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
C                Create a permanent link area for master pointers
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
