// @(#)root/geom:$Name:  $:$Id: TGeoManager.h,v 1.77 2006/03/27 09:28:10 brun Exp $
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoManager
#define ROOT_TGeoManager

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TGeoNodeCache
#include "TGeoCache.h"
#endif

// forward declarations
class TVirtualGeoTrack;
class TGeoNode;
class TGeoPhysicalNode;
class TGeoVolume;
class TGeoVolumeMulti;
class TGeoMatrix;
class TGeoHMatrix;
class TGeoMaterial;
class TGeoMedium;
class TGeoShape;
class TVirtualGeoPainter;
class THashList;

/*************************************************************************
 * TGeoManager - class description
 *
 *************************************************************************/


class TGeoManager : public TNamed
{
protected:
   static Bool_t         fgLock;            //! Lock preventing a second geometry to be loaded
private :
   Double_t              fStep;             //! step to be done from current point and direction
   Double_t              fSafety;           //! safety radius from current point
   Double_t              fLastSafety;       //! last computed safety radius
   Double_t              fPhimin;           //! lowest range for phi cut
   Double_t              fPhimax;           //! highest range for phi cut
   Double_t              fTmin;             //! lower time limit for tracks drawing
   Double_t              fTmax;             //! upper time limit for tracks drawing
   Int_t                 fLevel;            //! current geometry level;
   Int_t                 fNNodes;           // total number of physical nodes
   TString               fPath;             //! path to current node
   TString               fParticleName;     //! particles to be drawn
   Double_t              fNormal[3];        //! cosine of incident angle on current checked surface
   Double_t             *fCldir;            //! unit vector to current closest shape
   Double_t             *fCldirChecked;     //! unit vector to current checked shape
   Double_t             *fPoint;            //![3] current point
   Double_t             *fDirection;        //![3] current direction
   Double_t              fVisDensity;       // transparency threshold by density
   Int_t                 fExplodedView;     // exploded view mode
   Int_t                 fVisOption;        // global visualization option
   Int_t                 fVisLevel;         // maximum visualization depth
   Int_t                 fNsegments;        // number of segments to approximate circles
   Int_t                 fNtracks;          // number of tracks
   Int_t                 fMaxVisNodes;      // maximum number of visible nodes
   TVirtualGeoTrack     *fCurrentTrack;     //! current track
   Int_t                 fNpdg;             // number of different pdg's stored
   Int_t                 fPdgId[256];       // pdg conversion table
   Bool_t                fSearchOverlaps;   //! flag set when an overlapping cluster is searched
   Bool_t                fCurrentOverlapping; //! flags the type of the current node
   Bool_t                fLoopVolumes;      //! flag volume lists loop
   Bool_t                fStartSafe;        //! flag a safe start for point classification
   Bool_t                fIsEntering;       //! flag if current step just got into a new node
   Bool_t                fIsExiting;        //! flag that current track is about to leave current node
   Bool_t                fIsStepEntering;   //! flag that next geometric step will enter new volume
   Bool_t                fIsStepExiting;    //! flaag that next geometric step will exit current volume
   Bool_t                fIsOutside;        //! flag that current point is outside geometry
   Bool_t                fIsOnBoundary;     //! flag that current point is on some boundary
   Bool_t                fIsSameLocation;   //! flag that a new point is in the same node as previous
   Bool_t                fIsNullStep;       //! flag that last geometric step was null
   Bool_t                fStreamVoxels;     // flag to allow voxelization I/O
   Bool_t                fIsGeomReading;    //! flag set when reading geometry
   Bool_t                fPhiCut;           // flag for phi cuts
   Bool_t                fTimeCut;          // time cut for tracks
   Bool_t                fDrawExtra;        //! flag that the list of physical nodes has to be drawn
   Bool_t                fMatrixTransform;  //! flag for using GL matrix
   Bool_t                fMatrixReflection; //! flag for GL reflections
   Bool_t                fActivity;         //! switch ON/OFF volume activity (default OFF - all volumes active))
   Bool_t                fIsNodeSelectable; //! flag that nodes are the selected objects in pad rather than volumes
   TGeoNodeCache        *fCache;            //! cache for physical nodes
   TVirtualGeoPainter   *fPainter;          //! current painter

   TObjArray            *fMatrices;         //-> list of local transformations
   TObjArray            *fShapes;           //-> list of shapes
   TObjArray            *fVolumes;          //-> list of volumes
   TObjArray            *fPhysicalNodes;    //-> list of physical nodes
   TObjArray            *fGShapes;          //! list of runtime shapes
   TObjArray            *fGVolumes;         //! list of runtime volumes
   TObjArray            *fTracks;           //-> list of tracks attached to geometry
   TObjArray            *fPdgNames;         //-> list of pdg names for tracks
   TList                *fMaterials;        //-> list of materials
   TList                *fMedia;            //-> list of tracking media
   TObjArray            *fNodes;            //-> current branch of nodes
   TObjArray            *fOverlaps;         //-> list of geometrical overlaps
   UChar_t              *fBits;             //! bits used for voxelization
   TGeoVolume           *fCurrentVolume;    //! current volume
   TGeoVolume           *fTopVolume;        //! top level volume in geometry
   TGeoNode             *fCurrentNode;      //! current node
   TGeoNode             *fTopNode;          //! top physical node
   TGeoNode             *fLastNode;         //! last searched node
   TGeoNode             *fNextNode;         //! next node that will be crossed
   TGeoVolume           *fMasterVolume;     // master volume
   TGeoHMatrix          *fCurrentMatrix;    //! current global matrix
   TGeoHMatrix          *fGLMatrix;         // matrix to be used for view transformations
   TObjArray            *fUniqueVolumes;    //-> list of unique volumes
   TGeoShape            *fClippingShape;    //! clipping shape for raytracing
   TGeoElementTable     *fElementTable;     //! table of elements

   Int_t                *fNodeIdArray;      //! array of node id's
   Int_t                 fIntSize;          //! int buffer size
   Int_t                 fDblSize;          //! dbl buffer size
   Int_t                 fOverlapSize;      //! current size of fOverlapClusters
   Int_t                 fOverlapMark;      //! current recursive position in fOverlapClusters
   Int_t                *fIntBuffer;        //! transient int buffer
   Int_t                *fOverlapClusters;  //! internal array for overlaps
   Int_t                 fNLevel;           // maximum accepted level in geometry
   Int_t                 fNmany;            //! number of overlapping nodes on current branch
   Double_t             *fDblBuffer;        //! transient dbl buffer
   Double_t              fLastPoint[3];     //! last point for which safety was computed
   TGeoVolume           *fPaintVolume;      //! volume currently painted
   THashList            *fHashVolumes;      //! hash list of volumes providing fast search
   THashList            *fHashGVolumes;     //! hash list of group volumes providing fast search
//--- private methods
   void                   BuildCache(Bool_t dummy=kFALSE, Bool_t nodeid=kFALSE);
   void                   BuildIdArray();
   TGeoNode              *FindInCluster(Int_t *cluster, Int_t nc);
   Int_t                  GetTouchedCluster(Int_t start, Double_t *point, Int_t *check_list,
                                            Int_t ncheck, Int_t *result);
   Bool_t                 IsLoopingVolumes() const     {return fLoopVolumes;}
   void                   Init();
   void                   SetLoopVolumes(Bool_t flag=kTRUE) {fLoopVolumes=flag;}
   void                   SafetyOverlaps();
   void                   Voxelize(Option_t *option = 0);

public:
   // constructors
   TGeoManager();
   TGeoManager(const char *name, const char *title);
   // destructor
   virtual ~TGeoManager();
   //--- adding geometrical objects
   Int_t                  AddMaterial(const TGeoMaterial *material);
   Int_t                  AddOverlap(const TNamed *ovlp);
   Int_t                  AddTransformation(const TGeoMatrix *matrix);
   Int_t                  AddShape(const TGeoShape *shape);
   Int_t                  AddTrack(Int_t id, Int_t pdgcode, TObject *particle=0);
   Int_t                  AddVolume(TGeoVolume *volume);
   void                   ClearOverlaps();
   void                   RegisterMatrix(const TGeoMatrix *matrix);
   void                   SortOverlaps();
   //--- browsing and tree navigation
   void                   Browse(TBrowser *b);
   void                   SetVisibility(TObject *obj, Bool_t vis);
   virtual Bool_t         cd(const char *path=""); // *MENU*
   void                   CdNode(Int_t nodeid);
   void                   CdDown(Int_t index);
   void                   CdUp();
   void                   CdTop();
   void                   GetBranchNames(Int_t *names) const;
   void                   GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const;
   void                   GetBranchOnlys(Int_t *isonly) const;
   Int_t                  GetNmany() const {return fNmany;}
   const char            *GetPdgName(Int_t pdg) const;
   void                   SetPdgName(Int_t pdg, const char *name);
   Bool_t                 IsFolder() const { return kTRUE; }
   //--- visualization settings
   virtual void           Edit(Option_t *option="");  // *MENU*
   void                   BombTranslation(const Double_t *tr, Double_t *bombtr);
   void                   UnbombTranslation(const Double_t *tr, Double_t *bombtr);
   void                   ClearAttributes(); // *MENU*
   void                   DefaultAngles();   // *MENU*
   void                   DefaultColors();   // *MENU*
   TGeoShape             *GetClippingShape() const {return fClippingShape;}
   Int_t                  GetNsegments() const;
   TVirtualGeoPainter    *GetGeomPainter();
   TVirtualGeoPainter    *GetPainter() const {return fPainter;}
   Int_t                  GetBombMode() const  {return fExplodedView;}
   void                   GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const;
   Int_t                  GetMaxVisNodes() const {return fMaxVisNodes;}
   Bool_t                 GetTminTmax(Double_t &tmin, Double_t &tmax) const;
   Double_t               GetTmax() const {return fTmax;}
   TGeoVolume            *GetPaintVolume() const {return fPaintVolume;}
   Double_t               GetVisDensity() const  {return fVisDensity;}
   Int_t                  GetVisLevel() const;
   Int_t                  GetVisOption() const;
   Bool_t                 IsInPhiRange() const;
   Bool_t                 IsDrawingExtra() const {return fDrawExtra;}
   Bool_t                 IsNodeSelectable() const {return fIsNodeSelectable;}
   Bool_t                 IsVisLeaves() const {return (fVisOption==1)?kTRUE:kFALSE;}
   void                   ModifiedPad() const;
   void                   OptimizeVoxels(const char *filename="tgeovox.C"); // *MENU*
   void                   SetClipping(Bool_t flag=kTRUE) {SetClippingShape(((flag)?fClippingShape:0));} // *MENU*
   void                   SetClippingShape(TGeoShape *clip);
   void                   SetExplodedView(Int_t iopt=0); // *MENU*
   void                   SetPhiRange(Double_t phimin=0., Double_t phimax=360.);
   void                   SetNsegments(Int_t nseg); // *MENU*
   void                   SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3,                                         Double_t bombr=1.3); // *MENU*
   void                   SetPaintVolume(TGeoVolume *vol) {fPaintVolume = vol;}
   void                   SetTopVisible(Bool_t vis=kTRUE);
   void                   SetTminTmax(Double_t tmin=0, Double_t tmax=999);
   void                   SetDrawExtraPaths(Bool_t flag=kTRUE) {fDrawExtra=flag;}
   void                   SetNodeSelectable(Bool_t flag=kTRUE) {fIsNodeSelectable=flag;}
   void                   SetVisDensity(Double_t dens=0.01); // *MENU*
   void                   SetVisLevel(Int_t level=3);   // *MENU*
   void                   SetVisOption(Int_t option=0);
   void                   ViewLeaves(Bool_t flag=kTRUE); // *TOGGLE* *GETTER=IsVisLeaves
   void                   SaveAttributes(const char *filename="tgeoatt.C"); // *MENU*
   void                   RestoreMasterVolume(); // *MENU*
   void                   SetMaxVisNodes(Int_t maxnodes=10000); // *MENU*
   //--- geometry checking
   void                   AnimateTracks(Double_t tmin=0, Double_t tmax=5E-8, Int_t nframes=200, Option_t *option="/*"); // *MENU*
   void                   CheckGeometry(Option_t *option="");
   void                   CheckOverlaps(Double_t ovlp=0.1, Option_t *option=""); // *MENU*
   void                   CheckPoint(Double_t x=0,Double_t y=0, Double_t z=0, Option_t *option=""); // *MENU*
   void                   DrawCurrentPoint(Int_t color=2); // *MENU*
   void                   DrawTracks(Option_t *option=""); // *MENU*
   void                   SetParticleName(const char *pname) {fParticleName=pname;}
   const char            *GetParticleName() const {return fParticleName.Data();}
   void                   DrawPath(const char *path);
   void                   PrintOverlaps() const; // *MENU*
   void                   RandomPoints(const TGeoVolume *vol, Int_t npoints=10000, Option_t *option="");
   void                   RandomRays(Int_t nrays=1000, Double_t startx=0, Double_t starty=0, Double_t startz=0);
   TGeoNode              *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil=1E-5,
                                       const char *g3path="");
   void                   Test(Int_t npoints=1000000, Option_t *option=""); // *MENU*
   void                   TestOverlaps(const char* path=""); // *MENU*
   Double_t               Weight(Double_t precision=0.01, Option_t *option="va"); // *MENU*

   //--- GEANT3-like geometry creation
   TGeoVolume            *Division(const char *name, const char *mother, Int_t iaxis, Int_t ndiv,
                                         Double_t start, Double_t step, Int_t numed=0, Option_t *option="");
   void                   Matrix(Int_t index, Double_t theta1, Double_t phi1,
                                       Double_t theta2, Double_t phi2,
                                       Double_t theta3, Double_t phi3);
   TGeoMaterial          *Material(const char *name, Double_t a, Double_t z, Double_t dens, Int_t uid, Double_t radlen=0, Double_t intlen=0);
   TGeoMaterial          *Mixture(const char *name, Float_t *a, Float_t *z, Double_t dens,
                                        Int_t nelem, Float_t *wmat, Int_t uid);
   TGeoMaterial          *Mixture(const char *name, Double_t *a, Double_t *z, Double_t dens,
                                        Int_t nelem, Double_t *wmat, Int_t uid);
   TGeoMedium            *Medium(const char *name, Int_t numed, Int_t nmat, Int_t isvol,
                                       Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                                       Double_t stemax, Double_t deemax, Double_t epsil,
                                       Double_t stmin);
   void                   Node(const char *name, Int_t nr, const char *mother,
                                     Double_t x, Double_t y, Double_t z, Int_t irot,
                                     Bool_t isOnly, Float_t *upar, Int_t npar=0);
   void                   Node(const char *name, Int_t nr, const char *mother,
                                     Double_t x, Double_t y, Double_t z, Int_t irot,
                                     Bool_t isOnly, Double_t *upar, Int_t npar=0);
   TGeoVolume            *Volume(const char *name, const char *shape, Int_t nmed,
                                       Float_t *upar, Int_t npar=0);
   TGeoVolume            *Volume(const char *name, const char *shape, Int_t nmed,
                                       Double_t *upar, Int_t npar=0);
   void                   SetVolumeAttribute(const char *name, const char *att, Int_t val);
   //--- geometry building
   void                   BuildDefaultMaterials();
   void                   CloseGeometry(Option_t *option="d");
   Bool_t                 IsClosed() const {return ((fCache==0)?kFALSE:kTRUE);}
   TGeoVolume            *MakeArb8(const char *name, const TGeoMedium *medium,
                                     Double_t dz, Double_t *vertices=0);
   TGeoVolume            *MakeBox(const char *name, const TGeoMedium *medium,
                                     Double_t dx, Double_t dy, Double_t dz);
   TGeoVolume            *MakeCone(const char *name, const TGeoMedium *medium,
                                      Double_t dz, Double_t rmin1, Double_t rmax1,
                                      Double_t rmin2, Double_t rmax2);
   TGeoVolume            *MakeCons(const char *name, const TGeoMedium *medium,
                                      Double_t dz, Double_t rmin1, Double_t rmax1,
                                      Double_t rmin2, Double_t rmax2,
                                      Double_t phi1, Double_t phi2);
   TGeoVolume            *MakeCtub(const char *name, const TGeoMedium *medium,
                                      Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                                      Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoVolume            *MakeEltu(const char *name, const TGeoMedium *medium,
                                      Double_t a, Double_t b, Double_t dz);
   TGeoVolume            *MakeGtra(const char *name, const TGeoMedium *medium,
                                   Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
                                   Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                   Double_t tl2, Double_t alpha2);
   TGeoVolume            *MakePara(const char *name, const TGeoMedium *medium,
                                     Double_t dx, Double_t dy, Double_t dz,
                                     Double_t alpha, Double_t theta, Double_t phi);
   TGeoVolume            *MakePcon(const char *name, const TGeoMedium *medium,
                                      Double_t phi, Double_t dphi, Int_t nz);
   TGeoVolume            *MakeParaboloid(const char *name, const TGeoMedium *medium,
                                      Double_t rlo, Double_t rhi, Double_t dz);
   TGeoVolume            *MakeHype(const char *name, const TGeoMedium *medium,
                                      Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz);
   TGeoVolume            *MakePgon(const char *name, const TGeoMedium *medium,
                                      Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoVolume            *MakeSphere(const char *name, const TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax,
                                     Double_t themin=0, Double_t themax=180,
                                     Double_t phimin=0, Double_t phimax=360);
   TGeoVolume            *MakeTorus(const char *name, const TGeoMedium *medium, Double_t r,
                                    Double_t rmin, Double_t rmax, Double_t phi1=0, Double_t dphi=360);
   TGeoVolume            *MakeTrap(const char *name, const TGeoMedium *medium,
                                   Double_t dz, Double_t theta, Double_t phi, Double_t h1,
                                   Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                   Double_t tl2, Double_t alpha2);
   TGeoVolume            *MakeTrd1(const char *name, const TGeoMedium *medium,
                                      Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoVolume            *MakeTrd2(const char *name, const TGeoMedium *medium,
                                      Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2,
                                      Double_t dz);
   TGeoVolume            *MakeTube(const char *name, const TGeoMedium *medium,
                                      Double_t rmin, Double_t rmax, Double_t dz);
   TGeoVolume            *MakeTubs(const char *name, const TGeoMedium *medium,
                                      Double_t rmin, Double_t rmax, Double_t dz,
                                      Double_t phi1, Double_t phi2);
   TGeoVolume            *MakeXtru(const char *name, const TGeoMedium *medium,
                                   Int_t nz);

   TGeoPhysicalNode      *MakePhysicalNode(const char *path=0);
   void                   ClearPhysicalNodes(Bool_t mustdelete=kFALSE);
   TVirtualGeoTrack      *MakeTrack(Int_t id, Int_t pdgcode, TObject *particle);
   TGeoVolumeAssembly    *MakeVolumeAssembly(const char *name);
   TGeoVolumeMulti       *MakeVolumeMulti(const char *name, const TGeoMedium *medium);
   void                   SetTopVolume(TGeoVolume *vol);

   //--- geometry queries
   TGeoNode              *CrossBoundaryAndLocate(Bool_t downwards, TGeoNode *skipnode);
   TGeoNode              *FindNextBoundary(Double_t stepmax=TGeoShape::Big(),const char *path="");
   TGeoNode              *FindNextDaughterBoundary(Double_t *point, Double_t *dir, Int_t &idaughter, Bool_t compmatrix=kFALSE);
   TGeoNode              *FindNextBoundaryAndStep(Double_t stepmax=TGeoShape::Big(), Bool_t compsafe=kFALSE);
   TGeoNode              *FindNode(Bool_t safe_start=kTRUE);
   TGeoNode              *FindNode(Double_t x, Double_t y, Double_t z);
   Double_t              *FindNormal(Bool_t forward=kTRUE);
   Double_t              *FindNormalFast();
   TGeoNode              *InitTrack(Double_t *point, Double_t *dir);
   TGeoNode              *InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz);
   Double_t               Safety(Bool_t inside=kFALSE);
   TGeoNode              *SearchNode(Bool_t downwards=kFALSE, const TGeoNode *skipnode=0);
   TGeoNode              *Step(Bool_t is_geom=kTRUE, Bool_t cross=kTRUE);
   void                   DisableInactiveVolumes() {fActivity=kTRUE;}
   void                   EnableInactiveVolumes()  {fActivity=kFALSE;}
   void                   SetCurrentTrack(Int_t i) {fCurrentTrack = (TVirtualGeoTrack*)fTracks->At(i);}
   void                   SetCurrentTrack(TVirtualGeoTrack *track) {fCurrentTrack=track;}
   Int_t                  GetNtracks() const {return fNtracks;}
   TVirtualGeoTrack      *GetCurrentTrack() {return fCurrentTrack;}
   TVirtualGeoTrack      *GetLastTrack() {return (TVirtualGeoTrack *)fTracks->At(fNtracks-1);}
   const Double_t        *GetLastPoint() const {return fLastPoint;}
   TVirtualGeoTrack      *GetTrack(Int_t index)         {return (index<fNtracks)?(TVirtualGeoTrack*)fTracks->At(index):0;}
   Int_t                  GetTrackIndex(Int_t id) const;
   TVirtualGeoTrack      *GetTrackOfId(Int_t id) const;
   TVirtualGeoTrack      *GetParentTrackOfId(Int_t id) const;
   Int_t                  GetVirtualLevel();
   Bool_t                 GotoSafeLevel();
   Int_t                  GetSafeLevel() const;
   Double_t               GetSafeDistance() const      {return fSafety;}
   Double_t               GetStep() const              {return fStep;}
   void                   InspectState() const;
   Bool_t                 IsAnimatingTracks() const    {return fIsGeomReading;}
   Bool_t                 IsCheckingOverlaps() const   {return fSearchOverlaps;}
   Bool_t                 IsMatrixTransform() const    {return fMatrixTransform;}
   Bool_t                 IsMatrixReflection() const   {return fMatrixReflection;}
   Bool_t                 IsSameLocation(Double_t x, Double_t y, Double_t z, Bool_t change=kFALSE);
   Bool_t                 IsSameLocation() const {return fIsSameLocation;}
   Bool_t                 IsSamePoint(Double_t x, Double_t y, Double_t z) const;
   Bool_t                 IsStartSafe() const {return fStartSafe;}
   void                   SetCheckingOverlaps(Bool_t flag=kTRUE) {fSearchOverlaps = flag;}
   void                   SetStartSafe(Bool_t flag=kTRUE)   {fStartSafe=flag;}
   void                   SetMatrixTransform(Bool_t on=kTRUE) {fMatrixTransform = on;}
   void                   SetMatrixReflection(Bool_t flag=kTRUE) {fMatrixReflection = flag;}
   void                   SetStep(Double_t step) {fStep=step;}
   Bool_t                 IsCurrentOverlapping() const {return fCurrentOverlapping;}
   Bool_t                 IsEntering() const           {return fIsEntering;}
   Bool_t                 IsExiting() const            {return fIsExiting;}
   Bool_t                 IsStepEntering() const       {return fIsStepEntering;}
   Bool_t                 IsStepExiting() const        {return fIsStepExiting;}
   Bool_t                 IsOutside() const            {return fIsOutside;}
   Bool_t                 IsOnBoundary() const         {return fIsOnBoundary;}
   Bool_t                 IsNullStep() const           {return fIsNullStep;}
   Bool_t                 IsActivityEnabled() const    {return fActivity;}
   void                   SetOutside(Bool_t flag=kTRUE) {fIsOutside = flag;}
   void                   UpdateCurrentPosition(Double_t *nextpoint);


   //--- cleaning
   void                   CleanGarbage();
   void                   ClearShape(const TGeoShape *shape);
   void                   ClearTracks() {fTracks->Delete(); fNtracks=0;}
   void                   RemoveMaterial(Int_t index);
   void                   ResetUserData();


   //--- utilities
   Int_t                  CountNodes(const TGeoVolume *vol=0, Int_t nlevels=10000, Int_t option=0);
   virtual void           ExecuteEvent(Int_t event, Int_t px, Int_t py);
   static Int_t           Parse(const char* expr, TString &expr1, TString &expr2, TString &expr3);
   Int_t                  ReplaceVolume(TGeoVolume *vorig, TGeoVolume *vnew);
   Int_t                  TransformVolumeToAssembly(const char *vname);
   UChar_t               *GetBits() {return fBits;}
   virtual Int_t          GetByteCount(Option_t *option=0);
   Int_t                 *GetIntBuffer(Int_t length);
   Double_t              *GetDblBuffer(Int_t length);
   void                   SetAllIndex();

   //--- I/O
   virtual Int_t          Export(const char *filename, const char *name="", Option_t *option="v");
   static  void           LockGeometry();
   static  void           UnlockGeometry();
   static TGeoManager    *Import(const char *filename, const char *name="", Option_t *option="");
   static Bool_t          IsLocked();
   Bool_t                 IsStreamingVoxels() const {return fStreamVoxels;}

   //--- list getters
   TObjArray             *GetListOfNodes()              {return fNodes;}
   TObjArray             *GetListOfPhysicalNodes()      {return fPhysicalNodes;}
   TObjArray             *GetListOfOverlaps()           {return fOverlaps;}
   TObjArray             *GetListOfMatrices() const     {return fMatrices;}
   TList                 *GetListOfMaterials() const    {return fMaterials;}
   TList                 *GetListOfMedia() const        {return fMedia;}
   TObjArray             *GetListOfVolumes() const      {return fVolumes;}
   TObjArray             *GetListOfGVolumes() const     {return fGVolumes;}
   TObjArray             *GetListOfShapes() const       {return fShapes;}
   TObjArray             *GetListOfUVolumes() const     {return fUniqueVolumes;}
   TObjArray             *GetListOfTracks() const       {return fTracks;}
   TGeoElementTable      *GetElementTable() const       {return fElementTable;}

   //--- modeler state getters/setters
   TGeoNode              *GetNode(Int_t level) const  {return (TGeoNode*)fNodes->UncheckedAt(level);}
   Int_t                  GetNodeId() const {return fCache->GetNodeId();}
   TGeoNode              *GetNextNode() const         {return fNextNode;}
   TGeoNode              *GetMother(Int_t up=1) const {return fCache->GetMother(up);}
   TGeoHMatrix           *GetMotherMatrix(Int_t up=1) const {return fCache->GetMotherMatrix(up);}
   TGeoHMatrix           *GetHMatrix();
   TGeoHMatrix           *GetCurrentMatrix() const    {return fCache->GetCurrentMatrix();}
   TGeoHMatrix           *GetGLMatrix() const         {return fGLMatrix;}
   TGeoNode              *GetCurrentNode() const      {return fCurrentNode;}
   Int_t                  GetCurrentNodeId() const;
   Double_t              *GetCurrentPoint() const     {return fPoint;}
   Double_t              *GetCurrentDirection() const {return fDirection;}
   TGeoVolume            *GetCurrentVolume() const {return fCurrentNode->GetVolume();}
   Double_t              *GetCldirChecked() const  {return fCldirChecked;}
   Double_t              *GetCldir() const         {return fCldir;}
//   Double_t               GetNormalChecked() const {return fNormalChecked;}
   const Double_t        *GetNormal() const        {return fNormal;}
   Int_t                  GetLevel() const         {return fLevel;}
   const char            *GetPath() const;
   Int_t                  GetStackLevel() const    {return fCache->GetStackLevel();}
   TGeoVolume            *GetMasterVolume() const  {return fMasterVolume;}
   TGeoVolume            *GetTopVolume() const     {return fTopVolume;}
   TGeoNode              *GetTopNode() const       {return fTopNode;}
   TGeoPhysicalNode      *GetPhysicalNode(Int_t i) const {return (TGeoPhysicalNode*)fPhysicalNodes->UncheckedAt(i);}
   void                   SetCurrentPoint(Double_t *point) {memcpy(fPoint,point,3*sizeof(Double_t));}
   void                   SetCurrentPoint(Double_t x, Double_t y, Double_t z) {
                                    fPoint[0]=x; fPoint[1]=y; fPoint[2]=z;}
   void                   SetLastPoint(Double_t x, Double_t y, Double_t z) {
                                    fLastPoint[0]=x; fLastPoint[1]=y; fLastPoint[2]=z;}
   void                   SetCurrentDirection(Double_t *dir) {memcpy(fDirection,dir,3*sizeof(Double_t));}
   void                   SetCurrentDirection(Double_t nx, Double_t ny, Double_t nz) {
                                    fDirection[0]=nx; fDirection[1]=ny; fDirection[2]=nz;}
//   void                   SetNormalChecked(Double_t norm) {fNormalChecked=norm;}
   void                   SetCldirChecked(Double_t *dir) {memcpy(fCldirChecked, dir, 3*sizeof(Double_t));}

   //--- point/vector reference frame conversion
   void                   LocalToMaster(const Double_t *local, Double_t *master) const {fCache->LocalToMaster(local, master);}
   void                   LocalToMasterVect(const Double_t *local, Double_t *master) const {fCache->LocalToMasterVect(local, master);}
   void                   LocalToMasterBomb(const Double_t *local, Double_t *master) const {fCache->LocalToMasterBomb(local, master);}
   void                   MasterToLocal(const Double_t *master, Double_t *local) const {fCache->MasterToLocal(master, local);}
   void                   MasterToLocalVect(const Double_t *master, Double_t *local) const {fCache->MasterToLocalVect(master, local);}
   void                   MasterToLocalBomb(const Double_t *master, Double_t *local) const {fCache->MasterToLocalBomb(master, local);}
   void                   MasterToTop(const Double_t *master, Double_t *top) const;
   void                   TopToMaster(const Double_t *top, Double_t *master) const;

   //--- general use getters/setters
   TGeoMaterial          *FindDuplicateMaterial(const TGeoMaterial *mat) const;
   TGeoVolume            *FindVolumeFast(const char*name, Bool_t multi=kFALSE);
   TGeoMaterial          *GetMaterial(const char *matname) const;
   TGeoMaterial          *GetMaterial(Int_t id) const;
   TGeoMedium            *GetMedium(const char *medium) const;
   TGeoMedium            *GetMedium(Int_t numed) const;
   Int_t                  GetMaterialIndex(const char *matname) const;
//   TGeoShape             *GetShape(const char *name) const;
   TGeoVolume            *GetVolume(const char*name) const;
   TGeoVolume            *GetVolume(Int_t uid) const {return (TGeoVolume*)fUniqueVolumes->At(uid);}
   Int_t                  GetUID(const char *volname) const;
   Int_t                  GetNNodes() {if (!fNNodes) CountNodes(); return fNNodes;}
   TGeoNodeCache         *GetCache() const         {return fCache;}
   void                   SetCache(const TGeoNodeCache *cache) {fCache = (TGeoNodeCache*)cache;}
   void                   SetAnimateTracks(Bool_t flag=kTRUE) {fIsGeomReading=flag;}
   virtual ULong_t        SizeOf(const TGeoNode *node, Option_t *option); // size of the geometry in memory
   void                   SelectTrackingMedia();

   //--- stack manipulation
   Int_t                  PushPath(Int_t startlevel=0) {return fCache->PushState(fCurrentOverlapping, startlevel, fNmany);}
   Bool_t                 PopPath() {fCurrentOverlapping=fCache->PopState(fNmany); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel();return fCurrentOverlapping;}
   Bool_t                 PopPath(Int_t index) {fCurrentOverlapping=fCache->PopState(fNmany,index); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel();return fCurrentOverlapping;}
   Int_t                  PushPoint(Int_t startlevel=0) {return fCache->PushState(fCurrentOverlapping, startlevel,fNmany,fPoint);}
   Bool_t                 PopPoint() {fCurrentOverlapping=fCache->PopState(fNmany,fPoint); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel(); return fCurrentOverlapping;}
   Bool_t                 PopPoint(Int_t index) {fCurrentOverlapping=fCache->PopState(fNmany,index, fPoint); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel(); return fCurrentOverlapping;}
   void                   PopDummy(Int_t ipop=9999) {fCache->PopDummy(ipop);}

   ClassDef(TGeoManager, 10)          // geometry manager
};

R__EXTERN TGeoManager *gGeoManager;

#endif

