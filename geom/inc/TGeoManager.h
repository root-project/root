// @(#)root/geom:$Name:  $:$Id: TGeoManager.h,v 1.2 2002/07/10 19:24:16 brun Exp $
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

#ifndef ROOT_TGeoNode
#include "TGeoNode.h"
#endif

#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

#ifndef ROOT_TGeoCache
#include "TGeoCache.h"
#endif

#ifndef ROOT_TVirtualGeoPainter
#include "TVirtualGeoPainter.h"
#endif



// forward declarations
class TGeoMatrix;
class TGeoHMatrix;
class TGeoMaterial;
class TGeoNodeCache;
class TGeoShape;
class TVirtualGeoPainter;

/*************************************************************************
 * TGeoManager - class description 
 *
 *************************************************************************/


class TGeoManager : public TNamed
{
private :
   Double_t              fStep;             // step to be done from current point and direction
   Double_t              fSafety;           // safety radius from current point
   Int_t                 fLevel;            // current geometry level;
   Int_t                 fNNodes;           // total number of physical nodes
   TString               fPath;             // path to current node
   Double_t             *fNormal;           // normal to shape at current intersection point
   Double_t             *fNormalChecked;    // normal to current checked shape at intersection
   Double_t             *fCldir;            // unit vector to current closest shape
   Double_t             *fCldirChecked;     // unit vector to current checked shape
   Double_t             *fPoint;            //[3] current point
   Double_t             *fDirection;        //[3] current direction
   Bool_t                fSearchOverlaps;   // flag set when an overlapping cluster is searched
   Bool_t                fCurrentOverlapping; // flags the type of the current node
   Bool_t                fLoopVolumes;      // flag volume lists loop
   Bool_t                fStartSafe;        // flag a safe start for point classification
   Bool_t                fIsEntering;       // flag if current step just got into a new node
   Bool_t                fIsExiting;        // flag that current track is about to leave current node
   Bool_t                fIsStepEntering;   // flag that next geometric step will enter new volume
   Bool_t                fIsStepExiting;    // flaag that next geometric step will exit current volume
   Bool_t                fIsOutside;        // flag that current point is outside geometry
   TGeoNodeCache        *fCache;            // cache for physical nodes
   TVirtualGeoPainter   *fPainter;          // current painter
   TGeoVolume           *fCurrentVolume;    // current volume
   TGeoVolume           *fTopVolume;        // top level volume in geometry
   TGeoNode             *fCurrentNode;      // current node
   TGeoNode             *fTopNode;          // top physical node
   TGeoNode             *fLastNode;         // last searched node
   TGeoVolume           *fMasterVolume;     // master volume
   TGeoHMatrix          *fCurrentMatrix;    // current global matrix
   TList                *fMatrices;         // list of local transformations
   TList                *fShapes;           // list of shapes
   TList                *fVolumes;          // list of volumes
   TList                *fGShapes;           // list of runtime shapes
   TList                *fGVolumes;          // list of runtime volumes
   TList                *fMaterials;        // list of materials
   TObjArray            *fGlobalMatrices;   // global transformations for current branch
   TObjArray            *fNodes;            // current branch of nodes
   UChar_t              *fBits;             // bits used for voxelization

//--- private methods
   void                   BuildCache();
   TGeoNode              *FindInCluster(Int_t *cluster, Int_t nc);
   Int_t                  GetTouchedCluster(Int_t start, Double_t *point, Int_t *check_list,
                                            Int_t ncheck, Int_t *result);
   Bool_t                 IsLoopingVolumes() const     {return fLoopVolumes;}
   void                   SetLoopVolumes(Bool_t flag=kTRUE) {fLoopVolumes=flag;}
   void                   Voxelize(Option_t *option = 0);

public:
   // constructors
   TGeoManager();
   TGeoManager(const char *name, const char *title);
   // destructor
   virtual ~TGeoManager();
   //--- adding geometrical objects - used by constructors
   Int_t                  AddMaterial(TGeoMaterial *material);
   Int_t                  AddTransformation(TGeoMatrix *matrix);
   Int_t                  AddShape(TGeoShape *shape);
   Int_t                  AddVolume(TGeoVolume *volume);
   //--- browsing and tree navigation
   void                   Browse(TBrowser *b);
   virtual Bool_t         cd(const char *path=""); // *MENU*
   void                   CdDown(Int_t index);
   void                   CdUp();
   void                   CdTop();
   Bool_t                 IsFolder() const { return kTRUE; }
   //--- visualization settings
   void                   BombTranslation(const Double_t *tr, Double_t *bombtr);
   void                   UnbombTranslation(const Double_t *tr, Double_t *bombtr);
   void                   ClearAttributes(); // *MENU*
   void                   ClearPad();
   void                   DefaultAngles();   // *MENU*
   void                   DefaultColors();   // *MENU*
   Int_t                  GetNsegments() const;
   TVirtualGeoPainter    *GetGeomPainter();
   Int_t                  GetBombMode() const  {return (fPainter)?(fPainter->GetBombMode()):0;}
   void                   GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const;
   Int_t                  GetVisLevel() const;
   Int_t                  GetVisOption() const;
   void                   ModifiedPad() const;
   void                   SetExplodedView(UInt_t iopt=0); // *MENU*
   void                   SetNsegments(Int_t nseg);
   void                   SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3,                                         Double_t bombr=1.3); // *MENU* 
   void                   SetVisLevel(Int_t level=3);   // *MENU*
   void                   SetVisOption(Int_t option=0); // *MENU*
   void                   SaveAttributes(const char *filename="tgeoatt.C"); // *MENU*
   void                   RestoreMasterVolume(); // *MENU*

   //--- geometry checking
   void                   CheckGeometry(Option_t *option="");
   void                   CheckPoint(Double_t x=0,Double_t y=0, Double_t z=0, Option_t *option=""); // *MENU*
   void                   DrawCurrentPoint(Int_t color=2); // *MENU*
   void                   DrawPath(const char *path) {GetGeomPainter()->DrawPath(path);} // *MENU*
   void                   RandomPoints(TGeoVolume *vol, Int_t npoints=10000, Option_t *option="");
   void                   RandomRays(Int_t nrays=1000);
   TGeoNode              *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil=1E-5,
                                       const char *g3path="");
   void                   Test(Int_t npoints=1000000, Option_t *option=""); // *MENU*
   void                   TestOverlaps(const char* path=""); // *MENU*

   //--- geometry building
   void                   BuildDefaultMaterials();
   void                   CloseGeometry();
   Bool_t                 IsClosed() const {return ((fCache==0)?kFALSE:kTRUE);}
   TGeoVolume            *MakeArb8(const char *name, const char *material,
                                     Double_t dz, Double_t *vertices=0);
   TGeoVolume            *MakeBox(const char *name, const char *material,
                                     Double_t dx, Double_t dy, Double_t dz);
   TGeoVolume            *MakePara(const char *name, const char *material,
                                     Double_t dx, Double_t dy, Double_t dz,
                                     Double_t alpha, Double_t theta, Double_t phi);
   TGeoVolume            *MakeSphere(const char *name, const char *material,
                                     Double_t rmin, Double_t rmax,
                                     Double_t themin=0, Double_t themax=180,
                                     Double_t phimin=0, Double_t phimax=360);
   TGeoVolume            *MakeTube(const char *name, const char *material,
                                      Double_t rmin, Double_t rmax, Double_t dz);
   TGeoVolume            *MakeTubs(const char *name, const char *material,
                                      Double_t rmin, Double_t rmax, Double_t dz,
                                      Double_t phi1, Double_t phi2);
   TGeoVolume            *MakeEltu(const char *name, const char *material,
                                      Double_t a, Double_t b, Double_t dz);
   TGeoVolume            *MakeCtub(const char *name, const char *material,
                                      Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                                      Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoVolume            *MakeCone(const char *name, const char *material,
                                      Double_t dz, Double_t rmin1, Double_t rmax1,
                                      Double_t rmin2, Double_t rmax2);
   TGeoVolume            *MakeCons(const char *name, const char *material,
                                      Double_t dz, Double_t rmin1, Double_t rmax1,
                                      Double_t rmin2, Double_t rmax2,
                                      Double_t phi1, Double_t phi2);
   TGeoVolume            *MakePcon(const char *name, const char *material,
                                      Double_t phi, Double_t dphi, Int_t nz);
   TGeoVolume            *MakePgon(const char *name, const char *material,
                                      Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoVolume            *MakeTrd1(const char *name, const char *material,
                                      Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoVolume            *MakeTrd2(const char *name, const char *material,
                                      Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2,
                                      Double_t dz);
   TGeoVolume            *MakeTrap(const char *name, const char *material,
                                   Double_t dz, Double_t theta, Double_t phi, Double_t h1,
                                   Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
                                   Double_t tl2, Double_t alpha2);
   TGeoVolume            *MakeGtra(const char *name, const char *material,
                                   Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
                                   Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
                                   Double_t tl2, Double_t alpha2);
   TGeoVolumeMulti       *MakeVolumeMulti(const char *name, const char *material);
   void                   SetTopVolume(TGeoVolume *vol);
   
   //--- geometry queries
   void                   AddCheckedNode(TGeoNode *node, Int_t level) {fNodes->AddAt(node,level);}
   TGeoNode              *FindNextBoundary(const char *path="");
   TGeoNode              *FindNode(Bool_t safe_start=kTRUE) {fSearchOverlaps=fIsOutside=kFALSE; fStartSafe=safe_start; return SearchNode();}
   void                   InitTrack(Double_t *point, Double_t *dir);
   void                   InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz);
   TGeoNode              *SearchNode(Bool_t downwards=kFALSE, TGeoNode *skipnode=0);
   TGeoNode              *Step(Bool_t is_geom=kTRUE, Bool_t cross=kTRUE);
   Int_t                  GetVirtualLevel();
   Bool_t                 GotoSafeLevel();
   Double_t               GetSafeDistance() const      {return fSafety;}
   Double_t               GetStep() const              {return fStep;}
   Bool_t                 IsStartSafe() const {return fStartSafe;}
   void                   SetStartSafe(Bool_t flag=kTRUE)   {fStartSafe=flag;}
   void                   SetStep(Double_t step) {fStep=step;}
   Bool_t                 IsCurrentOverlapping() const {return fCurrentOverlapping;}
   Bool_t                 IsEntering() const           {return fIsEntering;}
   Bool_t                 IsExiting() const            {return fIsExiting;}
   Bool_t                 IsStepEntering() const       {return fIsStepEntering;}
   Bool_t                 IsStepExiting() const        {return fIsStepExiting;}
   Bool_t                 IsOutside() const            {return fIsOutside;} 
   void                   UpdateCurrentPosition(Double_t *nextpoint);
   

   //--- cleaning
   void                   CleanGarbage();
   void                   ClearShape(TGeoShape *shape);
   void                   RemoveMaterial(Int_t index);


   //--- utilities
   Int_t                  CountNodes(TGeoVolume *vol=0, Int_t nlevels=1000);
   void                   ComputeGlobalMatrices(Option_t *option = 0);
   UChar_t               *GetBits() {return fBits;}
   virtual Int_t          GetByteCount(Option_t *option=0);

   //--- list getters
   TList                 *GetListOfMatrices() const     {return fMatrices;}
   TList                 *GetListOfMaterials() const    {return fMaterials;}
   TList                 *GetListOfVolumes() const      {return fVolumes;}
   TList                 *GetListOfGVolumes() const     {return fGVolumes;} 
   TList                 *GetListOfShapes() const       {return fShapes;}

   //--- modeler state getters/setters
   TGeoNode              *GetNode(Int_t level) const  {return (TGeoNode*)fNodes->At(level);}
   TGeoNode              *GetMother(Int_t up=1) const {return fCache->GetMother(up);}
   TGeoNode              *GetCurrentNode() const      {return fCurrentNode;}
   Double_t              *GetCurrentPoint() const     {return fPoint;}
   TGeoVolume            *GetCurrentVolume() const {return fCurrentNode->GetVolume();}
   TGeoHMatrix           *GetCurrentMatrix() const {return (TGeoHMatrix*)fGlobalMatrices->At(fLevel);}
   Double_t              *GetCldirChecked() const  {return fCldirChecked;}
   Double_t              *GetCldir() const         {return fCldir;}
   Double_t              *GetNormalChecked() const {return fNormalChecked;}
   Double_t              *GetNormal() const        {return fNormal;}
   Int_t                  GetLevel() const         {return fLevel;}
   const char            *GetPath() const;
   Int_t                  GetStackLevel() const    {return fCache->GetStackLevel();}
   TGeoVolume            *GetMasterVolume() const  {return fMasterVolume;}
   TGeoVolume            *GetTopVolume() const     {return fTopVolume;}
   TGeoNode              *GetTopNode() const       {return fTopNode;}
   void                   SetCurrentPoint(Double_t *point) {memcpy(fPoint,point,3*sizeof(Double_t));}
   void                   SetCurrentPoint(Double_t x, Double_t y, Double_t z) { 
                                    fPoint[0]=x; fPoint[1]=y; fPoint[2]=z;}
   void                   SetCurrentDirection(Double_t *dir) {memcpy(fDirection,dir,3*sizeof(Double_t));}
   void                   SetCurrentDirection(Double_t nx, Double_t ny, Double_t nz) { 
                                    fDirection[0]=nx; fDirection[1]=ny; fDirection[2]=nz;}
   void                   SetNormalChecked(Double_t *norm) {memcpy(fNormalChecked, norm, 3*sizeof(Double_t));}
   void                   SetCldirChecked(Double_t *dir) {memcpy(fCldirChecked, dir, 3*sizeof(Double_t));}
   
   //--- point/vector reference frame conversion   
   void                   LocalToMaster(Double_t *local, Double_t *master) const
                            {fCache->LocalToMaster(local, master);}
   void                   LocalToMasterVect(Double_t *local, Double_t *master) const
                            {fCache->LocalToMasterVect(local, master);}
   void                   LocalToMasterBomb(Double_t *local, Double_t *master) const
                            {fCache->LocalToMasterBomb(local, master);}
   void                   MasterToLocal(Double_t *master, Double_t *local) const
                            {fCache->MasterToLocal(master, local);}
   void                   MasterToLocalVect(Double_t *master, Double_t *local) const
                            {fCache->MasterToLocalVect(master, local);}
   void                   MasterToLocalBomb(Double_t *master, Double_t *local) const
                            {fCache->MasterToLocalBomb(master, local);}

   //--- general use getters/setters
   TGeoMaterial          *GetMaterial(const char *matname) const;
   TGeoMaterial          *GetMaterial(Int_t id) const;
   Int_t                  GetMaterialIndex(const char *matname) const;
//   TGeoShape             *GetShape(const char *name) const;
   TGeoVolume            *GetVolume(const char*name) const;
   Int_t                  GetNNodes() {if (!fNNodes) CountNodes(); return fNNodes;}
   TGeoNodeCache         *GetCache() const         {return fCache;}
   void                   SetCache(TGeoNodeCache *cache) {fCache = cache;}   
   virtual ULong_t        SizeOf(TGeoNode *node, Option_t *option); // size of the geometry in memory
   void                   SelectTrackingMedia();

   //--- stack manipulation
   Int_t                  PushPath() {return fCache->PushState(fCurrentOverlapping);}
   Bool_t                 PopPath() {Bool_t ret=fCache->PopState(); fCurrentNode=fCache->GetNode();
                                     fLevel=fCache->GetLevel();return ret;}
   Bool_t                 PopPath(Int_t index) {Bool_t ret=fCache->PopState(index);
                                     fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel();return ret;}
   Int_t                  PushPoint() {return fCache->PushState(fCurrentOverlapping, fPoint);}
   Bool_t                 PopPoint() {fCurrentNode=fCache->GetNode();
                                     fLevel=fCache->GetLevel();return fCache->PopState(fPoint);}
   Bool_t                 PopPoint(Int_t index) {fCurrentNode=fCache->GetNode();
                                     fLevel=fCache->GetLevel(); return fCache->PopState(index, fPoint);}
   void                   PopDummy(Int_t ipop=9999) {fCache->PopDummy(ipop);}

  ClassDef(TGeoManager, 1)          // geometry manager
};

R__EXTERN TGeoManager *gGeoManager;

#endif

