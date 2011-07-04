// @(#)root/geom:$Id$
// Author: Mihaela Gheata   30/05/07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoNavigator
#define ROOT_TGeoNavigator

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TGeoNodeCache
#include "TGeoCache.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoNavigator - Class containing the implementation of all navigation  //
//   methods.
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoManager;
class TGeoNode;
class TGeoVolume;
class TGeoMatrix;
class TGeoHMatrix;


class TGeoNavigator : public TObject
{

protected:
   TGeoNavigator(const TGeoNavigator&); 
   TGeoNavigator& operator=(const TGeoNavigator&); 
   TGeoNode             *FindInCluster(Int_t *cluster, Int_t nc);
   Int_t                 GetTouchedCluster(Int_t start, Double_t *point, Int_t *check_list,
                                           Int_t ncheck, Int_t *result);
   TGeoNode             *CrossDivisionCell();
   void                  SafetyOverlaps();

private :
   Double_t              fStep;             //! step to be done from current point and direction
   Double_t              fSafety;           //! safety radius from current point
   Double_t              fLastSafety;       //! last computed safety radius
   Double_t              fNormal[3];        //! cosine of incident angle on current checked surface
   Double_t              fCldir[3];         //! unit vector to current closest shape
   Double_t              fCldirChecked[3];  //! unit vector to current checked shape
   Double_t              fPoint[3];         //! current point
   Double_t              fDirection[3];     //! current direction
   Double_t              fLastPoint[3];     //! last point for which safety was computed
   Int_t                 fLevel;            //! current geometry level;
   Int_t                 fNmany;            //! number of overlapping nodes on current branch
   Int_t                 fNextDaughterIndex; //! next daughter index after FindNextBoundary
   Int_t                 fOverlapSize;      //! current size of fOverlapClusters
   Int_t                 fOverlapMark;      //! current recursive position in fOverlapClusters
   Int_t                *fOverlapClusters;  //! internal array for overlaps
   Bool_t                fSearchOverlaps;   //! flag set when an overlapping cluster is searched
   Bool_t                fCurrentOverlapping; //! flags the type of the current node
   Bool_t                fStartSafe;        //! flag a safe start for point classification
   Bool_t                fIsEntering;       //! flag if current step just got into a new node
   Bool_t                fIsExiting;        //! flag that current track is about to leave current node
   Bool_t                fIsStepEntering;   //! flag that next geometric step will enter new volume
   Bool_t                fIsStepExiting;    //! flaag that next geometric step will exit current volume
   Bool_t                fIsOutside;        //! flag that current point is outside geometry
   Bool_t                fIsOnBoundary;     //! flag that current point is on some boundary
   Bool_t                fIsSameLocation;   //! flag that a new point is in the same node as previous
   Bool_t                fIsNullStep;       //! flag that last geometric step was null
   TGeoManager          *fGeometry;         //! current geometry
   TGeoNodeCache        *fCache;            //! cache of states
   TGeoVolume           *fCurrentVolume;    //! current volume
   TGeoNode             *fCurrentNode;      //! current node    
   TGeoNode             *fTopNode;          //! top physical node
   TGeoNode             *fLastNode;         //! last searched node
   TGeoNode             *fNextNode;         //! next node that will be crossed
   TGeoNode             *fForcedNode;       //! current point is supposed to be inside this node
   TGeoCacheState       *fBackupState;      //! backup state
   TGeoHMatrix          *fCurrentMatrix;    //! current stored global matrix
   TGeoHMatrix          *fGlobalMatrix;     //! current pointer to cached global matrix
   TGeoHMatrix          *fDivMatrix;        //! current local matrix of the selected division cell
   TString               fPath;             //! path to current node
    
public :
   TGeoNavigator();
   TGeoNavigator(TGeoManager* geom);
   virtual ~TGeoNavigator();

   void                   BuildCache(Bool_t dummy=kFALSE, Bool_t nodeid=kFALSE);
   Bool_t                 cd(const char *path="");
   Bool_t                 CheckPath(const char *path) const;
   void                   CdNode(Int_t nodeid);
   void                   CdDown(Int_t index);
   void                   CdUp();
   void                   CdTop();
   void                   CdNext();
   void                   GetBranchNames(Int_t *names) const;
   void                   GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const;
   void                   GetBranchOnlys(Int_t *isonly) const;
   Int_t                  GetNmany() const {return fNmany;}
   //--- geometry queries
   TGeoNode              *CrossBoundaryAndLocate(Bool_t downwards, TGeoNode *skipnode);
   TGeoNode              *FindNextBoundary(Double_t stepmax=TGeoShape::Big(),const char *path="", Bool_t frombdr=kFALSE);
   TGeoNode              *FindNextDaughterBoundary(Double_t *point, Double_t *dir, Int_t &idaughter, Bool_t compmatrix=kFALSE);
   TGeoNode              *FindNextBoundaryAndStep(Double_t stepmax=TGeoShape::Big(), Bool_t compsafe=kFALSE);
   TGeoNode              *FindNode(Bool_t safe_start=kTRUE);
   TGeoNode              *FindNode(Double_t x, Double_t y, Double_t z);
   Double_t              *FindNormal(Bool_t forward=kTRUE);
   Double_t              *FindNormalFast();
   TGeoNode              *InitTrack(const Double_t *point, const Double_t *dir);
   TGeoNode              *InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz);
   void                   ResetState();
   void                   ResetAll();
   Double_t               Safety(Bool_t inside=kFALSE);
   TGeoNode              *SearchNode(Bool_t downwards=kFALSE, const TGeoNode *skipnode=0);
   TGeoNode              *Step(Bool_t is_geom=kTRUE, Bool_t cross=kTRUE);
   const Double_t        *GetLastPoint() const {return fLastPoint;}
   Int_t                  GetVirtualLevel();
   Bool_t                 GotoSafeLevel();
   Int_t                  GetSafeLevel() const;
   Double_t               GetSafeDistance() const      {return fSafety;}
   Double_t               GetLastSafety() const        {return fLastSafety;}
   Double_t               GetStep() const              {return fStep;}
   void                   InspectState() const;
   Bool_t                 IsSafeStep(Double_t proposed, Double_t &newsafety) const;
   Bool_t                 IsSameLocation(Double_t x, Double_t y, Double_t z, Bool_t change=kFALSE);
   Bool_t                 IsSameLocation() const {return fIsSameLocation;}
   Bool_t                 IsSamePoint(Double_t x, Double_t y, Double_t z) const;
   Bool_t                 IsStartSafe() const {return fStartSafe;}
   void                   SetStartSafe(Bool_t flag=kTRUE)   {fStartSafe=flag;}
   void                   SetStep(Double_t step) {fStep=step;}
   Bool_t                 IsCheckingOverlaps() const   {return fSearchOverlaps;}
   Bool_t                 IsCurrentOverlapping() const {return fCurrentOverlapping;}
   Bool_t                 IsEntering() const           {return fIsEntering;}
   Bool_t                 IsExiting() const            {return fIsExiting;}
   Bool_t                 IsStepEntering() const       {return fIsStepEntering;}
   Bool_t                 IsStepExiting() const        {return fIsStepExiting;}
   Bool_t                 IsOutside() const            {return fIsOutside;}
   Bool_t                 IsOnBoundary() const         {return fIsOnBoundary;}
   Bool_t                 IsNullStep() const           {return fIsNullStep;}
   void                   SetCheckingOverlaps(Bool_t flag=kTRUE) {fSearchOverlaps = flag;}
   void                   SetOutside(Bool_t flag=kTRUE) {fIsOutside = flag;}
   //--- modeler state getters/setters
   void                   DoBackupState();
   void                   DoRestoreState();
   Int_t                  GetNodeId() const           {return fCache->GetNodeId();}
   Int_t                  GetNextDaughterIndex() const {return fNextDaughterIndex;}
   TGeoNode              *GetNextNode() const         {return fNextNode;}
   TGeoNode              *GetMother(Int_t up=1) const {return fCache->GetMother(up);}
   TGeoHMatrix           *GetMotherMatrix(Int_t up=1) const {return fCache->GetMotherMatrix(up);}
   TGeoHMatrix           *GetHMatrix();
   TGeoHMatrix           *GetCurrentMatrix() const    {return fCache->GetCurrentMatrix();}
   TGeoNode              *GetCurrentNode() const      {return fCurrentNode;}
   Int_t                  GetCurrentNodeId() const    {return fCache->GetCurrentNodeId();}
   const Double_t        *GetCurrentPoint() const     {return fPoint;}
   const Double_t        *GetCurrentDirection() const {return fDirection;}
   TGeoVolume            *GetCurrentVolume() const {return fCurrentNode->GetVolume();}
   const Double_t        *GetCldirChecked() const  {return fCldirChecked;}
   const Double_t        *GetCldir() const         {return fCldir;}
   TGeoHMatrix           *GetDivMatrix() const     {return fDivMatrix;}
//   Double_t               GetNormalChecked() const {return fNormalChecked;}
   const Double_t        *GetNormal() const        {return fNormal;}
   Int_t                  GetLevel() const         {return fLevel;}
   const char            *GetPath() const;
   Int_t                  GetStackLevel() const    {return fCache->GetStackLevel();}
   void                   SetCurrentPoint(const Double_t *point) {memcpy(fPoint,point,3*sizeof(Double_t));}
   void                   SetCurrentPoint(Double_t x, Double_t y, Double_t z) {
                                    fPoint[0]=x; fPoint[1]=y; fPoint[2]=z;}
   void                   SetLastPoint(Double_t x, Double_t y, Double_t z) {
                                    fLastPoint[0]=x; fLastPoint[1]=y; fLastPoint[2]=z;}
   void                   SetCurrentDirection(const Double_t *dir) {memcpy(fDirection,dir,3*sizeof(Double_t));}
   void                   SetCurrentDirection(Double_t nx, Double_t ny, Double_t nz) {
                                    fDirection[0]=nx; fDirection[1]=ny; fDirection[2]=nz;}
//   void                   SetNormalChecked(Double_t norm) {fNormalChecked=norm;}
   void                   SetCldirChecked(Double_t *dir) {memcpy(fCldirChecked, dir, 3*sizeof(Double_t));}
   void                   SetLastSafetyForPoint(Double_t safe, const Double_t *point) {fLastSafety=safe; memcpy(fLastPoint,point,3*sizeof(Double_t));}
   
   //--- point/vector reference frame conversion
   void                   LocalToMaster(const Double_t *local, Double_t *master) const {fCache->LocalToMaster(local, master);}
   void                   LocalToMasterVect(const Double_t *local, Double_t *master) const {fCache->LocalToMasterVect(local, master);}
   void                   LocalToMasterBomb(const Double_t *local, Double_t *master) const {fCache->LocalToMasterBomb(local, master);}
   void                   MasterToLocal(const Double_t *master, Double_t *local) const {fCache->MasterToLocal(master, local);}
   void                   MasterToLocalVect(const Double_t *master, Double_t *local) const {fCache->MasterToLocalVect(master, local);}
   void                   MasterToLocalBomb(const Double_t *master, Double_t *local) const {fCache->MasterToLocalBomb(master, local);}
   void                   MasterToTop(const Double_t *master, Double_t *top) const;
   void                   TopToMaster(const Double_t *top, Double_t *master) const;
   TGeoNodeCache         *GetCache() const         {return fCache;}
//   void                   SetCache(const TGeoNodeCache *cache) {fCache = (TGeoNodeCache*)cache;}
   //--- stack manipulation
   Int_t                  PushPath(Int_t startlevel=0) {return fCache->PushState(fCurrentOverlapping, startlevel, fNmany);}
   Bool_t                 PopPath() {fCurrentOverlapping=fCache->PopState(fNmany); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel();fGlobalMatrix=fCache->GetCurrentMatrix();return fCurrentOverlapping;}
   Bool_t                 PopPath(Int_t index) {fCurrentOverlapping=fCache->PopState(fNmany,index); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel();fGlobalMatrix=fCache->GetCurrentMatrix();return fCurrentOverlapping;}
   Int_t                  PushPoint(Int_t startlevel=0) {return fCache->PushState(fCurrentOverlapping, startlevel,fNmany,fPoint);}
   Bool_t                 PopPoint() {fCurrentOverlapping=fCache->PopState(fNmany,fPoint); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel(); fGlobalMatrix=fCache->GetCurrentMatrix();return fCurrentOverlapping;}
   Bool_t                 PopPoint(Int_t index) {fCurrentOverlapping=fCache->PopState(fNmany,index, fPoint); fCurrentNode=fCache->GetNode(); fLevel=fCache->GetLevel(); fGlobalMatrix=fCache->GetCurrentMatrix();return fCurrentOverlapping;}
   void                   PopDummy(Int_t ipop=9999) {fCache->PopDummy(ipop);}
   
   ClassDef(TGeoNavigator, 0)          // geometry navigator class
};

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoNavigatorArray - Class representing an array of navigators working //
//   in a single thread.                                                  //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoNavigatorArray : public TObjArray
{
private:
   TGeoNavigator         *fCurrentNavigator; // Current navigator
   TGeoManager           *fGeoManager;       // Manager to which it applies
   
   TGeoNavigatorArray(const TGeoNavigatorArray&);
   TGeoNavigatorArray& operator=(const TGeoNavigatorArray&);

public:
   TGeoNavigatorArray(TGeoManager *mgr) : TObjArray(), fCurrentNavigator(0), fGeoManager(mgr) {SetOwner();}
   virtual ~TGeoNavigatorArray() {}
   
   TGeoNavigator         *AddNavigator();
   inline TGeoNavigator  *GetCurrentNavigator() const {return fCurrentNavigator;}   
   TGeoNavigator         *SetCurrentNavigator(Int_t inav) {return (fCurrentNavigator=(TGeoNavigator*)At(inav));}

   ClassDef(TGeoNavigatorArray, 0)       // An array of navigators
};   
#endif
   
