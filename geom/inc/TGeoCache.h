/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata -           date Wed 12 Dec 2001 09:45:08 AM CET

#ifndef ROOT_TGeoCache
#define ROOT_TGeoCache

#ifndef ROOT_TSystem
#include "TSystem.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

#ifndef ROOT_TGeoNode
#include "TGeoNode.h"
#endif

// forward declarations
class TBits;
class TGeoNodePos;
class TGeoMatrixCache;
class TGeoNodeArray;
class TGeoMatHandler;

/*************************************************************************
 * TGeoCacheState - class storing the state of the cache at a given moment
 *    
 *
 *************************************************************************/

class TGeoCacheState : public TObject
{
protected:
   Bool_t               fOverlapping; // overlap flag
   Int_t                fLevel;     // level in the current branch
   Double_t            *fPoint;     // last point in master frame
public:
   Int_t               *fBranch;    // last branch stored
   Int_t               *fMatrices;  // global matrices for last branch

public:
   TGeoCacheState();
   TGeoCacheState(Int_t capacity);
   virtual ~TGeoCacheState();
   
   virtual void         SetState(Int_t level, Bool_t ovlp, Double_t *point=0);
   virtual Bool_t       GetState(Int_t &level, Double_t *point);
               
  ClassDef(TGeoCacheState, 1)       // class storing the cache state
};

/*************************************************************************
 * TGeoCacheStateDummy - class storing the state of the cache at a given moment
 *    
 *
 *************************************************************************/

class TGeoCacheStateDummy : public TGeoCacheState
{
public:
   TGeoNode           **fNodeBranch;    // last node branch stored
   TGeoHMatrix        **fMatrixBranch;  // global matrices for last branch

public:
   TGeoCacheStateDummy();
   TGeoCacheStateDummy(Int_t capacity);
   virtual ~TGeoCacheStateDummy();
   
   virtual void         SetState(Int_t level, Bool_t ovlp, Double_t *point=0);
   virtual Bool_t       GetState(Int_t &level, Double_t *point);
               
  ClassDef(TGeoCacheStateDummy, 1)       // class storing the cache state
};

/*************************************************************************
 * TGeoNodeCache - cache of reusable physical nodes
 *    
 *
 *************************************************************************/

class TGeoNodeCache
{
public:
   static const Int_t    kGeoCacheMaxDaughters; // max ndaugters for TGeoNodeArray
   static const Int_t    kGeoCacheMaxSize;   // maximum initial cache size
   static const Int_t    kGeoCacheDefaultLevel; // default level down to store nodes
   static const Int_t    kGeoCacheMaxLevels; // maximum supported number of levels
   static const Int_t    kGeoCacheObjArrayInd; // maximum number of daughters stored as node arrays
   static const Int_t    kGeoCacheStackSize;   // maximum size of the stack
   static const Double_t kGeoCacheUsageRatio; // percentage of total usage count that triggers persistency
protected:
   Int_t                 fLevel;            // level in the current branch   
   TString               fPath;             // path for current branch
   TObjArray            *fStack;            // stack of cache states

private:   
   Int_t                 fSize;             // current size of the cache
   Int_t                 fNused;            // number of used nodes
   Int_t                 fDefaultLevel;     // level down to which nodes will be persistent
   TGeoNodeArray       **fCache;            //[128] cache of node arrays
   
   Int_t                 fTopNode;          // top level physical node
   Int_t                 fCount;            // total usage count
   Int_t                 fCountLimit;       // count limit for which nodes become persistent
   Int_t                 fCurrentNode;      // current physical node
   Int_t                 fCurrentCache;     // current cache number
   Int_t                 fCurrentIndex;     // index of current node in current cache
   Int_t                *fBranch;           // nodes from current branch
   Int_t                *fMatrices;         // matrix indices from current branch
   Int_t                 fStackLevel;       // level in the stack of states
   TGeoHMatrix          *fGlobalMatrix;     // current global matrix

   TGeoMatrixCache      *fMatrixPool;       // pool of compressed global matrices
public:
   TGeoNodeCache();
   TGeoNodeCache(Int_t size);
   virtual ~TGeoNodeCache();

   Int_t                AddNode(TGeoNode *node);
//   Int_t                AddDaughter(Int_t mother, TGeoNode *node, Int_t index);
   Int_t                CacheId(Int_t nindex)
                         {return (*((UChar_t*)&nindex+3)>kGeoCacheMaxDaughters)?
                                 kGeoCacheObjArrayInd:*((UChar_t*)&nindex+3);}
   void                 CdCache();
   virtual Bool_t       CdDown(Int_t index, Bool_t make=kTRUE);
   virtual void         CdTop() {fLevel=1; CdUp();}
   virtual void         CdUp();
   virtual void         CleanCache();
   virtual void         ClearDaughter(Int_t index);
   virtual void         ClearNode(Int_t nindex);
   virtual void         Compact();
   virtual void         DeleteCaches();
   virtual Bool_t       DumpNodes();
   virtual void        *GetBranch() {return fBranch;}
   virtual void        *GetMatrices() {return fMatrices;}
   virtual TGeoHMatrix *GetCurrentMatrix() {return fGlobalMatrix;}
   Int_t                GetCurrentNode() {return fCurrentNode;}
   virtual TGeoNode    *GetMother(Int_t up=1);
   virtual TGeoNode    *GetNode();
   Int_t                GetStackLevel()  {return fStackLevel;}
   Int_t                GetTopNode()     {return fTopNode;}
   Int_t                GetLevel()       {return fLevel;}
   virtual Int_t        GetFreeSpace()   {return (kGeoCacheMaxSize-fSize);}
   TGeoMatrixCache     *GetMatrixPool()  {return fMatrixPool;}
   virtual Int_t        GetNfree()       {return (fSize-fNused);}
   virtual Int_t        GetNused()       {return fNused;}
   virtual const char  *GetPath(); 
   Int_t                GetSize()        {return fSize;}
   virtual Int_t        GetUsageCount();
   virtual void         IncreasePool(Int_t size) {fSize+=size;}
   virtual void         IncrementUsageCount();
   Int_t                Index(Int_t nindex) {return (nindex & 0xFFFFFF);}
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         MasterToLocal(Double_t *master, Double_t *local);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master);
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local);
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);
   virtual void         PrintNode();
   virtual Int_t        PushState(Bool_t ovlp, Double_t *point=0);
   virtual Bool_t       PopState(Double_t *point=0);
   virtual Bool_t       PopState(Int_t level, Double_t *point=0);
   virtual void         PopDummy(Int_t ipop=9999) {fStackLevel=(ipop>fStackLevel)?(fStackLevel-1):(ipop-1);}
   virtual void         Refresh();
   void                 SetDefaultLevel(Int_t level) {fDefaultLevel = level;}
   Bool_t               SetPersistency();                 
   void                 Status();

 ClassDef(TGeoNodeCache, 0)        // cache of reusable physical nodes
};

R__EXTERN TGeoNodeCache *gGeoNodeCache;

/*************************************************************************
 * TGeoCacheDummy - dummy cache of nodes
 *    
 *
 *************************************************************************/

class TGeoCacheDummy : public TGeoNodeCache
{
private:
   TGeoNode            *fTop;           // top node
   TGeoNode            *fNode;          // current node  
   TGeoHMatrix         *fMatrix;        // current matrix
   TGeoHMatrix        **fMatrixBranch;  // current branch of global matrices
   TGeoNode           **fNodeBranch;    // current branch of nodes
public:
   TGeoCacheDummy();
   TGeoCacheDummy(TGeoNode *top);
   virtual ~TGeoCacheDummy();

   virtual Bool_t       CdDown(Int_t index, Bool_t make=kTRUE);
   virtual void         CdTop() {fLevel=0; fNode=fTop; fMatrix=fMatrixBranch[0];}
   virtual void         CdUp();
   virtual void         CleanCache() {;}
   virtual void         ClearDaughter(Int_t index) {;}
   virtual void         ClearNode(Int_t nindex) {;}
   virtual void         Compact() {;}
   virtual void         DeleteCaches() {;}
   virtual Bool_t       DumpNodes() {return kFALSE;}

   virtual void        *GetBranch() {return fNodeBranch;}
   virtual TGeoHMatrix *GetCurrentMatrix() {return fMatrix;}
   Int_t                GetCurrentNode() {return 0;}
   virtual Int_t        GetFreeSpace()   {return kGeoCacheMaxSize;}
   virtual void        *GetMatrices() {return fMatrixBranch;}
   virtual TGeoNode    *GetMother(Int_t up=1) {return ((fLevel-up)>=0)?fNodeBranch[fLevel-up]:0;}
   virtual TGeoNode    *GetNode() {return fNode;}
   virtual Int_t        GetNfree()       {return kGeoCacheMaxSize;}
   virtual Int_t        GetNused()       {return 0;}
   virtual const char  *GetPath(); 
   virtual Int_t        GetUsageCount() {return 0;}
   virtual void         IncreasePool(Int_t size) {;}
   virtual void         IncrementUsageCount() {;}
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         MasterToLocal(Double_t *master, Double_t *local);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master);
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local);
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);
   virtual void         PrintNode() {;}
//   virtual Int_t        PushState(Bool_t ovlp, Double_t *point=0);
//   virtual Bool_t       PopState(Double_t *point=0);
//   virtual Bool_t       PopState(Int_t level, Double_t *point=0);
   virtual void         Refresh();
   Bool_t               SetPersistency() {return kFALSE;}                 
   void                 Status() {;}

 ClassDef(TGeoCacheDummy, 0)        // dummy cache of physical nodes
};


/*************************************************************************
 * TGeoMatrixCache - cache of global matrices
 *    
 *
 *************************************************************************/

class TGeoMatrixCache
{
public:
   static const Int_t    kGeoDefaultIncrease;
   static const Int_t    kGeoMinCacheSize;
   static const UChar_t  kGeoMaskX;
   static const UChar_t  kGeoMaskY;
   static const UChar_t  kGeoMaskZ;
   static const UChar_t  kGeoMaskXYZ;
   static const UChar_t  kGeoMaskRot;
   static const UChar_t  kGeoMaskScale;
private:
   UInt_t                fSize[7];    // size of matrix caches
   UInt_t                fFree[7];    // offset of first free matrices
   Double_t             *fCache[7];   // pointers to all caches
   TBits                *fBits[7];    // flags for matrix usage
   Int_t                 fMatrix;     // current global transformation
   Int_t                 fHandler;    // current matrix handler
   Int_t                 fCacheId;    // current cache id
   Int_t                 fLength;     // length of current matrix
   TGeoMatHandler      **fHandlers;   // handlers for cached matrices
protected:
   void                  IncreaseCache();
public:
   TGeoMatrixCache();
   TGeoMatrixCache(Int_t size);
   virtual ~TGeoMatrixCache();

   Int_t                 AddMatrix(TGeoMatrix *matrix);
   void                  cd(Int_t mindex);
   void                  ClearMatrix(Int_t index);
   void                  GetMatrix(TGeoHMatrix *matrix);
   void                  LocalToMaster(Double_t *local, Double_t *master);
   void                  LocalToMasterVect(Double_t *local, Double_t *master);
   void                  LocalToMasterBomb(Double_t *local, Double_t *master);
   void                  MasterToLocal(Double_t *master, Double_t *local);
   void                  MasterToLocalVect(Double_t *master, Double_t *local);
   void                  MasterToLocalBomb(Double_t *master, Double_t *local);
   void                  Status();

 ClassDef(TGeoMatrixCache, 0)    // cache of compressed global matrices
};

R__EXTERN TGeoMatrixCache *gGeoMatrixCache;


/*************************************************************************
 * TGeoNodePos - the physical geometry node with links to mother and
 *   daughters. 
 *
 *************************************************************************/

class TGeoNodePos : public TObject
{
public:
   static const Int_t  kPersistentNodeMask; // byte mask for persistent nodes 
   static const UChar_t  kPersistentMatrixMask; // byte mask for persistent matrices 
   static const UInt_t   kNoMatrix;     // initial value for fGlobalMatrix
private:
   Int_t                 fNdaughters;   // number of daughters
   Int_t                *fDaughters;    // [fNdaughters] list of daughters offsets
   Int_t                 fMatrix;       // index of global matrix
   Int_t                 fCount;        // usage counter
   TGeoNode             *fNode;         // pointer to corresponding node from the logical tree
public:
   TGeoNodePos();
   TGeoNodePos(Int_t ndaughters);
   virtual ~TGeoNodePos();

   Int_t                 AddDaughter(Int_t ind, Int_t nindex) {return (fDaughters[ind]=nindex);}
   Int_t                 AddMatrix(TGeoMatrix *global);
   void                  ClearDaughter(Int_t ind) {fDaughters[ind] = 0;}
   void                  ClearMatrix();
   Int_t                 GetDaughter(Int_t ind);
//   void                  GetMatrix(TGeoHMatrix *matrix);
   Int_t                 GetMatrixInd()       {return fMatrix;}
   const char           *GetName() const      {return fNode->GetName();}
   Int_t                 GetNdaughters()      {return fNdaughters;}
   TGeoNode             *GetNode()            {return fNode;}
   Int_t                 GetUsageCount()      {return (fCount & 0x7FFFFFFF);}
   Bool_t                HasDaughters();
   Bool_t                IsPersistent() 
                            {return (((fCount & kPersistentNodeMask)==0)?kFALSE:kTRUE);}
   void                  IncrementUsageCount()     {fCount++;}
   void                  Map(TGeoNode *node);
//   void                  Reset();
   void                  ResetCount()         {fCount &= kPersistentNodeMask;}
   void                  SetMatrix(Int_t mat_ind) {fMatrix = mat_ind;}
   void                  SetPersistency(Bool_t flag=kTRUE);

 ClassDef(TGeoNodePos, 1)      // the physical nodes
};

/*************************************************************************
 * TGeoNodeArray - base class for physical nodes arrays
 *    The structure of a node is stored in the following way :
 *    Int_t *offset = fArray+inode*nodesize position of node 'inode' in fArray
 *      ->offset+0   - pointer to physical node : fNode-gSystem
 *
 *                            |bit7 | b6  | b5 | b4  | b3 | b2  | b1 | b0 |
 *      ->offset+1   - Byte0= |scale|rot  | Z  | Y   | X  |matrix cache id| 
 *                     | B3 | B2 | B1 | - matrix index in cache
 *      ->offset+2   - B0|b7 = node persistency ; b6 = has daughters
 *                     B3|B2|B1|B0 - usage count
 *      ->offset+3+i - Byte0=daughter array index, 
 *                   |B3|B2|B1| - index of daughter i 
 *      Total length : nodesize = (3+fNdaughters)*sizeof(Int_t) 
 *
 *************************************************************************/

class TGeoNodeArray : public TObject
{
public:
   static const Int_t   kGeoArrayMaxSize;   // maximum cache size
   static const Int_t   kGeoArrayInitSize;  // initial cache size
   static const Int_t   kGeoReleasedSpace;  // default number of nodes released on cleaning
private:
   Int_t                fNodeSize;    // size of a node in bytes
   Int_t                fNdaughters;  // number of daughters for nodes in this array
   Int_t               *fOffset;      // [fSize*fNodeSize] offset of the current node
protected:
   Int_t                fSize;        // number of nodes stored in array
   Int_t                fFirstFree;   // index of first free location
   Int_t                fCurrent;     // index of current node
   Int_t                fNused;       // number of used nodes
   TBits               *fBits;        // occupancy flags
private:
   Int_t               *fArray;       // array of nodes   
public:
   TGeoNodeArray();
   TGeoNodeArray(Int_t ndaughters, Int_t size=0);
   virtual ~TGeoNodeArray();

   virtual Int_t        AddDaughter(TGeoNode *node, Int_t i)
                                {return (fOffset[3+i]=gGeoNodeCache->AddNode(node));}
   virtual Int_t        AddNode(TGeoNode *node);
   virtual Int_t        AddMatrix(TGeoMatrix *global) 
                                 {return (fOffset[1]=gGeoMatrixCache->AddMatrix(global));}
   virtual void         cd(Int_t inode)  {fOffset = fArray+inode*fNodeSize;
                                          fCurrent = inode;}
   virtual void         ClearDaughter(Int_t ind);
   virtual void         ClearMatrix();
   virtual void         ClearNode();
   virtual void         Compact();
   void                 DeleteArray();
   virtual Int_t        GetDaughter(Int_t ind) {return fOffset[3+ind];}
//   virtual void         GetMatrix(TGeoHMatrix *matrix) 
//                                 {gGeoMatrixCache->GetMatrix(fOffset[1], matrix);}
   virtual Int_t        GetMatrixInd()   {return fOffset[1];}
   virtual Int_t        GetNdaughters()  {return fNdaughters;}
   virtual TGeoNode    *GetNode()        {return (TGeoNode*)(fOffset[0]+(Int_t)gSystem);}
   Int_t                GetNused()       {return fNused;}
   Int_t                GetSize()        {return fSize;}
   virtual Int_t        GetUsageCount()  {return (fOffset[2]&0x3FFFFFFF);}
   virtual Bool_t       HasDaughters();
   virtual void         IncreaseArray();
   virtual void         IncrementUsageCount() {fOffset[2]++;}
   virtual Bool_t       IsPersistent();
   virtual void         SetMatrix(Int_t mind) {fOffset[1] = mind;}
   virtual void         SetPersistency(Bool_t flag=kTRUE);   

  ClassDef(TGeoNodeArray, 1)     // array of cached physical nodes
};

/*************************************************************************
 * TGeoNodeObjArray - container class for nodes with more than 254
 *     daughters. 
 *
 *************************************************************************/

class TGeoNodeObjArray : public TGeoNodeArray
{
private:
   TObjArray           *fObjArray;   //[fSize] array of TGeoNodePos objects
   TGeoNodePos         *fCurrent;    // current node
   Int_t                fIndex;      // index of current node
public:
   TGeoNodeObjArray();
   TGeoNodeObjArray(Int_t size);
   virtual ~TGeoNodeObjArray();

   virtual Int_t        AddDaughter(TGeoNode *node, Int_t i);
   virtual Int_t        AddNode(TGeoNode *node);
   virtual Int_t        AddMatrix(TGeoMatrix *global);
   virtual void         cd(Int_t inode);
   virtual void         ClearDaughter(Int_t ind);
   virtual void         ClearMatrix();
   virtual void         ClearNode();
//   virtual void         GetMatrix(TGeoHMatrix *matrix) {fCurrent->GetMatrix(matrix);}
   virtual Int_t        GetMatrixInd() {return fCurrent->GetMatrixInd();}
   virtual Int_t        GetDaughter(Int_t ind) {return fCurrent->GetDaughter(ind);} 
   virtual Int_t        GetNdaughters() {return fCurrent->GetNdaughters();}
   virtual TGeoNode    *GetNode() {return fCurrent->GetNode();}
   virtual Int_t        GetUsageCount() {return fCurrent->GetUsageCount();}
   virtual Bool_t       HasDaughters() {return fCurrent->HasDaughters();}
   virtual void         IncreaseArray();
   virtual void         IncrementUsageCount() {fCurrent->IncrementUsageCount();}
   virtual Bool_t       IsPersistent() {return fCurrent->IsPersistent();}
   virtual void         SetPersistency(Bool_t flag=kTRUE) {fCurrent->SetPersistency(flag);}   
   virtual void         SetMatrix(Int_t mind) {fCurrent->SetMatrix(mind);}

  ClassDef(TGeoNodeObjArray, 1)     // array of physical nodes objects
};



/*************************************************************************
 * TGeoMatHandler - generic matrix handlers for computing master->local
 *    and local->master transformations directly from matrix cache
 *
 *************************************************************************/

class TGeoMatHandler
{
protected:
   Double_t            *fLocation;  // adress of current matrix
public:
   TGeoMatHandler();
   virtual ~TGeoMatHandler() {}

   void                 SetLocation(Double_t *add) {fLocation=add;}   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix) = 0;
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix) = 0;
   virtual void         LocalToMaster(Double_t *local, Double_t *master) = 0;
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) = 0;
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) = 0;
   virtual void         MasterToLocal(Double_t *master, Double_t *local) = 0;   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local) = 0;   
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local) = 0;   
  ClassDef(TGeoMatHandler,0)      // global matrix cache handler
};

/*************************************************************************
 * TGeoMatHandlerId - handler for id transformations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerId : public TGeoMatHandler
{
public:
   TGeoMatHandlerId() {}
   virtual ~TGeoMatHandlerId() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix) {;}
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix) {;}
   virtual void         LocalToMaster(Double_t *local, Double_t *master) 
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) 
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) 
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   
  ClassDef(TGeoMatHandlerId,0)      // global matrix cache handler id
};


/*************************************************************************
 * TGeoMatHandlerX - handler for translations on X axis
 *    
 *
 *************************************************************************/

class TGeoMatHandlerX : public TGeoMatHandler
{
public:
   TGeoMatHandlerX() {}
   virtual ~TGeoMatHandlerX() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerX,0)      // global matrix cache handler X
};

/*************************************************************************
 * TGeoMatHandlerY - handler for translations on Y axis
 *    
 *
 *************************************************************************/

class TGeoMatHandlerY : public TGeoMatHandler
{
public:
   TGeoMatHandlerY() {}
   virtual ~TGeoMatHandlerY() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerY,0)      // global matrix cache handler Y
};

/*************************************************************************
 * TGeoMatHandlerZ - handler for translations on Z axis
 *    
 *
 *************************************************************************/

class TGeoMatHandlerZ : public TGeoMatHandler
{
public:
   TGeoMatHandlerZ() {}
   virtual ~TGeoMatHandlerZ() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerZ,0)      // global matrix cache handler Z
};

/*************************************************************************
 * TGeoMatHandlerXY - handler for XY translations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerXY : public TGeoMatHandler
{
public:
   TGeoMatHandlerXY() {}
   virtual ~TGeoMatHandlerXY() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerXY,0)      // global matrix cache handler XY
};

/*************************************************************************
 * TGeoMatHandlerXZ - handler for XZ translations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerXZ : public TGeoMatHandler
{
public:
   TGeoMatHandlerXZ() {}
   virtual ~TGeoMatHandlerXZ() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerXZ,0)      // global matrix cache handler XZ
};

/*************************************************************************
 * TGeoMatHandlerYZ - handler for YZ translations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerYZ : public TGeoMatHandler
{
public:
   TGeoMatHandlerYZ() {}
   virtual ~TGeoMatHandlerYZ() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerYZ,0)      // global matrix cache handler YZ
};

/*************************************************************************
 * TGeoMatHandlerXYZ - handler for general translations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerXYZ : public TGeoMatHandler
{
public:
   TGeoMatHandlerXYZ() {}
   virtual ~TGeoMatHandlerXYZ() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerXYZ,0)      // global matrix cache handler XYZ
};

/*************************************************************************
 * TGeoMatHandlerRot - handler for rotations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerRot : public TGeoMatHandler
{
public:
   TGeoMatHandlerRot() {}
   virtual ~TGeoMatHandlerRot() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master)
                           {TGeoMatHandlerRot::LocalToMaster(local, master);}
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local)
                           {TGeoMatHandlerRot::MasterToLocal(master, local);}
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master)
                           {LocalToMaster(local, master);}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local)
                           {MasterToLocal(master, local);}
   
  ClassDef(TGeoMatHandlerRot,0)      // global matrix cache handler rot
};

/*************************************************************************
 * TGeoMatHandlerRotTr - handler for general transformations without scaling
 *    
 *
 *************************************************************************/

class TGeoMatHandlerRotTr : public TGeoMatHandler
{
public:
   TGeoMatHandlerRotTr() {}
   virtual ~TGeoMatHandlerRotTr() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master);
   virtual void         MasterToLocal(Double_t *master, Double_t *local);   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local);   
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local);   
   
  ClassDef(TGeoMatHandlerRotTr,0)      // global matrix cache handler rot-tr
};

/*************************************************************************
 * TGeoMatHandlerScl - handler for scale transformations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerScl : public TGeoMatHandler
{
public:
   TGeoMatHandlerScl() {}
   virtual ~TGeoMatHandlerScl() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master) {;}
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocal(Double_t *master, Double_t *local) {;}   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local) {;}   
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local) {;}   
   
  ClassDef(TGeoMatHandlerScl,0)      // global matrix cache handler scale
};

/*************************************************************************
 * TGeoMatHandlerTrScl - handler for translations + scale
 *    
 *
 *************************************************************************/

class TGeoMatHandlerTrScl : public TGeoMatHandler
{
public:
   TGeoMatHandlerTrScl() {}
   virtual ~TGeoMatHandlerTrScl() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master) {;}
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocal(Double_t *master, Double_t *local) {;}   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local) {;}   
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local) {;}   
   
  ClassDef(TGeoMatHandlerTrScl,0)      // global matrix cache handler tr-scale
};

/*************************************************************************
 * TGeoMatHandlerRotScl - handler for rotations + scale
 *    
 *
 *************************************************************************/

class TGeoMatHandlerRotScl : public TGeoMatHandler
{
public:
   TGeoMatHandlerRotScl() {}
   virtual ~TGeoMatHandlerRotScl() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master) {;}
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocal(Double_t *master, Double_t *local) {;}   
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local) {;}   
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local) {;}   
   
  ClassDef(TGeoMatHandlerRotScl,0)      // global matrix cache handler rot-scale
};

/*************************************************************************
 * TGeoMatHandlerTrScl - handler for most general transformations
 *    
 *
 *************************************************************************/

class TGeoMatHandlerRotTrScl : public TGeoMatHandler
{
public:
   TGeoMatHandlerRotTrScl() {}
   virtual ~TGeoMatHandlerRotTrScl() {}
   
   virtual void         AddMatrix(Double_t *to, TGeoMatrix *matrix);
   virtual void         GetMatrix(Double_t *from, TGeoHMatrix *matrix);
   virtual void         LocalToMaster(Double_t *local, Double_t *master) {;}
   virtual void         LocalToMasterVect(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocal(Double_t *master, Double_t *local) {;} 
   virtual void         MasterToLocalVect(Double_t *master, Double_t *local) {;}   
   virtual void         LocalToMasterBomb(Double_t *local, Double_t *master) {;}
   virtual void         MasterToLocalBomb(Double_t *master, Double_t *local) {;}   
   
  ClassDef(TGeoMatHandlerRotTrScl,0)      // global matrix cache handler rot-tr-scale
};

inline void TGeoNodeCache::CdCache() {fCache[fCurrentCache]->cd(fCurrentIndex);}
inline void TGeoNodeCache::ClearDaughter(Int_t index) {CdCache(); 
                                   fCache[fCurrentCache]->ClearDaughter(index);}
inline void TGeoNodeCache::IncrementUsageCount() {fCache[fCurrentCache]->IncrementUsageCount();
                                   fCount++;}
inline Int_t TGeoNodeCache::GetUsageCount() {return fCache[fCurrentCache]->GetUsageCount();}
inline TGeoNode *TGeoNodeCache::GetNode() {return fCache[fCurrentCache]->GetNode();}
inline Int_t TGeoNodeCache::PushState(Bool_t ovlp, Double_t *point) 
                           {if (fStackLevel>=kGeoCacheStackSize) return 0; 
                            ((TGeoCacheState*)fStack->At(fStackLevel))->SetState(fLevel,ovlp,point);
                            return ++fStackLevel;}
inline Bool_t TGeoNodeCache::PopState(Double_t *point) 
                           {if (!fStackLevel) return 0;
                            ((TGeoCacheState*)fStack->At(--fStackLevel))->GetState(fLevel,point);
                            Refresh(); return (fStackLevel+1);}
inline Bool_t TGeoNodeCache::PopState(Int_t level, Double_t *point) 
                           {if (level<=0) return 0;
                            ((TGeoCacheState*)fStack->At(level-1))->GetState(fLevel,point);
                            Refresh(); return level;}
inline void TGeoNodeCache::Refresh() 
                           {fCurrentNode=fBranch[fLevel]; fCurrentCache=CacheId(fCurrentNode);
                            fCurrentIndex=Index(fCurrentNode); fCache[fCurrentCache]->cd(fCurrentIndex);
                            gGeoMatrixCache->cd(fMatrices[fLevel]);}
inline void TGeoCacheDummy::Refresh() 
                           {fNode=fNodeBranch[fLevel]; fMatrix=fMatrixBranch[fLevel];}
inline void TGeoMatrixCache::LocalToMaster(Double_t *local, Double_t *master)
                            {fHandlers[fHandler]->LocalToMaster(local, master);}
inline void TGeoMatrixCache::LocalToMasterVect(Double_t *local, Double_t *master)
                            {fHandlers[fHandler]->LocalToMasterVect(local, master);}
inline void TGeoMatrixCache::LocalToMasterBomb(Double_t *local, Double_t *master)
                            {fHandlers[fHandler]->LocalToMasterBomb(local, master);}
inline void TGeoMatrixCache::MasterToLocal(Double_t *master, Double_t *local)
                            {fHandlers[fHandler]->MasterToLocal(master, local);}
inline void TGeoMatrixCache::MasterToLocalVect(Double_t *master, Double_t *local)
                            {fHandlers[fHandler]->MasterToLocalVect(master, local);}
inline void TGeoMatrixCache::MasterToLocalBomb(Double_t *master, Double_t *local)
                            {fHandlers[fHandler]->MasterToLocalBomb(master, local);}
inline void TGeoNodeCache::LocalToMaster(Double_t *local, Double_t *master)
                            {gGeoMatrixCache->LocalToMaster(local, master);}
inline void TGeoNodeCache::LocalToMasterVect(Double_t *local, Double_t *master)
                            {gGeoMatrixCache->LocalToMasterVect(local, master);}
inline void TGeoNodeCache::LocalToMasterBomb(Double_t *local, Double_t *master)
                            {gGeoMatrixCache->LocalToMasterBomb(local, master);}
inline void TGeoNodeCache::MasterToLocal(Double_t *master, Double_t *local)
                            {gGeoMatrixCache->MasterToLocal(master, local);}
inline void TGeoNodeCache::MasterToLocalVect(Double_t *master, Double_t *local)
                            {gGeoMatrixCache->MasterToLocalVect(master, local);}
inline void TGeoNodeCache::MasterToLocalBomb(Double_t *master, Double_t *local)
                            {gGeoMatrixCache->MasterToLocalBomb(master, local);}

#endif

 
