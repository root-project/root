// @(#):$Id$
// Author: Andrei Gheata   01/03/11

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoBranchArray
#define ROOT_TGeoBranchArray

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoBranchArray - An array of daughter indices making a geometry path. //
//   Can be used to backup/restore a state. Allocated contiguously in     //
//   memory.                                                              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoNavigator;
class TGeoNode;

class TGeoBranchArray : public TObject
{
protected:
   Int_t             fLevel;          // Branch depth
   Int_t             fMaxLevel;       // Array length
   TGeoHMatrix       fMatrix;         // Global matrix (owned)
   TGeoNode        **fArray;          //![fMaxLevel+1] Array of nodes
   TGeoNode         *fRealArray[1];   // Beginning address of the array of nodes

private:
   TGeoBranchArray(Int_t level);                       // not allowed
   TGeoBranchArray(const TGeoBranchArray&);            // not allowed
public:
   enum EGeoBATypes {
      kBASelfAlloc =  BIT(14)             // does self allocation or not
   };   
   // This replaces the dummy constructor to make sure that I/O can be
   // performed while the user is only allowed to use the static maker
   TGeoBranchArray(TRootIOCtor*) : TObject(), fLevel(0), fMaxLevel(0), fMatrix(), fArray(0) {}

   // The static maker to be use to create an instance of the branch array
   static TGeoBranchArray *MakeInstance(size_t maxlevel);

   // The static maker to be use to create an instance of the branch array
   static TGeoBranchArray *MakeInstanceAt(size_t maxlevel, void *addr);

   // The equivalent of the copy constructor
   static TGeoBranchArray *MakeCopy(const TGeoBranchArray &other);

   // The equivalent of the copy constructor
   static TGeoBranchArray *MakeCopyAt(const TGeoBranchArray &other, void *addr);

   // The equivalent of the destructor
   static void             ReleaseInstance(TGeoBranchArray *obj);

   // Assignment allowed
   TGeoBranchArray& operator=(const TGeoBranchArray&);

   // Fast copy based on memcpy to destination array
   void                    CopyTo(TGeoBranchArray *dest);
   
   // Equivalent of sizeof function
   static size_t SizeOf(size_t maxlevel)
      { return (sizeof(TGeoBranchArray)+sizeof(TGeoBranchArray*)*(maxlevel)); }

   // Equivalent of sizeof function
   static size_t SizeOfInstance(size_t maxlevel)
      { return (sizeof(TGeoBranchArray)+sizeof(TGeoBranchArray*)*(maxlevel)); }

   inline size_t SizeOf() const
      { return (sizeof(TGeoBranchArray)+sizeof(TGeoBranchArray*)*(fMaxLevel)); }
   
   // The data start should point to the address of the first data member,
   // after the virtual table
   void       *DataStart() const {return (void*)&fLevel;}

   // The actual size of the data for an instance, excluding the virtual table
   size_t      DataSize() const {return SizeOf()-size_t(&fLevel)+(size_t)this;}

   // Update the internal addresses of n contiguous branch array objects, starting
   // with this one
   void UpdateArray(size_t nobj);

   // Destructor. Release instance to be called instead
   virtual ~TGeoBranchArray() {}

   Bool_t operator ==(const TGeoBranchArray& other) const;
   Bool_t operator !=(const TGeoBranchArray& other) const;
   Bool_t operator >(const TGeoBranchArray& other) const;
   Bool_t operator <(const TGeoBranchArray& other) const;
   Bool_t operator >=(const TGeoBranchArray& other) const;
   Bool_t operator <=(const TGeoBranchArray& other) const;
   
   void              AddLevel(Int_t dindex);
   static Long64_t   BinarySearch(Long64_t n, const TGeoBranchArray **array, TGeoBranchArray *value);
   virtual Int_t     Compare(const TObject *obj) const;
   void              CleanMatrix();
   TGeoNode        **GetArray() const    {return fArray;}
   size_t            GetLevel() const    {return fLevel;}
   size_t            GetMaxLevel() const {return fMaxLevel;}
   const TGeoHMatrix  
                    *GetMatrix() const  {return &fMatrix;}
   TGeoNode         *GetNode(Int_t level) const {return fArray[level];}
   TGeoNode         *GetCurrentNode() const {return fArray[fLevel];}
   void              GetPath(TString &path) const;
   void              Init(TGeoNode **branch, TGeoMatrix *global, Int_t level);
   void              InitFromNavigator(TGeoNavigator *nav);
   virtual Bool_t    IsSortable() const {return kTRUE;}
   Bool_t            IsOutside() const {return (fLevel<0)?kTRUE:kFALSE;}
   virtual void      Print(Option_t *option="") const;
   static void       Sort(Int_t n, TGeoBranchArray **array, Int_t *index, Bool_t down=kTRUE);
   void              UpdateNavigator(TGeoNavigator *nav) const;
   
   ClassDef(TGeoBranchArray, 4)
};

struct compareBAasc {
   compareBAasc(TGeoBranchArray **d) : fData(d) {}
   bool operator ()(Int_t i1, Int_t i2) {return **(fData+i1) < **(fData+i2);}
   TGeoBranchArray **fData;
};

struct compareBAdesc {
   compareBAdesc(TGeoBranchArray **d) : fData(d) {}
   bool operator ()(Int_t i1, Int_t i2) {return **(fData+i1) > **(fData+i2);}
   TGeoBranchArray **fData;
};

#endif
