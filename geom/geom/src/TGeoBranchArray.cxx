// @(#):$Id$
// Author: Andrei Gheata   01/03/11

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoBranchArray
\ingroup Geometry_classes

An array of daughter indices making a geometry path. Can be used to
backup/restore a state. Allocated contiguously in memory.

To setup an object of this type, one should use:

~~~ {.cpp}
   TGeoBranchArray *array = new TGeoBranchArray(level);
   array->InitFromNavigator(nav); // To initialize from current navigator state
~~~

The navigator can be updated to reflect this path array: `array->UpdateNavigator();`
*/

#include "TGeoBranchArray.h"

#include "TMath.h"
#include "TString.h"
#include "TGeoNavigator.h"
#include "TGeoCache.h"
#include "TGeoManager.h"

ClassImp(TGeoBranchArray);

////////////////////////////////////////////////////////////////////////////////
/// Constructor. Allocates the array with a size given by level.

TGeoBranchArray::TGeoBranchArray(Int_t maxlevel)
                :fLevel(-1),
                 fMaxLevel(maxlevel),
                 fMatrix(),
                 fArray(&fRealArray[0])
{
   memset(fRealArray, 0, fMaxLevel*sizeof(TGeoNode*));
}

////////////////////////////////////////////////////////////////////////////////
/// Make an instance of the class which allocates the node array. To be
/// released using ReleaseInstance. If addr is non-zero, the user promised that
/// addr contains at least that many bytes:  size_t needed = SizeOf(maxlevel);

TGeoBranchArray * TGeoBranchArray::MakeInstance(size_t maxlevel)
{
   TGeoBranchArray* ba = 0;
   size_t needed = SizeOf(maxlevel);
   char *ptr = new char[ needed ];
   if (!ptr) return 0;
   new (ptr) TGeoBranchArray(maxlevel);
   ba = reinterpret_cast<TGeoBranchArray*>(ptr);
   ba->SetBit(kBASelfAlloc, kTRUE);
   return ba;
}

////////////////////////////////////////////////////////////////////////////////
/// Make an instance of the class which allocates the node array. To be
/// released using ReleaseInstance. If addr is non-zero, the user promised that
/// addr contains at least that many bytes:  size_t needed = SizeOf(maxlevel);

TGeoBranchArray * TGeoBranchArray::MakeInstanceAt(size_t maxlevel, void *addr)
{
   TGeoBranchArray* ba = 0;
   new (addr) TGeoBranchArray(maxlevel);
   ba = reinterpret_cast<TGeoBranchArray*>(addr);
   ba->SetBit(kBASelfAlloc, kFALSE);
   return ba;
}


////////////////////////////////////////////////////////////////////////////////
/// Make a copy of a branch array at the location (if indicated)

TGeoBranchArray * TGeoBranchArray::MakeCopy(const TGeoBranchArray &other)
{
   TGeoBranchArray *copy = 0;
   size_t needed = SizeOf(other.fMaxLevel);
   char *ptr = new char[ needed ];
   if (!ptr) return 0;
   new (ptr) TGeoBranchArray(other.fMaxLevel);
   copy = reinterpret_cast<TGeoBranchArray*>(ptr);
   copy->SetBit(kBASelfAlloc, kTRUE);
   copy->fLevel = other.fLevel;
   copy->fMatrix = other.fMatrix;
   if (other.fLevel+1) memcpy(copy->fArray, other.fArray, (other.fLevel+1)*sizeof(TGeoNode*));
   return copy;
}

////////////////////////////////////////////////////////////////////////////////
/// Make a copy of a branch array at the location (if indicated)

TGeoBranchArray * TGeoBranchArray::MakeCopyAt(const TGeoBranchArray &other, void *addr)
{
   TGeoBranchArray *copy = 0;
   new (addr) TGeoBranchArray(other.fMaxLevel);
   copy = reinterpret_cast<TGeoBranchArray*>(addr);
   copy->SetBit(kBASelfAlloc, kFALSE);
   copy->fLevel = other.fLevel;
   copy->fMatrix = other.fMatrix;
   if (other.fLevel+1) memcpy(copy->fArray, other.fArray, (other.fLevel+1)*sizeof(TGeoNode*));
   return copy;
}


////////////////////////////////////////////////////////////////////////////////
/// Raw memcpy of the branch array content to an existing destination.

void TGeoBranchArray::CopyTo(TGeoBranchArray *dest)
{
   memcpy(dest->DataStart(), DataStart(), DataSize());
   dest->fArray = &(dest->fRealArray[0]);
}

////////////////////////////////////////////////////////////////////////////////
/// Releases the space allocated for the object

void TGeoBranchArray::ReleaseInstance(TGeoBranchArray *obj)
{
   obj->~TGeoBranchArray();
   if (obj->TestBit(kBASelfAlloc)) delete [] (char*)obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the internal addresses for n contiguous objects which have the same
/// fMaxLevel
/// Updates the internal addresses for n contiguous objects which have the same fMaxLevel

void TGeoBranchArray::UpdateArray(size_t nobj)
{
   size_t needed = SizeOf();
//   char *where = &fArray;
//   for (size_t i=0; i<nobj; ++i, where += needed) {
//      TGeoNode ***array =  reinterpret_cast<TGeoNode***>(where);
//      *array = ((void**)where)+1;
//   }
   char *where = reinterpret_cast<char*>(this);
   for (size_t i=0; i<nobj; ++i, where += needed) {
      TGeoBranchArray *obj = reinterpret_cast<TGeoBranchArray*>(where);
      obj->fArray = &(obj->fRealArray[0]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Not callable anymore. Use TGeoBranchArray::MakeCopy instead

TGeoBranchArray::TGeoBranchArray(const TGeoBranchArray&  other)
                :TObject(other),
                 fLevel(other.fLevel),
                 fMaxLevel(other.fMaxLevel),
                 fMatrix(other.fMatrix),
                 fArray(NULL)
{
   if (fMaxLevel) {
      fArray = new TGeoNode*[fMaxLevel];
      if (fLevel+1) memcpy(fArray, other.fArray, (fLevel+1)*sizeof(TGeoNode*));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment. Not valid anymore. Use TGeoBranchArray::MakeCopy instead

TGeoBranchArray& TGeoBranchArray::operator=(const TGeoBranchArray& other)
{
   if (&other == this) return *this;
   fLevel = other.fLevel;
   fMatrix.CopyFrom(&other.fMatrix);
   if (fLevel+1) memcpy(fArray, other.fArray, (fLevel+1)*sizeof(TGeoNode*));
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add and extra daughter to the current path array. No validity check performed !

void TGeoBranchArray::AddLevel(Int_t dindex)
{
   if (fLevel<0) {
      Error("AddLevel", "You must initialize from navigator or copy from another branch array first.");
      return;
   }
   if (fLevel>fMaxLevel) {
      Fatal("AddLevel", "Max level = %d reached\n", fMaxLevel);
      return;
   }
   fLevel++;
/*
   if (fLevel+1>fMaxLevel) {
      TGeoNode **array = new TGeoNode*[fLevel+1];
      memcpy(array, fArray, fLevel*sizeof(TGeoNode*));
      delete [] fArray;
      fArray = array;
   }
*/
   fArray[fLevel] = fArray[fLevel-1]->GetVolume()->GetNode(dindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Is equal operator.

Bool_t TGeoBranchArray::operator ==(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value==0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Not equal operator.

Bool_t TGeoBranchArray::operator !=(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value!=0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is equal operator.

Bool_t TGeoBranchArray::operator >(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value>0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is equal operator.

Bool_t TGeoBranchArray::operator <(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value<0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is equal operator.

Bool_t TGeoBranchArray::operator >=(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value>=0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is equal operator.

Bool_t TGeoBranchArray::operator <=(const TGeoBranchArray& other) const
{
   Int_t value = Compare(&other);
   if (value<=0) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Binary search in an array of n pointers to branch arrays, to locate value.
/// Returns element index or index of nearest element smaller than value

Long64_t TGeoBranchArray::BinarySearch(Long64_t n, const TGeoBranchArray **array, TGeoBranchArray *value)
{
   Long64_t nabove, nbelow, middle;
   const TGeoBranchArray *pind;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      pind = array[middle-1];
      if (*value == *pind) return middle-1;
      if (*value  < *pind) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;
}

////////////////////////////////////////////////////////////////////////////////
/// Compare with other object of same type. Returns -1 if this is smaller (first
/// smaller array value prevails), 0 if equal (size and values) and 1 if this is
/// larger.

Int_t TGeoBranchArray::Compare(const TObject *obj) const
{
   Int_t i;
   TGeoBranchArray *other = (TGeoBranchArray*)obj;
   Int_t otherLevel = other->GetLevel();
   Int_t maxLevel = TMath::Min(fLevel, otherLevel);
   TGeoNode **otherArray = other->GetArray();
   for (i=0; i<maxLevel+1; i++) {
      if (fArray[i]==otherArray[i]) continue;
      if ((Long64_t)fArray[i]<(Long64_t)otherArray[i]) return -1;
      return 1;
   }
   if (fLevel==otherLevel) return 0;
   if (fLevel<otherLevel) return -1;
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Garbage collect the stored matrix.

void TGeoBranchArray::CleanMatrix()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Init the branch array from an array of nodes, the global matrix for the path and
/// the level.

void TGeoBranchArray::Init(TGeoNode **branch, TGeoMatrix *global, Int_t level)
{
   fMatrix.CopyFrom(global);
   if (level>fMaxLevel) {
      Fatal("Init", "Requested level %d exceeds maximum level %d", level+1, fMaxLevel);
      return;
   }
   fLevel = level;
   memcpy(fArray, branch, (fLevel+1)*sizeof(TGeoNode*));
}

////////////////////////////////////////////////////////////////////////////////
/// Init the branch array from current navigator state.

void TGeoBranchArray::InitFromNavigator(TGeoNavigator *nav)
{
   TGeoNodeCache *cache = nav->GetCache();
   const TGeoNode **branch = (const TGeoNode**)cache->GetBranch();
   Int_t level = cache->GetLevel();
   fMatrix.CopyFrom(cache->GetCurrentMatrix());
   if (level>fMaxLevel) {
      Fatal("InitFromNavigator", "Requested level %d exceeds maximum level %d", level+1, fMaxLevel);
      return;
   }
   fLevel = level;
   memcpy(fArray, branch, (fLevel+1)*sizeof(TGeoNode*));
   if (nav->IsOutside()) fLevel = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill path pointed by the array.

void TGeoBranchArray::GetPath(TString &path) const
{
   path = "";
   if (!fArray || !fArray[0]) return;
   for (Int_t i=0; i<fLevel+1; i++) {
      path += "/";
      path += fArray[i]->GetName();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print branch information

void TGeoBranchArray::Print(Option_t *) const
{
   TString path;
   GetPath(path);
   printf("branch:    %s\n", path.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Sorting of an array of branch array pointers.

void TGeoBranchArray::Sort(Int_t n, TGeoBranchArray **array, Int_t *index, Bool_t down)
{
   for (Int_t i=0; i<n; i++) index[i] = i;
   if (down)
      std::sort(index, index + n, compareBAdesc(array));
   else
      std::sort(index, index + n, compareBAasc(array));
}

////////////////////////////////////////////////////////////////////////////////
/// Update the navigator to reflect the branch.
///   nav->CdTop();

void TGeoBranchArray::UpdateNavigator(TGeoNavigator *nav) const
{
   if (fLevel<0) {nav->SetOutside(kTRUE); return;}
   Int_t matchlev = 0;
   Int_t navlev = nav->GetLevel();
   Int_t i;
   Int_t maxlev = TMath::Min(fLevel, navlev);
   for (i=1; i<maxlev+1; ++i) {
     if (fArray[i] != nav->GetMother(navlev-i)) break;
     matchlev++;
   }
   // Go to matching level
   for (i=0; i<navlev-matchlev; i++) nav->CdUp();
   for (i=matchlev+1; i<fLevel+1; i++) nav->CdDown(fArray[i]);
}
