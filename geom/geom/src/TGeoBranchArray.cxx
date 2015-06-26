// @(#):$Id$
// Author: Andrei Gheata   01/03/11

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////
//
// TGeoBranchArray - An array of daughter indices making a geometry path.
//   Can be used to backup/restore a state. To setup an object of this type,
// one should use:
//   TGeoBranchArray *array = new TGeoBranchArray(level);
//   array->InitFromNavigator(nav); (To initialize from current navigator state)
// The navigator can be updated to reflect this path array:
//   array->UpdateNavigator();
//
/////////////////////////////////////////////////////////////////////////////

#include "TGeoBranchArray.h"

#include "TMath.h"
#include "TThread.h"
#include "TString.h"
#include "TGeoMatrix.h"
#include "TGeoNavigator.h"
#include "TGeoCache.h"
#include "TGeoManager.h"

ClassImp(TGeoBranchArray)

////////////////////////////////////////////////////////////////////////////////
/// Constructor. Alocates the array with a size given by level.

TGeoBranchArray::TGeoBranchArray(Int_t level)
                :fLevel(level),
                 fMaxLevel(0),
                 fArray(NULL),
                 fMatrix(NULL),
                 fClient(NULL)
{
   fMaxLevel = (fLevel+1 > 10) ? fLevel+1:10;
   fArray = new TGeoNode*[fMaxLevel];
   fMatrix = new TGeoHMatrix();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGeoBranchArray::~TGeoBranchArray()
{
   delete [] fArray;
   delete fMatrix;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TGeoBranchArray::TGeoBranchArray(const TGeoBranchArray&  other)
                :TObject(other),
                 fLevel(other.fLevel),
                 fMaxLevel(other.fMaxLevel),
                 fArray(NULL),
                 fMatrix(NULL),
                 fClient(other.fClient)
{
   if (fMaxLevel) {
      fArray = new TGeoNode*[fMaxLevel];
      if (fLevel+1) memcpy(fArray, other.fArray, (fLevel+1)*sizeof(TGeoNode*));
   }
   if (other.fMatrix) fMatrix = new TGeoHMatrix(*(other.fMatrix));
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment.

TGeoBranchArray& TGeoBranchArray::operator=(const TGeoBranchArray& other)
{
   if (&other == this) return *this;
//   TThread::Lock();
   TObject::operator=(other);
   // Check if the array exists and has to be resized
   if (fArray) {
      if (fMaxLevel<other.fLevel+1) {
         fMaxLevel = other.fMaxLevel;
         delete [] fArray;
         fArray = new TGeoNode*[fMaxLevel];
      }
   } else {
      fMaxLevel = other.fMaxLevel;
      fArray = new TGeoNode*[fMaxLevel];
   }
   fLevel = other.fLevel;
   if (fLevel+1) memcpy(fArray, other.fArray, (fLevel+1)*sizeof(TGeoNode*));
   if (other.fMatrix) {
      if (!fMatrix) fMatrix = new TGeoHMatrix();
      fMatrix->CopyFrom(other.fMatrix);
   }
   fClient = other.fClient;
//   TThread::UnLock();
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add and extra daughter to the current path array. No validity check performed !

void TGeoBranchArray::AddLevel(Int_t dindex)
{
   if (fLevel<=0) {
      Error("AddLevel", "You must initialize from navigator or copy from another branch array first.");
      return;
   }
   fLevel++;
   if (fLevel+1>fMaxLevel) {
      TGeoNode **array = new TGeoNode*[fLevel+1];
      memcpy(array, fArray, fLevel*sizeof(TGeoNode*));
      delete [] fArray;
      fArray = array;
   }
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
   delete fMatrix; fMatrix = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Init the branch array from an array of nodes, the global matrix for the path and
/// the level.

void TGeoBranchArray::Init(TGeoNode **branch, TGeoMatrix *global, Int_t level)
{
   if (!fMatrix) fMatrix = new TGeoHMatrix();
   fMatrix->CopyFrom(global);
   if (!fArray || level+1>fMaxLevel) {
      delete [] fArray;
      fMaxLevel = level+1;
      fArray = new TGeoNode*[fMaxLevel];
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
   if (!fMatrix) fMatrix = new TGeoHMatrix();
   fMatrix->CopyFrom(cache->GetCurrentMatrix());
   if (!fArray || level+1>fMaxLevel) {
      delete [] fArray;
      fMaxLevel = level+1;
      fArray = new TGeoNode*[fMaxLevel];
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
   if (!fArray) return;
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

void TGeoBranchArray::UpdateNavigator(TGeoNavigator *nav) const
{
   nav->CdTop();
   if (fLevel<0) {nav->SetOutside(kTRUE); return;}
   for (Int_t i=1; i<fLevel+1; i++) nav->CdDown(fArray[i]);
}
