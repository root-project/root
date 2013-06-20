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

//______________________________________________________________________________
TGeoBranchArray::TGeoBranchArray(Int_t level)
                :fLevel(level),
                 fMaxLevel(0),
                 fArray(NULL),
                 fMatrix(NULL),
                 fClient(NULL)
{
// Constructor. Alocates the array with a size given by level.
   fMaxLevel = (fLevel+1 > 10) ? fLevel+1:10;
   fArray = new TGeoNode*[fMaxLevel];
}

//______________________________________________________________________________
TGeoBranchArray::~TGeoBranchArray()
{
// Destructor.
   delete [] fArray;
   delete fMatrix;
}

//______________________________________________________________________________
TGeoBranchArray::TGeoBranchArray(const TGeoBranchArray&  other)
                :TObject(other),
                 fLevel(other.fLevel),
                 fMaxLevel(other.fMaxLevel),
                 fArray(NULL),
                 fMatrix(NULL),
                 fClient(other.fClient)
{
// Copy constructor.
   if (fMaxLevel) {
      fArray = new TGeoNode*[fMaxLevel];
      if (fLevel+1) memcpy(fArray, other.fArray, (fLevel+1)*sizeof(TGeoNode*));
   }
   if (other.fMatrix) fMatrix = new TGeoHMatrix(*(other.fMatrix));   
}   
      
//______________________________________________________________________________
TGeoBranchArray& TGeoBranchArray::operator=(const TGeoBranchArray& other)
{
// Assignment.
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
      fMatrix = new TGeoHMatrix();
      fMatrix->CopyFrom(other.fMatrix);
   }
   fClient = other.fClient;
//   TThread::UnLock();
   return *this;
}   

//______________________________________________________________________________
void TGeoBranchArray::AddLevel(Int_t dindex)
{
// Add and extra daughter to the current path array. No validity check performed !
   if (!fLevel) {
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

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator ==(const TGeoBranchArray& other) const
{
// Is equal operator.
   Int_t value = Compare(&other);
   if (value==0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator !=(const TGeoBranchArray& other) const
{
// Not equal operator.
   Int_t value = Compare(&other);
   if (value!=0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator >(const TGeoBranchArray& other) const
{
// Is equal operator.
   Int_t value = Compare(&other);
   if (value>0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator <(const TGeoBranchArray& other) const
{
// Is equal operator.
   Int_t value = Compare(&other);
   if (value<0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator >=(const TGeoBranchArray& other) const
{
// Is equal operator.
   Int_t value = Compare(&other);
   if (value>=0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoBranchArray::operator <=(const TGeoBranchArray& other) const
{
// Is equal operator.
   Int_t value = Compare(&other);
   if (value<=0) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
Long64_t TGeoBranchArray::BinarySearch(Long64_t n, const TGeoBranchArray **array, TGeoBranchArray *value)
{
// Binary search in an array of n pointers to branch arrays, to locate value.
// Returns element index or index of nearest element smaller than value
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

//______________________________________________________________________________
Int_t TGeoBranchArray::Compare(const TObject *obj) const
{
// Compare with other object of same type. Returns -1 if this is smaller (first
// smaller array value prevails), 0 if equal (size and values) and 1 if this is 
// larger.
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

//______________________________________________________________________________
void TGeoBranchArray::CleanMatrix()
{
// Garbage collect the stored matrix.
   delete fMatrix; fMatrix = 0;
}

//______________________________________________________________________________
void TGeoBranchArray::Init(TGeoNode **branch, TGeoMatrix *global, Int_t level)
{
// Init the branch array from an array of nodes, the global matrix for the path and 
// the level.
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
   
//______________________________________________________________________________
void TGeoBranchArray::InitFromNavigator(TGeoNavigator *nav)
{
// Init the branch array from current navigator state.
   TGeoNodeCache *cache = nav->GetCache();
   const TGeoNode **branch = (const TGeoNode**)cache->GetBranch();
   Int_t level = cache->GetLevel();
   if (!fMatrix) fMatrix = new TGeoHMatrix();
   fMatrix->CopyFrom(cache->GetCurrentMatrix());
//   TThread::Lock();
   if (!fArray || level+1>fMaxLevel) {
      delete [] fArray; 
      fMaxLevel = level+1;
      fArray = new TGeoNode*[fMaxLevel];
   }
   fLevel = level;
   memcpy(fArray, branch, (fLevel+1)*sizeof(TGeoNode*));
//   TThread::UnLock();
}

//______________________________________________________________________________
void TGeoBranchArray::GetPath(TString &path) const
{
// Fill path pointed by the array.
   path = "";
   if (!fArray) return;
   for (Int_t i=0; i<fLevel+1; i++) {
      path += "/";
      path += fArray[i]->GetName();
   }
}   

//______________________________________________________________________________
void TGeoBranchArray::Print(Option_t *) const
{
// Print branch information
   TString path;
   GetPath(path);
   printf("branch:    %s\n", path.Data());
}      

//______________________________________________________________________________
void TGeoBranchArray::Sort(Int_t n, TGeoBranchArray **array, Int_t *index, Bool_t down)
{
// Sorting of an array of branch array pointers.
   for (Int_t i=0; i<n; i++) index[i] = i;
   if (down)
      std::sort(index, index + n, compareBAdesc(array));
   else   
      std::sort(index, index + n, compareBAasc(array));
}      

//______________________________________________________________________________
void TGeoBranchArray::UpdateNavigator(TGeoNavigator *nav) const
{
// Update the navigator to reflect the branch.
   nav->CdTop();
   for (Int_t i=1; i<fLevel+1; i++) nav->CdDown(fArray[i]);
}
