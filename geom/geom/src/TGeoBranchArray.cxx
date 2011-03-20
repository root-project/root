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
#include "TString.h"
#include "TGeoMatrix.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"

ClassImp(TGeoBranchArray)

//______________________________________________________________________________
TGeoBranchArray::TGeoBranchArray(UShort_t level)
                :fLevel(level),
                 fArray(NULL),
                 fMatrix(NULL),
                 fClient(NULL)
{
// Constructor. Alocates the array with a size given by level.
   fArray = new UShort_t[level];
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
                 fArray(NULL),
                 fMatrix(NULL)
{
// Copy constructor.
   if (fLevel) fArray = new UShort_t[fLevel];
   if (other.fMatrix) fMatrix = new TGeoHMatrix(*(other.fMatrix));
}   
      
//______________________________________________________________________________
TGeoBranchArray& TGeoBranchArray::operator=(const TGeoBranchArray& other)
{
// Assignment.
   if (&other == this) return *this;
   fLevel = other.fLevel;
   if (fLevel) fArray = new UShort_t[fLevel];
   if (other.fMatrix) {
      fMatrix = new TGeoHMatrix();
      fMatrix->CopyFrom(other.fMatrix);
   }
   return *this;
}   

//______________________________________________________________________________
void TGeoBranchArray::AddLevel(UShort_t dindex)
{
// Add and extra daughter to the current path array. No validity check performed !
   if (!fLevel) {
      Error("AddLevel", "You must initialize from navigator or copy from another branch array first.");
      return;
   }
   fLevel++;
   UShort_t *array = new UShort_t[fLevel];
   memcpy(array, fArray, (fLevel-1)*sizeof(UShort_t));
   array[fLevel-1] = dindex;
   delete [] fArray;
   fArray = array;
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
   UShort_t i;
   TGeoBranchArray *other = (TGeoBranchArray*)obj;
   UShort_t otherLevel = other->GetLevel();
   UShort_t maxLevel = TMath::Min(fLevel, otherLevel);
   UShort_t *otherArray = other->GetArray();
   for (i=0; i<maxLevel; i++) {
      if (fArray[i]==otherArray[i]) continue;
      if (fArray[i]<otherArray[i]) return -1;
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
TGeoNode *TGeoBranchArray::GetNode(UShort_t level) const
{
   TGeoNode *node = gGeoManager->GetTopNode();
   if (!level) return node;
   if (level>fLevel) return NULL;
   for (Int_t i=0; i<level; i++) node = node->GetVolume()->GetNode(fArray[i]);
   return node;
}
   
//______________________________________________________________________________
void TGeoBranchArray::InitFromNavigator(TGeoNavigator *nav)
{
// Init the branch array from current navigator state.
   UShort_t level = (UShort_t)nav->GetLevel();
   if (!fMatrix) fMatrix = new TGeoHMatrix();
   fMatrix->CopyFrom(nav->GetCurrentMatrix());
   if (!level) {
//      delete [] fArray; fArray = 0;
      fLevel = 0;
      return;
   }
   if (!fArray || level>fLevel) {
      delete [] fArray; 
      fArray = new UShort_t[level];
   }
   fLevel = level;
   TGeoNode *mother = nav->GetMother(fLevel);
   for (Int_t i=fLevel-1; i>=0; i--) {
      TGeoNode *node = nav->GetMother(i);
      Int_t index = mother->GetVolume()->GetIndex(node);
      fArray[fLevel-i-1] = index;
      mother = node;
   }   
}

//______________________________________________________________________________
void TGeoBranchArray::Print(Option_t *) const
{
// Print branch information
   TString path = "/";
   TGeoNode *node = GetNode(0);
   path += node->GetName();
   for (Int_t i=0; i<fLevel; i++) {
      path += "/";
      node = node->GetVolume()->GetNode(fArray[i]);
      path += node->GetName();
   }
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
   for (Int_t i=0; i<fLevel; i++) nav->CdDown(fArray[i]);
}
