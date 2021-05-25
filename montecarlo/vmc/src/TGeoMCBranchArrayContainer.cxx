// @(#)root/vmc:$Id$
// Authors: Benedikt Volkel 07/03/2019

/*************************************************************************
 * Copyright (C) 2019, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2019, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoMCBranchArrayContainer
    \ingroup vmc

Storing and re-using geometry states of the TGeoManager in use by storing
them as TGeoBranchArrays.

After having initialized a navigator using a stored state, it can be freed
to be used again for storing another geometry state. This makes it easy to
handle many events with many stored geometry states and the memory used is
kept as small as possible.
*/

#include "TGeoMCBranchArrayContainer.h"
#include "TGeoManager.h"
#include "TError.h"

void TGeoMCBranchArrayContainer::Initialize(UInt_t maxLevels, UInt_t size)
{
   fMaxLevels = maxLevels;
   if (fIsInitialized) {
      ResetCache();
   }
   ExtendCache(size);
   fIsInitialized = kTRUE;
}

void TGeoMCBranchArrayContainer::InitializeFromGeoManager(TGeoManager *man, UInt_t size)
{
   Initialize(man->GetMaxLevels(), size);
}

void TGeoMCBranchArrayContainer::ResetCache()
{
   fCache.clear();
   fFreeIndices.clear();
   fIsInitialized = kFALSE;
}

TGeoBranchArray *TGeoMCBranchArrayContainer::GetNewGeoState(UInt_t &userIndex)
{
   if (fFreeIndices.empty()) {
      ExtendCache(2 * fCache.size());
   }
   // Get index from the back
   UInt_t internalIndex = fFreeIndices.back();
   fFreeIndices.pop_back();
   // indices seen by the user are +1
   userIndex = internalIndex + 1;
   fCache[internalIndex]->SetUniqueID(userIndex);
   return fCache[internalIndex].get();
}

const TGeoBranchArray *TGeoMCBranchArrayContainer::GetGeoState(UInt_t userIndex)
{
   if (userIndex == 0) {
      return nullptr;
   }
   if (userIndex > fCache.size()) {
      ::Fatal("TGeoMCBranchArrayContainer::GetGeoState",
              "ID %u is not an index referring to TGeoBranchArray "
              "managed by this TGeoMCBranchArrayContainer",
              userIndex);
   }
   if (fCache[userIndex - 1]->GetUniqueID() == 0) {
      ::Fatal("TGeoMCBranchArrayContainer::GetGeoState", "Passed index %u refers to an empty/unused geo state",
              userIndex);
   }
   return fCache[userIndex - 1].get();
}

void TGeoMCBranchArrayContainer::FreeGeoState(UInt_t userIndex)
{
   if (userIndex > fCache.size() || userIndex == 0) {
      return;
   }
   // Unlock this index so it is free for later use. No need to delete since TGeoBranchArray can be re-used
   if (fCache[userIndex - 1]->GetUniqueID() > 0) {
      fFreeIndices.push_back(userIndex - 1);
      fCache[userIndex - 1]->SetUniqueID(0);
   }
}

void TGeoMCBranchArrayContainer::FreeGeoState(const TGeoBranchArray *geoState)
{
   if (geoState) {
      FreeGeoState(geoState->GetUniqueID());
   }
}

void TGeoMCBranchArrayContainer::FreeGeoStates()
{
   // Start counting at 1 since that is the index seen by the user which is assumed by
   // TGeoMCBranchArrayContainer::FreeGeoState(UInt_t userIndex)
   for (UInt_t i = 0; i < fCache.size(); i++) {
      FreeGeoState(i + 1);
   }
}

void TGeoMCBranchArrayContainer::ExtendCache(UInt_t targetSize)
{
   if (targetSize <= fCache.size()) {
      targetSize = 2 * fCache.size();
   }
   fFreeIndices.reserve(targetSize);
   fCache.reserve(targetSize);
   for (UInt_t i = fCache.size(); i < targetSize; i++) {
      fCache.emplace_back(TGeoBranchArray::MakeInstance(fMaxLevels));
      fCache.back()->SetUniqueID(0);
      fFreeIndices.push_back(i);
   }
}
