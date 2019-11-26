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

// Class TGeoMCBranchArrayContainer
// ---------------------
// cache for storing TGeoBranchArray objects
//

#ifndef ROOT_TGeoMCBranchArrayContainer
#define ROOT_TGeoMCBranchArrayContainer

#include <vector>
#include <memory>

#include "TGeoBranchArray.h"

class TGeoManager;

class TGeoMCBranchArrayContainer {
public:
   /// Default constructor
   TGeoMCBranchArrayContainer() = default;
   /// Destructor
   ~TGeoMCBranchArrayContainer() = default;

   /// Initialize manually specifying initial number of internal
   /// TGeoBranchArray objects
   void Initialize(UInt_t maxlevels = 100, UInt_t size = 8);
   /// Initialize from TGeoManager to extract maxlevels
   void InitializeFromGeoManager(TGeoManager *man, UInt_t size = 8);
   /// Clear the internal cache
   void ResetCache();

   /// Get a TGeoBranchArray to set to current geo state.
   TGeoBranchArray *GetNewGeoState(UInt_t &userIndex);
   /// Get a TGeoBranchArray to read the current state from.
   const TGeoBranchArray *GetGeoState(UInt_t userIndex);
   /// Free the index of this geo state such that it can be re-used
   void FreeGeoState(UInt_t userIndex);
   /// Free the index of this geo state such that it can be re-used
   void FreeGeoState(const TGeoBranchArray *geoState);
   /// Free all geo states at once but keep the container size
   void FreeGeoStates();

private:
   /// Copying kept private
   TGeoMCBranchArrayContainer(const TGeoMCBranchArrayContainer &);
   /// Assignement kept private
   TGeoMCBranchArrayContainer &operator=(const TGeoMCBranchArrayContainer &);
   /// Resize the cache
   void ExtendCache(UInt_t targetSize = 1);

private:
   /// Cache states via TGeoBranchArray
   std::vector<std::unique_ptr<TGeoBranchArray>> fCache;
   /// Maximum level of node array inside a chached state.
   UInt_t fMaxLevels = 100;
   /// Provide indices in fCachedStates which are already popped and can be
   /// re-populated again.
   std::vector<UInt_t> fFreeIndices;
   /// Flag if initialized
   Bool_t fIsInitialized = kFALSE;

   ClassDefNV(TGeoMCBranchArrayContainer, 1)
};

#endif /* ROOT_TGeoMCBranchArrayContainer */
