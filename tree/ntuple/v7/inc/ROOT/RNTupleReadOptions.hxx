/// \file ROOT/RNTupleReadOptions.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleReadOptions
#define ROOT7_RNTupleReadOptions

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleReadOptions
\ingroup NTuple
\brief Common user-tunable settings for reading ntuples

All page source classes need to support the common options.
*/
// clang-format on
class RNTupleReadOptions {
public:
   enum EClusterCache {
      kOff,
      kOn,
      kDefault = kOn,
   };

private:
   EClusterCache fClusterCache = EClusterCache::kDefault;
   unsigned int fClusterBunchSize = 1;

public:
   EClusterCache GetClusterCache() const { return fClusterCache; }
   void SetClusterCache(EClusterCache val) { fClusterCache = val; }
   unsigned int GetClusterBunchSize() const { return fClusterBunchSize; }
   void SetClusterBunchSize(unsigned int val) { fClusterBunchSize = val; }
};

} // namespace Experimental
} // namespace ROOT

#endif
