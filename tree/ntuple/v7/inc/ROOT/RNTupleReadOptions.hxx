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

class RNTupleReadOptions;

namespace Internal {

class RNTupleReadOptionsManip final {
public:
   static unsigned int GetClusterBunchSize(const RNTupleReadOptions &options);
   static void SetClusterBunchSize(RNTupleReadOptions &options, unsigned int val);
};

} // namespace Internal

// clang-format off
/**
\class ROOT::RNTupleReadOptions
\ingroup NTuple
\brief Common user-tunable settings for reading ntuples

All page source classes need to support the common options.
*/
// clang-format on
class RNTupleReadOptions {
   friend class Internal::RNTupleReadOptionsManip;

public:
   /// Controls if the prefetcher (including the prefetcher thread) is used
   enum class EClusterCache {
      kOff,
      kOn,
      kDefault = kOn,
   };

   /// Allows to disable parallel page compression and decompression even if ROOT uses implicit MT.
   /// This is useful, e.g., in the context of RDataFrame where the threads are fully managed by RDataFrame.
   enum class EImplicitMT {
      kOff,
      kDefault,
   };

private:
   EClusterCache fClusterCache = EClusterCache::kDefault;
   /// The number of cluster to be prefetched in a single batch; this option is transitional and will be replaced
   /// by an option that allows to control the amount of memory that the prefetcher uses.
   unsigned int fClusterBunchSize = 1;
   EImplicitMT fUseImplicitMT = EImplicitMT::kDefault;
   /// If true, the RNTupleReader will track metrics straight from its construction, as
   /// if calling `RNTupleReader::EnableMetrics()` before having created the object.
   bool fEnableMetrics = false;

public:
   EClusterCache GetClusterCache() const { return fClusterCache; }
   void SetClusterCache(EClusterCache val) { fClusterCache = val; }

   EImplicitMT GetUseImplicitMT() const { return fUseImplicitMT; }
   void SetUseImplicitMT(EImplicitMT val) { fUseImplicitMT = val; }

   bool GetEnableMetrics() const { return fEnableMetrics; }
   void SetEnableMetrics(bool val) { fEnableMetrics = val; }
}; // class RNTupleReadOptions

namespace Internal {

inline unsigned int RNTupleReadOptionsManip::GetClusterBunchSize(const RNTupleReadOptions &options)
{
   return options.fClusterBunchSize;
}

inline void RNTupleReadOptionsManip::SetClusterBunchSize(RNTupleReadOptions &options, unsigned int val)
{
   options.fClusterBunchSize = val;
}

} // namespace Internal

namespace Experimental {
// TODO(gparolini): remove before branching ROOT v6.36
using RNTupleReadOptions [[deprecated("ROOT::Experimental::RNTupleReadOptions moved to ROOT::RNTupleReadOptions")]] =
   ROOT::RNTupleReadOptions;
} // namespace Experimental

} // namespace ROOT

#endif
