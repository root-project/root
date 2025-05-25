/// \file ROOT/RNTupleReadOptions.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleReadOptions
#define ROOT_RNTupleReadOptions

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
\brief Common user-tunable settings for reading RNTuples

All page source classes need to support the common options.

<table>
<tr>
<th>Option name</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>

<tr>
<td>`ClusterCache`</td>
<td>EClusterCache</td>
<td>EClusterCache::kDefault</td>
<td>
Controls if the prefetcher (including the prefetcher thread) is used
</td>
</tr>

<tr>
<td>`UseImplicitMT`</td>
<td>EImplicitMT</td>
<td>EImplicitMT::kDefault</td>
<td>
Allows to disable parallel page compression and decompression even if ROOT uses implicit MT.
This is useful, e.g., in the context of RDataFrame where the threads are fully managed by RDataFrame.
</td>
</tr>

<tr>
<td>`EnableMetrics`</td>
<td>`bool`</td>
<td>`false`</td>
<td>
If `true`, the RNTupleReader will track metrics straight from its construction, as
if calling RNTupleReader::EnableMetrics() before having created the object.
</td>
</tr>
</table>
*/
// clang-format on
class RNTupleReadOptions {
   friend class Internal::RNTupleReadOptionsManip;

public:
   enum class EClusterCache {
      kOff,
      kOn,
      kDefault = kOn,
   };

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
} // namespace ROOT

#endif
