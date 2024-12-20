/// \file ROOT/RNTupleFillStatus.hxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-04-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleFillStatus
#define ROOT7_RNTupleFillStatus

#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleFillStatus
\ingroup NTuple
\brief A status object after filling an entry

After passing an instance to RNTupleWriter::FillNoFlush or RNTupleFillContext::FillNoFlush, the caller must check
ShouldFlushCluster and call RNTupleWriter::FlushCluster or RNTupleFillContext::FlushCluster if necessary.
*/
// clang-format on
class RNTupleFillStatus {
   friend class RNTupleFillContext;

private:
   /// Number of entries written into the current cluster
   NTupleSize_t fNEntriesSinceLastFlush = 0;
   /// Number of bytes written into the current cluster
   std::size_t fUnzippedClusterSize = 0;
   /// Number of bytes written for the last entry
   std::size_t fLastEntrySize = 0;
   bool fShouldFlushCluster = false;

public:
   /// Return the number of entries written into the current cluster.
   NTupleSize_t GetNEntries() const { return fNEntriesSinceLastFlush; }
   /// Return the number of bytes written into the current cluster.
   std::size_t GetUnzippedClusterSize() const { return fUnzippedClusterSize; }
   /// Return the number of bytes for the last entry.
   std::size_t GetLastEntrySize() const { return fLastEntrySize; }
   /// Return true if the caller should call FlushCluster.
   bool ShouldFlushCluster() const { return fShouldFlushCluster; }
}; // class RNTupleFillContext

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleFillStatus
