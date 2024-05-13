/// \file ROOT/RNTupleFillContext.hxx
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

#ifndef ROOT7_RNTupleFillContext
#define ROOT7_RNTupleFillContext

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ROOT {
namespace Experimental {

namespace Internal {
class RPageSink;
}

// clang-format off
/**
\class ROOT::Experimental::RNTupleFillContext
\ingroup NTuple
\brief A context for filling entries (data) into clusters of an RNTuple

An output cluster can be filled with entries. The caller has to make sure that the data that gets filled into a cluster
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by CommitCluster() or by destructing the context.  On I/O errors, an exception is thrown.

Instances of this class are not meant to be used in isolation and can be created from an RNTupleParallelWriter. For
sequential writing, please refer to RNTupleWriter.
*/
// clang-format on
class RNTupleFillContext {
   friend class RNTupleWriter;
   friend class RNTupleParallelWriter;

private:
   std::unique_ptr<Internal::RPageSink> fSink;
   /// Needs to be destructed before fSink
   std::unique_ptr<RNTupleModel> fModel;

   Detail::RNTupleMetrics fMetrics;

   NTupleSize_t fLastCommitted = 0;
   NTupleSize_t fNEntries = 0;
   /// Keeps track of the number of bytes written into the current cluster
   std::size_t fUnzippedClusterSize = 0;
   /// The total number of bytes written to storage (i.e., after compression)
   std::uint64_t fNBytesCommitted = 0;
   /// The total number of bytes filled into all the so far committed clusters,
   /// i.e. the uncompressed size of the written clusters
   std::uint64_t fNBytesFilled = 0;
   /// Limit for committing cluster no matter the other tunables
   std::size_t fMaxUnzippedClusterSize;
   /// Estimator of uncompressed cluster size, taking into account the estimated compression ratio
   std::size_t fUnzippedClusterSizeEst;

   RNTupleFillContext(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);
   RNTupleFillContext(const RNTupleFillContext &) = delete;
   RNTupleFillContext &operator=(const RNTupleFillContext &) = delete;

public:
   ~RNTupleFillContext();

   /// Fill an entry into this context.  This method will perform a light check whether the entry comes from the
   /// context's own model.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(REntry &entry)
   {
      if (R__unlikely(entry.GetModelId() != fModel->GetModelId()))
         throw RException(R__FAIL("mismatch between entry and model"));

      const std::size_t bytesWritten = entry.Append();
      fUnzippedClusterSize += bytesWritten;
      fNEntries++;
      if ((fUnzippedClusterSize >= fMaxUnzippedClusterSize) || (fUnzippedClusterSize >= fUnzippedClusterSizeEst))
         CommitCluster();
      return bytesWritten;
   }
   /// Ensure that the data from the so far seen Fill calls has been written to storage
   void CommitCluster();

   std::unique_ptr<REntry> CreateEntry() { return fModel->CreateEntry(); }

   /// Return the entry number that was last committed in a cluster.
   NTupleSize_t GetLastCommitted() const { return fLastCommitted; }
   /// Return the number of entries filled so far.
   NTupleSize_t GetNEntries() const { return fNEntries; }

   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
}; // class RNTupleFillContext

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleFillContext
