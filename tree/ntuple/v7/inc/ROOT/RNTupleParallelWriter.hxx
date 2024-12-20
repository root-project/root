/// \file ROOT/RNTupleParallelWriter.hxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-02-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleParallelWriter
#define ROOT7_RNTupleParallelWriter

#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

class TDirectory;

namespace ROOT {
namespace Experimental {

namespace Internal {
class RPageSink;
} // namespace Internal

class RNTupleFillContext;
class RNTupleModel;

/**
\class ROOT::Experimental::RNTupleParallelWriter
\ingroup NTuple
\brief A writer to fill an RNTuple from multiple contexts

Compared to the sequential RNTupleWriter, a parallel writer enables the creation of multiple RNTupleFillContext (see
RNTupleParallelWriter::CreateFillContext).  Each fill context prepares independent clusters that are appended to the
common ntuple with internal synchronization.  Before destruction, all fill contexts must have flushed their data and
been destroyed (or data could be lost!).

For user convenience, RNTupleParallelWriter::CreateFillContext is thread-safe and may be called from multiple threads
in parallel at any time, also after some data has already been written.  Internally, the original model is cloned and
ownership is passed to a newly created RNTupleFillContext.  For that reason, it is recommended to use
RNTupleModel::CreateBare when creating the model for parallel writing and avoid the allocation of a useless default
REntry per context.

Note that the sequence of independently prepared clusters is indeterminate and therefore entries are only partially
ordered:  Entries from one context are totally ordered as they were filled.  However, there is no orderering with other
contexts and the entries may be appended to the ntuple either before or after other entries written in parallel into
other contexts.  In addition, two consecutive entries in one fill context can end up separated in the final ntuple, if
they happen to fall onto a cluster boundary and other contexts append more entries before the next cluster is full.

At the moment, the parallel writer does not (yet) support incremental updates of the underlying model. Please refer to
RNTupleWriter::CreateModelUpdater if required for your use case.
*/
class RNTupleParallelWriter {
private:
   /// A global mutex to protect the internal data structures of this object.
   std::mutex fMutex;
   /// A mutex to synchronize the final page sink.
   std::mutex fSinkMutex;
   /// The final RPageSink that represents the synchronization point.
   std::unique_ptr<Internal::RPageSink> fSink;
   /// The original RNTupleModel connected to fSink; needs to be destructed before it.
   std::unique_ptr<RNTupleModel> fModel;
   Detail::RNTupleMetrics fMetrics;
   /// List of all created helpers. They must be destroyed before this RNTupleParallelWriter is destructed.
   std::vector<std::weak_ptr<RNTupleFillContext>> fFillContexts;

   RNTupleParallelWriter(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);
   RNTupleParallelWriter(const RNTupleParallelWriter &) = delete;
   RNTupleParallelWriter &operator=(const RNTupleParallelWriter &) = delete;

public:
   /// Recreate a new file and return a writer to write an ntuple.
   static std::unique_ptr<RNTupleParallelWriter> Recreate(std::unique_ptr<RNTupleModel> model,
                                                          std::string_view ntupleName, std::string_view storage,
                                                          const RNTupleWriteOptions &options = RNTupleWriteOptions());
   /// Append an ntuple to the existing file, which must not be accessed while data is filled into any created context.
   static std::unique_ptr<RNTupleParallelWriter> Append(std::unique_ptr<RNTupleModel> model,
                                                        std::string_view ntupleName, TDirectory &fileOrDirectory,
                                                        const RNTupleWriteOptions &options = RNTupleWriteOptions());

   ~RNTupleParallelWriter();

   /// Create a new RNTupleFillContext that can be used to fill entries and prepare clusters in parallel. This method is
   /// thread-safe and may be called from multiple threads in parallel at any time, also after some data has already
   /// been written.
   ///
   /// Note that all fill contexts must be destroyed before RNTupleParallelWriter::CommitDataset() is called.
   std::shared_ptr<RNTupleFillContext> CreateFillContext();

   /// Automatically called by the destructor
   void CommitDataset();

   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
};

} // namespace Experimental
} // namespace ROOT

#endif
