/// \file RNTupleFillContext.cxx
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

#include <ROOT/RNTupleFillContext.hxx>

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RPageStorage.hxx>

#include <algorithm>
#include <utility>

ROOT::Experimental::RNTupleFillContext::RNTupleFillContext(std::unique_ptr<RNTupleModel> model,
                                                           std::unique_ptr<Internal::RPageSink> sink)
   : fSink(std::move(sink)), fModel(std::move(model)), fMetrics("RNTupleFillContext")
{
   fModel->Freeze();
   fSink->Init(*fModel.get());
   fMetrics.ObserveMetrics(fSink->GetMetrics());

   const auto &writeOpts = fSink->GetWriteOptions();
   fMaxUnzippedClusterSize = writeOpts.GetMaxUnzippedClusterSize();
   // First estimate is a factor 2 compression if compression is used at all
   const int scale = writeOpts.GetCompression() ? 2 : 1;
   fUnzippedClusterSizeEst = scale * writeOpts.GetApproxZippedClusterSize();
}

ROOT::Experimental::RNTupleFillContext::~RNTupleFillContext()
{
   try {
      CommitCluster();
   } catch (const RException &err) {
      R__LOG_ERROR(NTupleLog()) << "failure committing ntuple: " << err.GetError().GetReport();
   }
}

void ROOT::Experimental::RNTupleFillContext::CommitCluster()
{
   if (fNEntries == fLastCommitted) {
      return;
   }
   if (fSink->GetWriteOptions().GetHasSmallClusters() &&
       (fUnzippedClusterSize > RNTupleWriteOptions::kMaxSmallClusterSize)) {
      throw RException(R__FAIL("invalid attempt to write a cluster > 512MiB with 'small clusters' option enabled"));
   }
   for (auto &field : fModel->GetFieldZero()) {
      Internal::CallCommitClusterOnField(field);
   }
   auto nEntriesInCluster = fNEntries - fLastCommitted;
   fNBytesCommitted += fSink->CommitCluster(nEntriesInCluster);
   fNBytesFilled += fUnzippedClusterSize;

   // Cap the compression factor at 1000 to prevent overflow of fUnzippedClusterSizeEst
   const float compressionFactor =
      std::min(1000.f, static_cast<float>(fNBytesFilled) / static_cast<float>(fNBytesCommitted));
   fUnzippedClusterSizeEst =
      compressionFactor * static_cast<float>(fSink->GetWriteOptions().GetApproxZippedClusterSize());

   fLastCommitted = fNEntries;
   fUnzippedClusterSize = 0;
}
