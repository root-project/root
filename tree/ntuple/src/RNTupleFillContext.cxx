/// \file RNTupleFillContext.cxx
/// \ingroup NTuple
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
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RPageStorage.hxx>

#include <algorithm>
#include <utility>

ROOT::Experimental::RNTupleFillContext::RNTupleFillContext(std::unique_ptr<ROOT::RNTupleModel> model,
                                                           std::unique_ptr<ROOT::Internal::RPageSink> sink)
   : fSink(std::move(sink)), fModel(std::move(model)), fMetrics("RNTupleFillContext")
{
   fModel->Freeze();
   fSink->Init(*fModel);
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
      FlushCluster();
   } catch (const RException &err) {
      R__LOG_ERROR(ROOT::Internal::NTupleLog()) << "failure flushing cluster: " << err.GetError().GetReport();
   }

   if (!fStagedClusters.empty()) {
      R__LOG_ERROR(ROOT::Internal::NTupleLog())
         << std::to_string(fStagedClusters.size()) << " staged clusters still pending, their data is lost";
   }
}

void ROOT::Experimental::RNTupleFillContext::FlushColumns()
{
   for (auto &field : ROOT::Internal::GetFieldZeroOfModel(*fModel)) {
      ROOT::Internal::CallFlushColumnsOnField(field);
   }
}

void ROOT::Experimental::RNTupleFillContext::FlushCluster()
{
   if (fNEntries == fLastFlushed) {
      return;
   }
   for (auto &field : ROOT::Internal::GetFieldZeroOfModel(*fModel)) {
      ROOT::Internal::CallCommitClusterOnField(field);
   }
   auto nEntriesInCluster = fNEntries - fLastFlushed;
   if (fStagedClusterCommitting) {
      auto stagedCluster = fSink->StageCluster(nEntriesInCluster);
      fNBytesFlushed += stagedCluster.fNBytesWritten;
      fStagedClusters.push_back(std::move(stagedCluster));
   } else {
      fNBytesFlushed += fSink->CommitCluster(nEntriesInCluster);
   }
   fNBytesFilled += fUnzippedClusterSize;

   // Cap the compression factor at 1000 to prevent overflow of fUnzippedClusterSizeEst
   const float compressionFactor =
      std::min(1000.f, static_cast<float>(fNBytesFilled) / static_cast<float>(fNBytesFlushed));
   fUnzippedClusterSizeEst =
      compressionFactor * static_cast<float>(fSink->GetWriteOptions().GetApproxZippedClusterSize());

   fLastFlushed = fNEntries;
   fUnzippedClusterSize = 0;
}

void ROOT::Experimental::RNTupleFillContext::CommitStagedClusters()
{
   if (fStagedClusters.empty()) {
      return;
   }
   if (fModel->IsExpired()) {
      throw RException(R__FAIL("invalid attempt to commit staged clusters after dataset was committed"));
   }

   fSink->CommitStagedClusters(fStagedClusters);
   fStagedClusters.clear();
}
