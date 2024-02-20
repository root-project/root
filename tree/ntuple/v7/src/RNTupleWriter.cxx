/// \file RNTupleReader.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleWriter.hxx>

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleImtTaskScheduler.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TROOT.h>

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

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleWriter::RNTupleWriter(std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
                                                 std::unique_ptr<ROOT::Experimental::Internal::RPageSink> sink)
   : fFillContext(std::move(model), std::move(sink)), fMetrics("RNTupleWriter")
{
#ifdef R__USE_IMT
   if (IsImplicitMTEnabled()) {
      fZipTasks = std::make_unique<Internal::RNTupleImtTaskScheduler>();
      fFillContext.fSink->SetTaskScheduler(fZipTasks.get());
   }
#endif
   // Observe directly the sink's metrics to avoid an additional prefix from the fill context.
   fMetrics.ObserveMetrics(fFillContext.fSink->GetMetrics());
}

ROOT::Experimental::RNTupleWriter::~RNTupleWriter()
{
   try {
      CommitCluster(true /* commitClusterGroup */);
      fFillContext.fSink->CommitDataset();
   } catch (const RException &err) {
      R__LOG_ERROR(NTupleLog()) << "failure committing ntuple: " << err.GetError().GetReport();
   }
}

std::unique_ptr<ROOT::Experimental::RNTupleWriter>
ROOT::Experimental::RNTupleWriter::Create(std::unique_ptr<RNTupleModel> model,
                                          std::unique_ptr<Internal::RPageSink> sink, const RNTupleWriteOptions &options)
{
   if (options.GetUseBufferedWrite()) {
      sink = std::make_unique<Internal::RPageSinkBuf>(std::move(sink));
   }
   return std::unique_ptr<RNTupleWriter>(new RNTupleWriter(std::move(model), std::move(sink)));
}

std::unique_ptr<ROOT::Experimental::RNTupleWriter>
ROOT::Experimental::RNTupleWriter::Recreate(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                            std::string_view storage, const RNTupleWriteOptions &options)
{
   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   return Create(std::move(model), std::move(sink), options);
}

std::unique_ptr<ROOT::Experimental::RNTupleWriter>
ROOT::Experimental::RNTupleWriter::Recreate(std::initializer_list<std::pair<std::string_view, std::string_view>> fields,
                                            std::string_view ntupleName, std::string_view storage,
                                            const RNTupleWriteOptions &options)
{
   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   auto model = RNTupleModel::Create();
   for (const auto &fieldDesc : fields) {
      std::string typeName(fieldDesc.first);
      std::string fieldName(fieldDesc.second);
      auto field = RFieldBase::Create(fieldName, typeName);
      model->AddField(field.Unwrap());
   }
   return Create(std::move(model), std::move(sink), options);
}

std::unique_ptr<ROOT::Experimental::RNTupleWriter>
ROOT::Experimental::RNTupleWriter::Append(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName, TFile &file,
                                          const RNTupleWriteOptions &options)
{
   auto sink = std::make_unique<Internal::RPageSinkFile>(ntupleName, file, options);
   return Create(std::move(model), std::move(sink), options);
}

void ROOT::Experimental::RNTupleWriter::CommitClusterGroup()
{
   if (GetNEntries() == fLastCommittedClusterGroup)
      return;
   fFillContext.fSink->CommitClusterGroup();
   fLastCommittedClusterGroup = GetNEntries();
}

std::unique_ptr<ROOT::Experimental::RNTupleWriter>
ROOT::Experimental::Internal::CreateRNTupleWriter(std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
                                                  std::unique_ptr<ROOT::Experimental::Internal::RPageSink> sink)
{
   return std::unique_ptr<ROOT::Experimental::RNTupleWriter>(
      new ROOT::Experimental::RNTupleWriter(std::move(model), std::move(sink)));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RCollectionNTupleWriter::RCollectionNTupleWriter(std::unique_ptr<REntry> defaultEntry)
   : fOffset(0), fDefaultEntry(std::move(defaultEntry))
{
}
