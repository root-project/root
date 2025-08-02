/// \file RNTupleReader.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleWriter.hxx>

#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleImtTaskScheduler.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TFile.h>
#include <TROOT.h>

#include <utility>

ROOT::RNTupleWriter::RNTupleWriter(std::unique_ptr<ROOT::RNTupleModel> model,
                                   std::unique_ptr<ROOT::Internal::RPageSink> sink)
   : fFillContext(std::move(model), std::move(sink)), fMetrics("RNTupleWriter")
{
#ifdef R__USE_IMT
   if (IsImplicitMTEnabled() &&
       fFillContext.fSink->GetWriteOptions().GetUseImplicitMT() == ROOT::RNTupleWriteOptions::EImplicitMT::kDefault) {
      fZipTasks = std::make_unique<ROOT::Experimental::Internal::RNTupleImtTaskScheduler>();
      fFillContext.fSink->SetTaskScheduler(fZipTasks.get());
   }
#endif
   // Observe directly the sink's metrics to avoid an additional prefix from the fill context.
   fMetrics.ObserveMetrics(fFillContext.fSink->GetMetrics());
}

ROOT::RNTupleWriter::~RNTupleWriter()
{
   try {
      CommitDataset();
   } catch (const RException &err) {
      R__LOG_ERROR(ROOT::Internal::NTupleLog()) << "failure committing ntuple: " << err.GetError().GetReport();
   }
}

std::unique_ptr<ROOT::RNTupleWriter> ROOT::RNTupleWriter::Create(std::unique_ptr<ROOT::RNTupleModel> model,
                                                                 std::unique_ptr<Internal::RPageSink> sink,
                                                                 const ROOT::RNTupleWriteOptions &options)
{
   if (model->GetRegisteredSubfieldNames().size() > 0) {
      throw RException(R__FAIL("cannot create an RNTupleWriter from a model with registered subfields"));
   }
   for (const auto &field : model->GetConstFieldZero()) {
      if (field.GetTraits() & ROOT::RFieldBase::kTraitEmulatedField)
         throw RException(
            R__FAIL("creating a RNTupleWriter from a model containing emulated fields is currently unsupported."));
   }
   if (options.GetUseBufferedWrite()) {
      sink = std::make_unique<Internal::RPageSinkBuf>(std::move(sink));
   }
   return std::unique_ptr<RNTupleWriter>(new RNTupleWriter(std::move(model), std::move(sink)));
}

std::unique_ptr<ROOT::RNTupleWriter>
ROOT::RNTupleWriter::Recreate(std::unique_ptr<ROOT::RNTupleModel> model, std::string_view ntupleName,
                              std::string_view storage, const ROOT::RNTupleWriteOptions &options)
{
   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   return Create(std::move(model), std::move(sink), options);
}

std::unique_ptr<ROOT::RNTupleWriter>
ROOT::RNTupleWriter::Recreate(std::initializer_list<std::pair<std::string_view, std::string_view>> fields,
                              std::string_view ntupleName, std::string_view storage,
                              const ROOT::RNTupleWriteOptions &options)
{
   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   auto model = ROOT::RNTupleModel::Create();
   for (const auto &fieldDesc : fields) {
      std::string typeName(fieldDesc.first);
      std::string fieldName(fieldDesc.second);
      auto field = ROOT::RFieldBase::Create(fieldName, typeName);
      model->AddField(field.Unwrap());
   }
   return Create(std::move(model), std::move(sink), options);
}

std::unique_ptr<ROOT::RNTupleWriter>
ROOT::RNTupleWriter::Append(std::unique_ptr<ROOT::RNTupleModel> model, std::string_view ntupleName,
                            TDirectory &fileOrDirectory, const ROOT::RNTupleWriteOptions &options)
{
   auto file = fileOrDirectory.GetFile();
   if (!file) {
      throw RException(R__FAIL("RNTupleWriter only supports writing to a ROOT file. Cannot write into a directory "
                               "that is not backed by a file"));
   }
   if (!file->IsBinary()) {
      throw RException(R__FAIL("RNTupleWriter only supports writing to a ROOT file. Cannot write into " +
                               std::string(file->GetName())));
   }

   auto sink = std::make_unique<Internal::RPageSinkFile>(ntupleName, fileOrDirectory, options);
   return Create(std::move(model), std::move(sink), options);
}

void ROOT::RNTupleWriter::CommitClusterGroup()
{
   if (GetNEntries() == fLastCommittedClusterGroup)
      return;
   fFillContext.fSink->CommitClusterGroup();
   fLastCommittedClusterGroup = GetNEntries();
}

ROOT::RNTupleModel &ROOT::RNTupleWriter::GetUpdatableModel()
{
   if (fFillContext.fModel->IsExpired()) {
      throw RException(R__FAIL("invalid attempt to update expired model"));
   }
   return *fFillContext.fModel;
}

void ROOT::RNTupleWriter::CommitDataset()
{
   if (fFillContext.GetModel().IsExpired())
      return;

   CommitCluster(true /* commitClusterGroup */);
   fFillContext.fSink->CommitDataset();
   fFillContext.fModel->Expire();
}

std::unique_ptr<ROOT::RNTupleWriter>
ROOT::Internal::CreateRNTupleWriter(std::unique_ptr<ROOT::RNTupleModel> model,
                                    std::unique_ptr<ROOT::Internal::RPageSink> sink)
{
   return std::unique_ptr<ROOT::RNTupleWriter>(new ROOT::RNTupleWriter(std::move(model), std::move(sink)));
}
