/// \file RNTupleImporter.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2022-11-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleImporter.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RStringView.hxx>

#include <TBranch.h>
#include <TLeaf.h>
#include <TLeafC.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>

namespace {

class RDefaultProgressCallback : public ROOT::Experimental::RNTupleImporter::RProgressCallback {
private:
   std::uint64_t fNbytesLast = 0;

public:
   void Call(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      if (nbytesWritten < (fNbytesLast + 50 * 1000 * 1000))
         return;
      std::cout << "Wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries" << std::endl;
      fNbytesLast = nbytesWritten;
   }

   void Finish(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      std::cout << "Done, wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries" << std::endl;
   }
};

} // anonymous namespace

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleImporter>>
ROOT::Experimental::RNTupleImporter::Create(std::string_view sourceFile, std::string_view treeName,
                                            std::string_view destFile)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());
   importer->fNTupleName = treeName;
   importer->fSourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFile).c_str()));
   if (!importer->fSourceFile || importer->fSourceFile->IsZombie()) {
      return R__FAIL("cannot open source file " + std::string(sourceFile));
   }
   importer->fSourceTree = std::unique_ptr<TTree>(importer->fSourceFile->Get<TTree>(std::string(treeName).c_str()));
   if (!importer->fSourceTree) {
      return R__FAIL("cannot read TTree " + std::string(treeName) + " from " + std::string(sourceFile));
   }
   // If we have IMT enabled, its best use is for parallel page compression
   importer->fSourceTree->SetImplicitMT(false);

   importer->fDestFileName = destFile;
   importer->fWriteOptions.SetCompression(importer->fSourceFile->GetCompressionSettings());
   importer->fDestFile = std::unique_ptr<TFile>(TFile::Open(importer->fDestFileName.c_str(), "UPDATE"));
   if (!importer->fDestFile || importer->fDestFile->IsZombie()) {
      return R__FAIL("cannot open dest file " + std::string(importer->fDestFileName));
   }

   return importer;
}

void ROOT::Experimental::RNTupleImporter::ReportSchema()
{
   for (const auto &f : fImportFeatures) {
      std::cout << "Importing '" << f.fLeafName << "'    -->    to field '" << f.fFieldName << "' [" << f.fTypeName
                << ']' << std::endl;
   }
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::PrepareSchema()
{
   fImportFeatures.clear();
   fSourceTree->SetBranchStatus("*", 0);

   for (auto b : TRangeDynCast<TBranch>(*fSourceTree->GetListOfBranches())) {
      assert(b);

      for (auto l : TRangeDynCast<TLeaf>(b->GetListOfLeaves())) {
         RImportFeature f;
         f.fBranchName = b->GetName();
         f.fLeafName = l->GetName();
         f.fFieldName = l->GetName();
         if (l->IsA() == TLeafC::Class()) {
            f.fTypeName = "std::string";
            f.fTreeBuffer = std::make_unique<unsigned char[]>(l->GetMaximum());
            fFeatureCStringIndexes.emplace_back(fImportFeatures.size());
         } else {
            f.fTypeName = l->GetTypeName();
         }
         fImportFeatures.emplace_back(std::move(f));
      }
   }

   fModel = RNTupleModel::Create();
   fModel->SetDescription(fSourceTree->GetTitle());
   for (const auto &f : fImportFeatures) {
      auto field = Detail::RFieldBase::Create(f.fFieldName, f.fTypeName);
      if (!field)
         return R__FORWARD_ERROR(field);
      fModel->AddField(std::move(field.Unwrap()));
   }

   fModel->Freeze();

   for (auto &f : fImportFeatures) {
      // We connect the model's default entry's memory location for the new field to the branch, so that we can
      // fill the ntuple with the data read from the TTree
      fSourceTree->SetBranchStatus(f.fBranchName.c_str(), 1);
      f.fFieldDataPtr = fModel->GetDefaultEntry()->GetValue(f.fFieldName).GetRawPtr();
      if (f.fTreeBuffer) {
         fSourceTree->SetBranchAddress(f.fLeafName.c_str(), reinterpret_cast<void *>(f.fTreeBuffer.get()));
      } else {
         fSourceTree->SetBranchAddress(f.fLeafName.c_str(), f.fFieldDataPtr);
      }
   }

   if (!fIsQuiet)
      ReportSchema();

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::Import()
{
   if (fDestFile->FindKey(fNTupleName.c_str()) != nullptr)
      return R__FAIL("Key '" + fNTupleName + "' already exists in file " + fDestFileName);

   PrepareSchema();

   auto sink = std::make_unique<Detail::RPageSinkFile>(fNTupleName, *fDestFile, fWriteOptions);
   sink->GetMetrics().Enable();
   auto ctrZippedBytes = sink->GetMetrics().GetCounter("RPageSinkFile.szWritePayload");

   auto ntplWriter = std::make_unique<RNTupleWriter>(std::move(fModel), std::move(sink));
   fModel = nullptr;

   fProgressCallback = fIsQuiet ? nullptr : std::make_unique<RDefaultProgressCallback>();
   auto nEntries = fSourceTree->GetEntries();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      fSourceTree->GetEntry(i);

      for (auto idx : fFeatureCStringIndexes) {
         *reinterpret_cast<std::string *>(fImportFeatures[idx].fFieldDataPtr) =
            reinterpret_cast<char *>(fImportFeatures[idx].fTreeBuffer.get());
      }

      ntplWriter->Fill();

      if (fProgressCallback)
         fProgressCallback->Call(ctrZippedBytes->GetValueAsInt(), i);
   }
   if (fProgressCallback)
      fProgressCallback->Finish(ctrZippedBytes->GetValueAsInt(), nEntries);

   return RResult<void>::Success();
}
