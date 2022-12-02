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
#include <TLeafObject.h>

#include <cassert>
#include <cstdint>
#include <cstring>
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

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleImporter::RCStringTransformation::Transform(std::int64_t /* entry */,
                                                                       const RImportBranch &branch, RImportField &field)
{
   *reinterpret_cast<std::string *>(field.fFieldBuffer) = reinterpret_cast<const char *>(branch.fBranchBuffer.get());
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleImporter::RLeafArrayTransformation::Transform(std::int64_t entry,
                                                                         const RImportBranch &branch,
                                                                         RImportField &field)
{
   if (entry != fEntry) {
      fEntry = entry;
      fNum = 0;
   }
   auto valueSize = field.fField->GetValueSize();
   memcpy(field.fFieldBuffer, branch.fBranchBuffer.get() + (fNum * valueSize), valueSize);
   fNum++;
   return RResult<void>::Success();
}

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
   importer->fWriteOptions.SetCompression(kDefaultCompressionSettings);
   importer->fDestFile = std::unique_ptr<TFile>(TFile::Open(importer->fDestFileName.c_str(), "UPDATE"));
   if (!importer->fDestFile || importer->fDestFile->IsZombie()) {
      return R__FAIL("cannot open dest file " + std::string(importer->fDestFileName));
   }

   return importer;
}

void ROOT::Experimental::RNTupleImporter::ReportSchema()
{
   for (const auto &f : fImportFields) {
      std::cout << "Importing '" << f.fField->GetName() << "' [" << f.fField->GetType() << ']' << std::endl;
   }
}

void ROOT::Experimental::RNTupleImporter::ResetSchema()
{
   fImportBranches.clear();
   fImportFields.clear();
   fLeafCountCollections.clear();
   fImportTransformations.clear();
   fSourceTree->SetBranchStatus("*", 0);
   fModel = RNTupleModel::CreateBare();
   fEntry = nullptr;
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::PrepareSchema()
{
   ResetSchema();

   for (auto b : TRangeDynCast<TBranch>(*fSourceTree->GetListOfBranches())) {
      assert(b);
      const auto firstLeaf = static_cast<TLeaf *>(b->GetListOfLeaves()->First());
      assert(firstLeaf);

      if (firstLeaf->IsRange()) {
         // This is a count leaf.  We expect that this is not part of a leave list. We also expect that the
         // leaf count comes before any array leaves that use it.
         // Count leaf branches do not end up as fields but they trigger the creation of an anonymous collection,
         // together the collection mode.
         RImportLeafCountCollection c;
         c.fCollectionModel = RNTupleModel::CreateBare();
         c.fMaxLength = firstLeaf->GetMaximum();
         c.fCountVal = std::make_unique<Int_t>(); // count leafs are integers
         fSourceTree->SetBranchStatus(b->GetName(), 1);
         fSourceTree->SetBranchAddress(b->GetName(), c.fCountVal.get());
         // We use the leaf pointer as a map key.  The array leafs return that leaf pointer from GetLeafCount(),
         // so that they will be able to find the information to attach their fields to the collection model.
         fLeafCountCollections.emplace(firstLeaf, std::move(c));
         continue;
      }

      const bool isLeafList = b->GetNleaves() > 1;
      Int_t firstLeafCountval;
      const bool isCString = !isLeafList && (firstLeaf->IsA() == TLeafC::Class()) &&
                             (!firstLeaf->GetLeafCounter(firstLeafCountval)) && (firstLeafCountval == 1);

      std::size_t branchBufferSize = 0; // Size of the memory location into which TTree reads the events' branch data
      // For leaf lists, every leaf translates into a sub field of an anonymous RNTuple record
      std::vector<std::unique_ptr<Detail::RFieldBase>> recordItems;
      for (auto l : TRangeDynCast<TLeaf>(b->GetListOfLeaves())) {
         if (l->IsA() == TLeafObject::Class()) {
            return R__FAIL(std::string("importing TObject branches not supported: ") +
                           std::string(l->GetFullName().View()));
         }

         Int_t countval = 0;
         auto *countleaf = l->GetLeafCounter(countval);
         const bool isLeafCountArray = (countleaf != nullptr);
         const bool isFixedSizeArray = (countleaf == nullptr) && (countval > 1);

         std::string fieldName = isLeafList ? l->GetName() : b->GetName();
         std::string fieldType = isCString ? "std::string" : l->GetTypeName();
         if (isFixedSizeArray)
            fieldType = "std::array<" + fieldType + "," + std::to_string(countval) + ">";

         RImportField f;
         std::unique_ptr<Detail::RFieldBase> field;
         if (isCString) {
            branchBufferSize = l->GetMaximum();
            auto result = Detail::RFieldBase::Create(fieldName, fieldType);
            if (!result)
               return R__FORWARD_ERROR(result);
            field = result.Unwrap();
            f.fFieldBuffer = field->GenerateValue().GetRawPtr();
            f.fOwnsFieldBuffer = true;
            fImportTransformations.emplace_back(
               std::make_unique<RCStringTransformation>(fImportBranches.size(), fImportFields.size()));
         } else {
            auto result = Detail::RFieldBase::Create(fieldName, fieldType);
            if (!result)
               return R__FORWARD_ERROR(result);
            field = result.Unwrap();
            if (isLeafCountArray) {
               branchBufferSize = fLeafCountCollections[countleaf].fMaxLength * field->GetValueSize();
            } else {
               branchBufferSize = l->GetOffset() + field->GetValueSize();
            }
         }
         f.fField = field.get();

         if (isLeafList) {
            recordItems.emplace_back(std::move(field));
         } else if (isLeafCountArray) {
            f.fFieldBuffer = field->GenerateValue().GetRawPtr();
            f.fOwnsFieldBuffer = true;
            f.fIsInUntypedCollection = true;
            fLeafCountCollections[countleaf].fCollectionModel->AddField(std::move(field));
            fLeafCountCollections[countleaf].fImportFieldIndexes.emplace_back(fImportFields.size());
            fImportTransformations.emplace_back(
               std::make_unique<RLeafArrayTransformation>(fImportBranches.size(), fImportFields.size()));
            fImportFields.emplace_back(std::move(f));
         } else {
            fModel->AddField(std::move(field));
            fImportFields.emplace_back(std::move(f));
         }
      }
      if (!recordItems.empty()) {
         auto recordField = std::make_unique<RRecordField>(b->GetName(), std::move(recordItems));
         RImportField f;
         f.fField = recordField.get();
         fImportFields.emplace_back(std::move(f));
         fModel->AddField(std::move(recordField));
      }

      RImportBranch ib;
      ib.fBranchName = b->GetName();
      ib.fBranchBuffer = std::make_unique<unsigned char[]>(branchBufferSize);
      fSourceTree->SetBranchStatus(b->GetName(), 1);
      fSourceTree->SetBranchAddress(b->GetName(), reinterpret_cast<void *>(ib.fBranchBuffer.get()));

      if (!fImportFields.back().fFieldBuffer)
         fImportFields.back().fFieldBuffer = ib.fBranchBuffer.get();

      fImportBranches.emplace_back(std::move(ib));
   }

   int iLeafCountCollection = 0;
   for (auto &[_, c] : fLeafCountCollections) {
      c.fCollectionModel->Freeze();
      c.fCollectionEntry = c.fCollectionModel->CreateBareEntry();
      for (auto idx : c.fImportFieldIndexes) {
         const auto name = fImportFields[idx].fField->GetName();
         const auto buffer = fImportFields[idx].fFieldBuffer;
         c.fCollectionEntry->CaptureValueUnsafe(name, buffer);
      }
      c.fFieldName = "_collection" + std::to_string(iLeafCountCollection);
      c.fCollectionWriter = fModel->MakeCollection(c.fFieldName, std::move(c.fCollectionModel));
      iLeafCountCollection++;
   }

   fModel->Freeze();
   fEntry = fModel->CreateBareEntry();
   for (const auto &f : fImportFields) {
      if (f.fIsInUntypedCollection)
         continue;
      fEntry->CaptureValueUnsafe(f.fField->GetName(), f.fFieldBuffer);
   }
   for (const auto &[_, c] : fLeafCountCollections) {
      fEntry->CaptureValueUnsafe(c.fFieldName, c.fCollectionWriter->GetOffsetPtr());
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

      for (const auto &[_, c] : fLeafCountCollections) {
         for (Int_t l = 0; l < *c.fCountVal; ++l) {
            for (auto &t : fImportTransformations) {
               if (!fImportFields[t->fImportFieldIdx].fIsInUntypedCollection)
                  continue;
               auto result = t->Transform(i, fImportBranches[t->fImportBranchIdx], fImportFields[t->fImportFieldIdx]);
               if (!result)
                  return R__FORWARD_ERROR(result);
            }
            c.fCollectionWriter->Fill(c.fCollectionEntry.get());
         }
      }

      for (auto &t : fImportTransformations) {
         if (fImportFields[t->fImportFieldIdx].fIsInUntypedCollection)
            continue;
         auto result = t->Transform(i, fImportBranches[t->fImportBranchIdx], fImportFields[t->fImportFieldIdx]);
         if (!result)
            return R__FORWARD_ERROR(result);
      }

      ntplWriter->Fill(*fEntry);

      if (fProgressCallback)
         fProgressCallback->Call(ctrZippedBytes->GetValueAsInt(), i);
   }
   if (fProgressCallback)
      fProgressCallback->Finish(ctrZippedBytes->GetValueAsInt(), nEntries);

   return RResult<void>::Success();
}
