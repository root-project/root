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
#include <TClass.h>
#include <TDataType.h>
#include <TFileMerger.h>
#include <TKey.h>
#include <TLeaf.h>
#include <TLeafC.h>
#include <TLeafElement.h>
#include <TLeafObject.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

namespace {

class RDefaultProgressCallback : public ROOT::Experimental::RNTupleImporter::RProgressCallback {
private:
   static constexpr std::uint64_t gUpdateFrequencyBytes = 50 * 1000 * 1000; // report every 50MB
   std::uint64_t fNbytesNext = gUpdateFrequencyBytes;

public:
   virtual ~RDefaultProgressCallback() {}
   void Call(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      // Report if more than 50MB (compressed) where written since the last status update
      if (nbytesWritten < fNbytesNext)
         return;
      std::cout << "Wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries" << std::endl;
      fNbytesNext += gUpdateFrequencyBytes;
   }

   void Finish(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      std::cout << "Done, wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries" << std::endl;
   }
};

} // anonymous namespace

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleImporter::RCStringTransformation::Transform(const RImportBranch &branch, RImportField &field)
{
   *reinterpret_cast<std::string *>(field.fFieldBuffer) = reinterpret_cast<const char *>(branch.fBranchBuffer.get());
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleImporter::RLeafArrayTransformation::Transform(const RImportBranch &branch,
                                                                         RImportField &field)
{
   auto valueSize = field.fField->GetValueSize();
   memcpy(field.fFieldBuffer, branch.fBranchBuffer.get() + (fNum * valueSize), valueSize);
   fNum++;
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleImporter>>
ROOT::Experimental::RNTupleImporter::Create(std::string_view sourceFileName, std::string_view treeName,
                                            std::string_view destFileName)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());
   importer->fNTupleNames[std::string(treeName)] = treeName;
   importer->fSourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!importer->fSourceFile || importer->fSourceFile->IsZombie()) {
      return R__FAIL("cannot open source file " + std::string(sourceFileName));
   }

   auto sourceTree = importer->fSourceFile->Get<TTree>(std::string(treeName).c_str());
   if (!sourceTree) {
      return R__FAIL("cannot read TTree " + std::string(treeName) + " from " + std::string(sourceFileName));
   }
   // If we have IMT enabled, its best use is for parallel page compression
   sourceTree->SetImplicitMT(false);
   importer->fSourceTrees.emplace_back(sourceTree);

   auto result = importer->InitDestination(destFileName);

   if (!result)
      return R__FORWARD_ERROR(result);

   return importer;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleImporter>>
ROOT::Experimental::RNTupleImporter::Create(std::string_view sourceFileName, std::string_view destFileName)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());
   importer->fSourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!importer->fSourceFile || importer->fSourceFile->IsZombie()) {
      return R__FAIL("cannot open source file " + std::string(sourceFileName));
   }

   auto objLink = importer->fSourceFile->GetListOfKeys()->FirstLink();
   while (objLink) {
      auto key = (TKey *)objLink->GetObject();
      auto keyName = key->GetName();
      bool isTree = TClass::GetClass(key->GetClassName())->GetBaseClass("TTree");
      bool isFirstOccurrence = importer->fNTupleNames.find(keyName) == importer->fNTupleNames.end();

      // TODO: Copy/clone non-TTree objects to the destination file, if desired by the user.
      if (isTree && isFirstOccurrence) {
         importer->fNTupleNames[keyName] = keyName;

         auto sourceTree = importer->fSourceFile->Get<TTree>(keyName);
         if (!sourceTree) {
            return R__FAIL("cannot read TTree " + std::string(keyName) + " from " + std::string(sourceFileName));
         }
         // If we have IMT enabled, its best use is for parallel page compression
         sourceTree->SetImplicitMT(false);
         importer->fSourceTrees.emplace_back(sourceTree);
      }

      objLink = objLink->Next();
   }

   auto result = importer->InitDestination(destFileName);

   if (!result)
      return R__FORWARD_ERROR(result);

   return importer;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleImporter>>
ROOT::Experimental::RNTupleImporter::Create(TTree *sourceTree, std::string_view destFileName)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());
   auto treeName = sourceTree->GetName();
   importer->fNTupleNames[treeName] = treeName;
   importer->fSourceTrees.emplace_back(sourceTree);

   // If we have IMT enabled, its best use is for parallel page compression
   importer->fSourceTrees.at(0)->SetImplicitMT(false);
   auto result = importer->InitDestination(destFileName);

   if (!result)
      return R__FORWARD_ERROR(result);

   return importer;
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::InitDestination(std::string_view destFileName)
{
   fDestFileName = destFileName;
   fWriteOptions.SetCompression(kDefaultCompressionSettings);
   fDestFile = std::unique_ptr<TFile>(TFile::Open(fDestFileName.c_str(), "UPDATE"));
   if (!fDestFile || fDestFile->IsZombie()) {
      return R__FAIL("cannot open dest file " + std::string(fDestFileName));
   }

   return RResult<void>::Success();
}

void ROOT::Experimental::RNTupleImporter::ReportSchema()
{
   for (const auto &f : fImportFields) {
      std::cout << "* Importing field '" << f.fField->GetName() << "' [" << f.fField->GetType() << ']' << std::endl;
   }
}

void ROOT::Experimental::RNTupleImporter::ResetSchema()
{
   fImportBranches.clear();
   fImportFields.clear();
   fLeafCountCollections.clear();
   fImportTransformations.clear();
   fModel = RNTupleModel::CreateBare();
   fEntry = nullptr;
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::PrepareSchema(TTree *sourceTree)
{
   ResetSchema();

   // Browse through all branches and their leaves, create corresponding fields and prepare the memory buffers for
   // reading and writing. Usually, reading and writing share the same memory buffer, i.e. the object is read from TTree
   // and written as-is to the RNTuple. There are exceptions, e.g. for leaf count arrays and C strings.
   for (auto b : TRangeDynCast<TBranch>(sourceTree->GetListOfBranches())) {
      assert(b);
      const auto firstLeaf = static_cast<TLeaf *>(b->GetListOfLeaves()->First());
      assert(firstLeaf);

      const bool isLeafList = b->GetNleaves() > 1;
      const bool isCountLeaf = firstLeaf->IsRange(); // A leaf storing the number of elements of a leaf count array
      const bool isClass = (firstLeaf->IsA() == TLeafElement::Class()); // STL or user-defined class
      if (isLeafList && isClass)
         return R__FAIL("unsupported: classes in leaf list, branch " + std::string(b->GetName()));
      if (isLeafList && isCountLeaf)
         return R__FAIL("unsupported: count leaf arrays in leaf list, branch " + std::string(b->GetName()));

      // Only plain leafs with type identifies 'C' are C strings. Otherwise, they are char arrays.
      Int_t firstLeafCountval;
      const bool isCString = !isLeafList && (firstLeaf->IsA() == TLeafC::Class()) &&
                             (!firstLeaf->GetLeafCounter(firstLeafCountval)) && (firstLeafCountval == 1);

      if (isCountLeaf) {
         // This is a count leaf.  We expect that this is not part of a leaf list. We also expect that the
         // leaf count comes before any array leaves that use it.
         // Count leaf branches do not end up as (physical) fields but they trigger the creation of an untyped
         // collection, together the collection mode.
         RImportLeafCountCollection c;
         c.fCollectionModel = RNTupleModel::CreateBare();
         c.fMaxLength = firstLeaf->GetMaximum();
         c.fCountVal = std::make_unique<Int_t>(); // count leafs are integers
         // Casting to void * makes it work for both Int_t and UInt_t
         sourceTree->SetBranchAddress(b->GetName(), static_cast<void *>(c.fCountVal.get()));
         fLeafCountCollections.emplace(firstLeaf->GetName(), std::move(c));
         continue;
      }

      std::size_t branchBufferSize = 0; // Size of the memory location into which TTree reads the events' branch data
      // For leaf lists, every leaf translates into a sub field of an untyped RNTuple record
      std::vector<std::unique_ptr<Detail::RFieldBase>> recordItems;
      for (auto l : TRangeDynCast<TLeaf>(b->GetListOfLeaves())) {
         if (l->IsA() == TLeafObject::Class()) {
            return R__FAIL("unsupported: TObject branches, branch: " + std::string(b->GetName()));
         }

         Int_t countval = 0;
         auto *countleaf = l->GetLeafCounter(countval);
         const bool isLeafCountArray = (countleaf != nullptr);
         const bool isFixedSizeArray = (countleaf == nullptr) && (countval > 1);

         // The base case for branches with fundamental, single numerical types.
         // For other types of branches, different field names or types are necessary,
         // which is determined below.
         std::string fieldName = b->GetName();
         std::string fieldType = l->GetTypeName();

         if (isLeafList)
            fieldName = l->GetName();

         if (isCString)
            fieldType = "std::string";

         if (isClass)
            fieldType = b->GetClassName();

         if (isFixedSizeArray)
            fieldType = "std::array<" + fieldType + "," + std::to_string(countval) + ">";

         RImportField f;
         f.fIsClass = isClass;
         auto fieldOrError = Detail::RFieldBase::Create(fieldName, fieldType);
         if (!fieldOrError)
            return R__FORWARD_ERROR(fieldOrError);
         auto field = fieldOrError.Unwrap();
         if (isCString) {
            branchBufferSize = l->GetMaximum();
            f.fFieldBuffer = field->GenerateValue().GetRawPtr();
            f.fOwnsFieldBuffer = true;
            fImportTransformations.emplace_back(
               std::make_unique<RCStringTransformation>(fImportBranches.size(), fImportFields.size()));
         } else {
            if (isClass) {
               // For classes, the branch buffer contains a pointer to object, which gets instantiated by TTree upon
               // calling SetBranchAddress()
               branchBufferSize = sizeof(void *) * countval;
            } else if (isLeafCountArray) {
               branchBufferSize = fLeafCountCollections[countleaf->GetName()].fMaxLength * field->GetValueSize();
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
            const std::string countleafName = countleaf->GetName();
            fLeafCountCollections[countleafName].fCollectionModel->AddField(std::move(field));
            fLeafCountCollections[countleafName].fImportFieldIndexes.emplace_back(fImportFields.size());
            fLeafCountCollections[countleafName].fTransformations.emplace_back(
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
      if (isClass) {
         auto klass = TClass::GetClass(b->GetClassName());
         if (!klass) {
            return R__FAIL("unable to load class " + std::string(b->GetClassName()) + " for branch " +
                           std::string(b->GetName()));
         }
         auto ptrBuf = reinterpret_cast<void **>(ib.fBranchBuffer.get());
         sourceTree->SetBranchAddress(b->GetName(), ptrBuf, klass, EDataType::kOther_t, true /* isptr*/);
      } else {
         sourceTree->SetBranchAddress(b->GetName(), reinterpret_cast<void *>(ib.fBranchBuffer.get()));
      }

      // If the TTree branch type and the RNTuple field type match, use the branch read buffer as RNTuple write buffer
      if (!fImportFields.back().fFieldBuffer) {
         fImportFields.back().fFieldBuffer =
            isClass ? *reinterpret_cast<void **>(ib.fBranchBuffer.get()) : ib.fBranchBuffer.get();
      }

      fImportBranches.emplace_back(std::move(ib));
   }

   int iLeafCountCollection = 0;
   for (auto &p : fLeafCountCollections) {
      // We want to capture this variable, which is not possible with a
      // structured binding in C++17. Explicitly defining a variable works.
      auto &countLeafName = p.first;
      auto &c = p.second;
      c.fCollectionModel->Freeze();
      c.fCollectionEntry = c.fCollectionModel->CreateBareEntry();
      for (auto idx : c.fImportFieldIndexes) {
         const auto name = fImportFields[idx].fField->GetName();
         const auto buffer = fImportFields[idx].fFieldBuffer;
         c.fCollectionEntry->CaptureValueUnsafe(name, buffer);
      }
      c.fFieldName = "_collection" + std::to_string(iLeafCountCollection);
      c.fCollectionWriter = fModel->MakeCollection(c.fFieldName, std::move(c.fCollectionModel));
      // Add projected fields for all leaf count arrays
      for (auto idx : c.fImportFieldIndexes) {
         const auto name = fImportFields[idx].fField->GetName();
         auto projectedField =
            Detail::RFieldBase::Create(name, "ROOT::RVec<" + fImportFields[idx].fField->GetType() + ">").Unwrap();
         fModel->AddProjectedField(std::move(projectedField), [&name, &c](const std::string &fieldName) {
            if (fieldName == name)
               return c.fFieldName;
            else
               return c.fFieldName + "." + name;
         });
      }
      // Add projected fields for count leaf
      auto projectedField =
         Detail::RFieldBase::Create(countLeafName, "ROOT::Experimental::RNTupleCardinality").Unwrap();
      fModel->AddProjectedField(std::move(projectedField), [&c](const std::string &) { return c.fFieldName; });
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

   if (!fIsQuiet) {
      ReportSchema();
   }

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::Import()
{
   for (const auto &sourceTree : fSourceTrees) {
      auto importResult = Import(sourceTree);

      if (!importResult)
         return R__FORWARD_ERROR(importResult);
   }

   // Since TFileMerger _has_ to take ownership of the destination file and deletes it afterwards, the copying of
   // non-TTree objects has to happen after the RNTuples are imported.
   if (fCopyNonTrees) {
      auto copyResult = CopyNonTrees();

      if (!copyResult)
         return R__FORWARD_ERROR(copyResult);
   }

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::Import(TTree *sourceTree)
{
   std::string ntupleName = fNTupleNames.at(sourceTree->GetName());
   if (fDestFile->FindKey(ntupleName.c_str()) != nullptr)
      return R__FAIL("Key '" + ntupleName + "' already exists in file " + fDestFileName);

   if (!fIsQuiet) {
      std::cout << "Importing '" << ntupleName << "'" << std::endl;
   }

   PrepareSchema(sourceTree);

   auto sink = std::make_unique<Detail::RPageSinkFile>(ntupleName, *fDestFile, fWriteOptions);
   sink->GetMetrics().Enable();
   auto ctrZippedBytes = sink->GetMetrics().GetCounter("RPageSinkFile.szWritePayload");

   auto ntplWriter = std::make_unique<RNTupleWriter>(std::move(fModel), std::move(sink));
   fModel = nullptr;

   fProgressCallback = fIsQuiet ? nullptr : std::make_unique<RDefaultProgressCallback>();

   auto nEntries = sourceTree->GetEntries();

   if (fMaxEntries >= 0 && fMaxEntries < nEntries) {
      nEntries = fMaxEntries;
   }

   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      sourceTree->GetEntry(i);

      for (const auto &[_, c] : fLeafCountCollections) {
         for (Int_t l = 0; l < *c.fCountVal; ++l) {
            for (auto &t : c.fTransformations) {
               auto result = t->Transform(fImportBranches[t->fImportBranchIdx], fImportFields[t->fImportFieldIdx]);
               if (!result)
                  return R__FORWARD_ERROR(result);
            }
            c.fCollectionWriter->Fill(c.fCollectionEntry.get());
         }
         for (auto &t : c.fTransformations)
            t->ResetEntry();
      }

      for (auto &t : fImportTransformations) {
         auto result = t->Transform(fImportBranches[t->fImportBranchIdx], fImportFields[t->fImportFieldIdx]);
         if (!result)
            return R__FORWARD_ERROR(result);
         t->ResetEntry();
      }

      ntplWriter->Fill(*fEntry);

      if (fProgressCallback)
         fProgressCallback->Call(ctrZippedBytes->GetValueAsInt(), i);
   }
   if (fProgressCallback)
      fProgressCallback->Finish(ctrZippedBytes->GetValueAsInt(), nEntries);

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::CopyNonTrees()
{
   if (!fIsQuiet)
      std::cout << "Importing remaining non-TTree files" << std::endl;

   auto fileMerger = std::make_unique<TFileMerger>();
   fileMerger->SetNotrees(true);
   if (!fileMerger->OutputFile(std::move(fDestFile)))
      return R__FAIL("Error opening destination file for non-TTree object copying");
   if (!fileMerger->AddFile(fSourceFile.get(), kFALSE))
      return R__FAIL("Error opening source file for non-TTree object copying");
   if (!fileMerger->PartialMerge(TFileMerger::kAll | TFileMerger::kRegular | TFileMerger::kKeepCompression))
      return R__FAIL("Error copying non-TTree objects");

   if (!fIsQuiet)
      std::cout << "Done" << std::endl;

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleImporter::SetNTupleName(std::string_view ntupleName)
{
   if (fSourceTrees.size() > 1) {
      return R__FAIL("Multiple trees are being imported, please specify the source tree name");
   }
   fNTupleNames.at(fSourceTrees.at(0)->GetName()) = ntupleName;
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleImporter::SetNTupleName(std::string_view treeName, std::string_view ntupleName)
{
   auto it = fNTupleNames.find(std::string(treeName));

   if (it == fNTupleNames.end()) {
      return R__FAIL("Tree '" + std::string(treeName) + "' not present in " + std::string(fSourceFile->GetName()));
   }

   (*it).second = ntupleName;
   return RResult<void>::Success();
}
