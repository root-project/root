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
#include <ROOT/RNTupleImporter.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <string_view>

#include <TBranch.h>
#include <TChain.h>
#include <TClass.h>
#include <TDataType.h>
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
   static constexpr std::uint64_t gUpdateFrequencyBytes = 100 * 1000 * 1000; // report every 100 MB
   std::uint64_t fNbytesNext = gUpdateFrequencyBytes;

public:
   ~RDefaultProgressCallback() override {}
   void Call(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      // Report if more than 100 MB (compressed) where written since the last status update
      if (nbytesWritten < fNbytesNext)
         return;
      std::cout << "Wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries\n";
      fNbytesNext += gUpdateFrequencyBytes;
      if (nbytesWritten > fNbytesNext) {
         // If we already passed the next threshold, increase by a sensible amount.
         fNbytesNext = nbytesWritten + gUpdateFrequencyBytes;
      }
   }

   void Finish(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) final
   {
      std::cout << "Done, wrote " << nbytesWritten / 1000 / 1000 << "MB, " << neventsWritten << " entries\n";
   }
};

} // anonymous namespace

ROOT::RResult<void>
ROOT::Experimental::RNTupleImporter::RCStringTransformation::Transform(const RImportBranch &branch, RImportField &field)
{
   *reinterpret_cast<std::string *>(field.fFieldBuffer) = reinterpret_cast<const char *>(branch.fBranchBuffer.get());
   return RResult<void>::Success();
}

std::unique_ptr<ROOT::Experimental::RNTupleImporter>
ROOT::Experimental::RNTupleImporter::Create(std::string_view sourceFileName, std::string_view treeName,
                                            std::string_view destFileName)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());
   importer->fNTupleName = treeName;
   importer->fSourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!importer->fSourceFile || importer->fSourceFile->IsZombie()) {
      throw RException(R__FAIL("cannot open source file " + std::string(sourceFileName)));
   }

   importer->fSourceTree = importer->fSourceFile->Get<TTree>(std::string(treeName).c_str());
   if (!importer->fSourceTree) {
      throw RException(R__FAIL("cannot read TTree " + std::string(treeName) + " from " + std::string(sourceFileName)));
   }

   // If we have IMT enabled, its best use is for parallel page compression
   importer->fSourceTree->SetImplicitMT(false);
   auto result = importer->InitDestination(destFileName);

   if (!result)
      throw RException(R__FORWARD_ERROR(result));

   return importer;
}

std::unique_ptr<ROOT::Experimental::RNTupleImporter>
ROOT::Experimental::RNTupleImporter::Create(TTree *sourceTree, std::string_view destFileName)
{
   auto importer = std::unique_ptr<RNTupleImporter>(new RNTupleImporter());

   if (sourceTree->IsA() == TChain::Class() && std::strcmp(sourceTree->GetName(), "") == 0) {
      if (sourceTree->LoadTree(0) != 0)
         throw RException(R__FAIL("failure retrieving first tree from provided chain"));
      importer->fNTupleName = sourceTree->GetTree()->GetName();
   } else {
      importer->fNTupleName = sourceTree->GetName();
   }

   importer->fSourceTree = sourceTree;

   // If we have IMT enabled, its best use is for parallel page compression
   importer->fSourceTree->SetImplicitMT(false);
   auto result = importer->InitDestination(destFileName);

   if (!result)
      throw RException(R__FORWARD_ERROR(result));

   return importer;
}

ROOT::RResult<void> ROOT::Experimental::RNTupleImporter::InitDestination(std::string_view destFileName)
{
   fDestFileName = destFileName;
   fDestFile = std::unique_ptr<TFile>(TFile::Open(fDestFileName.c_str(), "UPDATE"));
   if (!fDestFile || fDestFile->IsZombie()) {
      return R__FAIL("cannot open dest file " + std::string(fDestFileName));
   }

   return RResult<void>::Success();
}

void ROOT::Experimental::RNTupleImporter::ReportSchema()
{
   for (const auto &f : fImportFields) {
      std::cout << "Importing '" << f.fField->GetFieldName() << "' [" << f.fField->GetTypeName() << "]\n";
   }
   for (const auto &f : Internal::GetProjectedFieldsOfModel(*fModel).GetFieldZero().GetSubFields()) {
      std::cout << "Importing (projected) '" << f->GetFieldName() << "' [" << f->GetTypeName() << "]\n";
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

ROOT::RResult<void> ROOT::Experimental::RNTupleImporter::PrepareSchema()
{
   ResetSchema();

   // Browse through all branches and their leaves, create corresponding fields and prepare the memory buffers for
   // reading and writing. Usually, reading and writing share the same memory buffer, i.e. the object is read from TTree
   // and written as-is to the RNTuple. There are exceptions, e.g. for leaf count arrays and C strings.
   for (auto b : TRangeDynCast<TBranch>(*fSourceTree->GetListOfBranches())) {
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
      // We use GetLeafCounter instead of GetLeafCount and GetLenStatic because the latter don't distinguish between
      // char arrays and C strings.
      Int_t firstLeafCountval;
      const bool isCString = !isLeafList && (firstLeaf->IsA() == TLeafC::Class()) &&
                             (!firstLeaf->GetLeafCounter(firstLeafCountval)) && (firstLeafCountval == 1);

      if (isCountLeaf) {
         // This is a count leaf.  We expect that this is not part of a leaf list. We also expect that the
         // leaf count comes before any array leaves that use it.
         // Count leaf branches do not end up as (physical) fields but they trigger the creation of an untyped
         // collection, together the collection mode.
         RImportLeafCountCollection c;
         c.fMaxLength = firstLeaf->GetMaximum();
         c.fCountVal = std::make_unique<Int_t>(); // count leafs are integers
         // Casting to void * makes it work for both Int_t and UInt_t
         fSourceTree->SetBranchAddress(b->GetName(), static_cast<void *>(c.fCountVal.get()));
         fLeafCountCollections.emplace(firstLeaf->GetName(), std::move(c));
         continue;
      }

      std::size_t branchBufferSize = 0; // Size of the memory location into which TTree reads the events' branch data
      // For leaf lists, every leaf translates into a sub field of an untyped RNTuple record
      std::vector<std::unique_ptr<RFieldBase>> recordItems;
      // For leaf count arrays, we expect to find a single leaf; we don't add a field right away but only
      // later through a projection
      bool isLeafCountArray = false;
      for (auto l : TRangeDynCast<TLeaf>(b->GetListOfLeaves())) {
         if (l->IsA() == TLeafObject::Class()) {
            return R__FAIL("unsupported: TObject branches, branch: " + std::string(b->GetName()));
         }

         // We don't use GetLeafCounter() because it relies on the correct format of the leaf title.
         // There are files in the public where the title is broken (empty).
         Int_t countval = l->GetLenStatic();
         auto *countleaf = l->GetLeafCount();
         const bool isFixedSizeArray = !isCString && (countleaf == nullptr) && (countval > 1);
         isLeafCountArray = (countleaf != nullptr);

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

         if (fConvertDotsInBranchNames) {
            // Replace any occurrence of a dot ('.') with an underscore.
            std::replace(fieldName.begin(), fieldName.end(), '.', '_');
         }

         RImportField f;
         auto fieldOrError = RFieldBase::Create(fieldName, fieldType);
         if (!fieldOrError)
            return R__FORWARD_ERROR(fieldOrError);
         auto field = fieldOrError.Unwrap();
         if (isCString) {
            branchBufferSize = l->GetMaximum();
            f.fValue = std::make_unique<RFieldBase::RValue>(field->CreateValue());
            f.fFieldBuffer = f.fValue->GetPtr<void>().get();
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
            const std::string countleafName = countleaf->GetName();
            fLeafCountCollections[countleafName].fLeafFields.emplace_back(std::move(field));
            fLeafCountCollections[countleafName].fLeafBranchIndexes.emplace_back(fImportBranches.size());
            R__ASSERT(b->GetListOfLeaves()->GetEntries() == 1);
            break;
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
         fSourceTree->SetBranchAddress(b->GetName(), ptrBuf, klass, EDataType::kOther_t, true /* isptr*/);
      } else {
         fSourceTree->SetBranchAddress(b->GetName(), reinterpret_cast<void *>(ib.fBranchBuffer.get()));
      }

      // If the TTree branch type and the RNTuple field type match, use the branch read buffer as RNTuple write buffer
      if (!isLeafCountArray && !fImportFields.back().fFieldBuffer) {
         fImportFields.back().fFieldBuffer =
            isClass ? *reinterpret_cast<void **>(ib.fBranchBuffer.get()) : ib.fBranchBuffer.get();
      }

      fImportBranches.emplace_back(std::move(ib));
   }

   int iLeafCountCollection = 0;
   for (auto &p : fLeafCountCollections) {
      // We want to capture this variable, which is not possible with a
      // structured binding in C++17. Explicitly defining a variable works.
      auto countLeafName = p.first;
      auto &c = p.second;

      c.fFieldName = "_collection" + std::to_string(iLeafCountCollection);
      auto recordField = std::make_unique<RRecordField>("_0", std::move(c.fLeafFields));
      c.fRecordField = recordField.get();
      auto collectionField = RVectorField::CreateUntyped(c.fFieldName, std::move(recordField));
      fModel->AddField(std::move(collectionField));

      // Add projected fields for all leaf count arrays
      for (const auto leaf : c.fRecordField->GetSubFields()) {
         const auto name = leaf->GetFieldName();
         auto projectedField = RFieldBase::Create(name, "ROOT::VecOps::RVec<" + leaf->GetTypeName() + ">").Unwrap();
         fModel->AddProjectedField(std::move(projectedField), [&name, &c](const std::string &fieldName) {
            if (fieldName == name)
               return c.fFieldName;
            else
               return c.fFieldName + "._0." + name;
         });
      }

      if (fConvertDotsInBranchNames) {
         // Replace any occurrenceof a dot ('.') in the count leaf name with an underscore.
         std::replace(countLeafName.begin(), countLeafName.end(), '.', '_');
      }

      // Add projected fields for count leaf
      auto projectedField = RFieldBase::Create(countLeafName, "ROOT::RNTupleCardinality<std::uint32_t>").Unwrap();
      fModel->AddProjectedField(std::move(projectedField), [&c](const std::string &) { return c.fFieldName; });

      iLeafCountCollection++;
   }

   if (fFieldModifier) {
      for (auto &field : fModel->GetMutableFieldZero()) {
         fFieldModifier(field);
      }
   }

   fModel->Freeze();

   fEntry = fModel->CreateBareEntry();
   for (const auto &f : fImportFields) {
      fEntry->BindRawPtr(f.fField->GetFieldName(), f.fFieldBuffer);
   }
   for (auto &[_, c] : fLeafCountCollections) {
      fEntry->BindRawPtr<void>(c.fFieldName, &c.fFieldBuffer);
   }

   if (!fIsQuiet)
      ReportSchema();

   return RResult<void>::Success();
}

void ROOT::Experimental::RNTupleImporter::Import()
{
   if (fDestFile->FindKey(fNTupleName.c_str()) != nullptr)
      throw RException(R__FAIL("Key '" + fNTupleName + "' already exists in file " + fDestFileName));

   PrepareSchema();

   std::unique_ptr<Internal::RPageSink> sink =
      std::make_unique<Internal::RPageSinkFile>(fNTupleName, *fDestFile, fWriteOptions);
   sink->GetMetrics().Enable();
   auto ctrZippedBytes = sink->GetMetrics().GetCounter("RPageSinkFile.szWritePayload");

   if (fWriteOptions.GetUseBufferedWrite()) {
      sink = std::make_unique<Internal::RPageSinkBuf>(std::move(sink));
   }

   auto ntplWriter = Internal::CreateRNTupleWriter(std::move(fModel), std::move(sink));
   // The guard needs to be destructed before the writer goes out of scope
   RImportGuard importGuard(*this);

   fProgressCallback = fIsQuiet ? nullptr : std::make_unique<RDefaultProgressCallback>();

   auto nEntries = fSourceTree->GetEntries();

   if (fMaxEntries >= 0 && fMaxEntries < nEntries) {
      nEntries = fMaxEntries;
   }

   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      fSourceTree->GetEntry(i);

      for (auto &[_, c] : fLeafCountCollections) {
         const auto sizeOfRecord = c.fRecordField->GetValueSize();
         c.fFieldBuffer.resize(sizeOfRecord * (*c.fCountVal));

         const auto nLeafs = c.fRecordField->GetSubFields().size();
         for (std::size_t l = 0; l < nLeafs; ++l) {
            const auto offset = c.fRecordField->GetOffsets()[l];
            const auto sizeOfLeaf = c.fRecordField->GetSubFields()[l]->GetValueSize();
            const auto idxImportBranch = c.fLeafBranchIndexes[l];
            for (Int_t j = 0; j < *c.fCountVal; ++j) {
               memcpy(c.fFieldBuffer.data() + j * sizeOfRecord + offset,
                      fImportBranches[idxImportBranch].fBranchBuffer.get() + (j * sizeOfLeaf), sizeOfLeaf);
            }
         }
      }

      for (auto &t : fImportTransformations) {
         auto result = t->Transform(fImportBranches[t->fImportBranchIdx], fImportFields[t->fImportFieldIdx]);
         if (!result)
            throw RException(R__FORWARD_ERROR(result));
      }

      ntplWriter->Fill(*fEntry);

      if (fProgressCallback)
         fProgressCallback->Call(ctrZippedBytes->GetValueAsInt(), i);
   }
   if (fProgressCallback)
      fProgressCallback->Finish(ctrZippedBytes->GetValueAsInt(), nEntries);
}
