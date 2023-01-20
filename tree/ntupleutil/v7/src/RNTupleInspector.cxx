/// \file RNTupleInspector.cxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.willemijn.de.geus@cern.ch>
/// \date 2023-01-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RError.hxx>

#include <TFile.h>

#include <cstring>
#include <iostream>
#include <algorithm>

void ROOT::Experimental::RNTupleInspector::CollectNTupleData()
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   int compressionSettings = -1;
   std::uint64_t compressedSize = 0;
   std::uint64_t uncompressedSize = 0;

   for (const auto &clusterDescriptor : descriptorGuard->GetClusterIterable()) {
      compressedSize += clusterDescriptor.GetBytesOnStorage();

      if (compressionSettings == -1) {
         compressionSettings = clusterDescriptor.GetColumnRange(0).fCompressionSettings;
      }
   }

   for (uint64_t colId = 0; colId < descriptorGuard->GetNPhysicalColumns(); ++colId) {
      const ROOT::Experimental::RColumnDescriptor &colDescriptor = descriptorGuard->GetColumnDescriptor(colId);

      uint64_t elemSize =
         ROOT::Experimental::Detail::RColumnElementBase::Generate(colDescriptor.GetModel().GetType())->GetSize();

      uint64_t nElems = descriptorGuard->GetNElements(colId);
      uncompressedSize += nElems * elemSize;
   }

   fCompressionSettings = compressionSettings;
   fCompressedSize = compressedSize;
   fUncompressedSize = uncompressedSize;
}

void ROOT::Experimental::RNTupleInspector::CollectColumnData()
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   for (DescriptorId_t colId = 0; colId < descriptorGuard->GetNPhysicalColumns(); ++colId) {
      const ROOT::Experimental::RColumnDescriptor &colDescriptor = descriptorGuard->GetColumnDescriptor(colId);

      RColumnInfo info;
      info.fPhysicalColumnId = colId;
      info.fLogicalColumnId = colDescriptor.GetLogicalId();
      info.fFieldId = colDescriptor.GetFieldId();
      info.fIndex = colDescriptor.GetIndex();
      info.fType = colDescriptor.GetModel().GetType();

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements.
      uint64_t elemSize =
         ROOT::Experimental::Detail::RColumnElementBase::Generate(colDescriptor.GetModel().GetType())->GetSize();
      info.fElementSize = elemSize;

      for (const auto &clusterDescriptor : descriptorGuard->GetClusterIterable()) {
         auto columnRange = clusterDescriptor.GetColumnRange(colId);
         info.fNElements += columnRange.fNElements;

         const auto &pageRange = clusterDescriptor.GetPageRange(colId);

         for (const auto &page : pageRange.fPageInfos) {
            info.fCompressedSize += page.fLocator.fBytesOnStorage;
            info.fUncompressedSize += page.fNElements * elemSize;
         }
      }

      fColumnInfo.emplace_back(info);
   }
}

std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetColumnsForField(ROOT::Experimental::DescriptorId_t fieldId)
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   const RFieldDescriptor &fieldDescriptor = descriptorGuard->GetFieldDescriptor(fieldId);
   std::vector<ROOT::Experimental::DescriptorId_t> colIds;

   switch (fieldDescriptor.GetStructure()) {
   case kLeaf:
      for (const auto &col : descriptorGuard->GetColumnIterable(fieldDescriptor)) {
         colIds.emplace_back(col.GetPhysicalId());
      }
      break;
   case kCollection:
      for (const auto &col : descriptorGuard->GetColumnIterable(fieldDescriptor)) {
         colIds.emplace_back(col.GetPhysicalId());
      }
   case kRecord:
      for (const auto &fld : descriptorGuard->GetFieldIterable(fieldId)) {
         auto rColIds = GetColumnsForField(fld.GetId());
         colIds.insert(colIds.end(), rColIds.begin(), rColIds.end());
      }
      break;
   default:
      // TODO fdegeus
      std::cerr << "structure type " << fieldDescriptor.GetStructure() << " not supported yet" << std::endl;
   }

   return colIds;
}

//------------------------------------------------------------------------------

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
{
   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector(std::move(pageSource)));

   // TODO Memoize instead of calling everything in the constructor?
   inspector->CollectNTupleData();

   return inspector;
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(ROOT::Experimental::RNTuple *sourceNTuple)
{
   if (!sourceNTuple) {
      return R__FAIL("provided RNTuple is null");
   }

   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource = sourceNTuple->MakePageSource();

   return ROOT::Experimental::RNTupleInspector::Create(std::move(pageSource));
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(std::string_view ntupleName, std::string_view sourceFileName)
{
   auto sourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!sourceFile || sourceFile->IsZombie()) {
      return R__FAIL("cannot open source file " + std::string(sourceFileName));
   }
   auto ntuple = std::unique_ptr<ROOT::Experimental::RNTuple>(
      sourceFile->Get<ROOT::Experimental::RNTuple>(std::string(ntupleName).c_str()));
   if (!ntuple) {
      return R__FAIL("cannot read RNTuple " + std::string(ntupleName) + " from " + std::string(sourceFileName));
   }

   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector(ntuple->MakePageSource()));
   inspector->fSourceFile = std::move(sourceFile);

   inspector->CollectSizeData();

   return inspector;
}

std::string ROOT::Experimental::RNTupleInspector::GetName()
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   return descriptorGuard->GetName();
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetNEntries()
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   return descriptorGuard->GetNEntries();
}

int ROOT::Experimental::RNTupleInspector::GetCompressionSettings()
{
   return fCompressionSettings;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetCompressedSize()
{
   return fCompressedSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetUncompressedSize()
{
   return fUncompressedSize;
}

float ROOT::Experimental::RNTupleInspector::GetCompressionFactor()
{
   return (float)fUncompressedSize / (float)fCompressedSize;
}

int ROOT::Experimental::RNTupleInspector::GetFieldTypeFrequency(std::string className)
{
   if (fFieldTypeFrequencies.empty()) {
      auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

      for (const auto &fieldDescriptor : descriptorGuard->GetTopLevelFields()) {
         fFieldTypeFrequencies[fieldDescriptor.GetTypeName()]++;
      }
   }

   return fFieldTypeFrequencies[className];
}

int ROOT::Experimental::RNTupleInspector::GetColumnTypeFrequency(ROOT::Experimental::EColumnType colType)
{
   if (fColumnInfo.empty()) {
      CollectColumnData();
   }

   int typeFrequency = 0;

   for (auto const &colInfo : fColumnInfo) {
      if (colInfo.fType == colType) {
         ++typeFrequency;
      }
   }

   return typeFrequency;
}

std::uint64_t
ROOT::Experimental::RNTupleInspector::GetCompressedColumnSize(ROOT::Experimental::DescriptorId_t logicalId)
{
   if (fColumnInfo.empty()) {
      CollectColumnData();
   }

   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   if (logicalId > descriptorGuard->GetNLogicalColumns()) {
      std::cerr << "no column with id " << logicalId << " present" << std::endl;
      return -1;
   }

   ROOT::Experimental::DescriptorId_t physicalId = descriptorGuard->GetColumnDescriptor(logicalId).GetPhysicalId();

   return fColumnInfo.at(physicalId).fCompressedSize;
}

std::uint64_t
ROOT::Experimental::RNTupleInspector::GetUncompressedColumnSize(ROOT::Experimental::DescriptorId_t logicalId)
{
   if (fColumnInfo.empty()) {
      CollectColumnData();
   }

   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   if (logicalId > descriptorGuard->GetNLogicalColumns()) {
      std::cerr << "no column with id " << logicalId << " present" << std::endl;
      return -1;
   }

   ROOT::Experimental::DescriptorId_t physicalId = descriptorGuard->GetColumnDescriptor(logicalId).GetPhysicalId();
   return fColumnInfo.at(physicalId).fUncompressedSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetCompressedFieldSize(ROOT::Experimental::DescriptorId_t fieldId)
{
   std::uint64_t fieldSize = 0;

   for (const auto colId : GetColumnsForField(fieldId)) {
      fieldSize += GetCompressedColumnSize(colId);
   }

   return fieldSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetCompressedFieldSize(std::string fieldName)
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   ROOT::Experimental::DescriptorId_t fieldId = descriptorGuard->FindFieldId(fieldName);

   if (fieldId == kInvalidDescriptorId) {
      std::cerr << "could not find field \"" + fieldName + "\"." << std::endl;
      return -1;
   }

   return GetCompressedFieldSize(fieldId);
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetUncompressedFieldSize(ROOT::Experimental::DescriptorId_t fieldId)
{
   std::uint64_t fieldSize = 0;

   for (const auto colId : GetColumnsForField(fieldId)) {
      fieldSize += GetUncompressedColumnSize(colId);
   }

   return fieldSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetUncompressedFieldSize(std::string fieldName)
{
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   ROOT::Experimental::DescriptorId_t fieldId = descriptorGuard->FindFieldId(fieldName);

   if (fieldId == kInvalidDescriptorId) {
      std::cerr << "could not find field \"" + fieldName + "\"." << std::endl;
      return -1;
   }

   return GetUncompressedFieldSize(fieldId);
}

ROOT::Experimental::EColumnType
ROOT::Experimental::RNTupleInspector::GetColumnType(ROOT::Experimental::DescriptorId_t logicalId)
{
   if (fColumnInfo.empty()) {
      CollectColumnData();
   }

   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();

   return descriptorGuard->GetColumnDescriptor(logicalId).GetModel().GetType();
}
