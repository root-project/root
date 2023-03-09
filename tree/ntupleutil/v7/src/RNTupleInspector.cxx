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
#include <deque>
#include <exception>

ROOT::Experimental::RNTupleInspector::RNTupleInspector(
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
   : fPageSource(std::move(pageSource))
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   fDescriptor = descriptorGuard->Clone();
}

void ROOT::Experimental::RNTupleInspector::CollectColumnInfo()
{
   for (DescriptorId_t colId = 0; colId < fDescriptor->GetNPhysicalColumns(); ++colId) {
      RColumnInfo info;
      info.fColumnDescriptor = &(fDescriptor->GetColumnDescriptor(colId));

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements.
      info.fElementSize = ROOT::Experimental::Detail::RColumnElementBase::Generate(info.GetType())->GetSize();

      for (const auto &clusterDescriptor : fDescriptor->GetClusterIterable()) {
         if (!clusterDescriptor.ContainsColumn(colId)) {
            continue;
         }

         auto columnRange = clusterDescriptor.GetColumnRange(colId);
         info.fNElements += columnRange.fNElements;

         if (fCompressionSettings == -1) {
            fCompressionSettings = columnRange.fCompressionSettings;
         }

         const auto &pageRange = clusterDescriptor.GetPageRange(colId);

         for (const auto &page : pageRange.fPageInfos) {
            info.fOnDiskSize += page.fLocator.fBytesOnStorage;
            info.fInMemorySize += page.fNElements * info.fElementSize;
            fOnDiskSize += info.fOnDiskSize;
            fInMemorySize += info.fInMemorySize;
         }
      }

      fColumnInfo.emplace_back(info);
   }
}

void ROOT::Experimental::RNTupleInspector::CollectFieldInfo()
{
   std::deque<DescriptorId_t> fieldIdQueue{fDescriptor->GetFieldZeroId()};

   while (!fieldIdQueue.empty()) {
      auto currId = fieldIdQueue.front();
      fieldIdQueue.pop_front();

      for (const auto &fieldDescriptor : fDescriptor->GetFieldIterable(currId)) {
         RFieldInfo info;
         info.fFieldDescriptor = &fieldDescriptor;

         for (const auto colId : GetColumnsForFieldTree(fieldDescriptor.GetId())) {
            auto colInfo = GetColumnInfo(colId);
            info.fOnDiskSize += colInfo.fOnDiskSize;
            info.fInMemorySize += colInfo.fInMemorySize;
         }

         fFieldInfo.emplace_back(info);
         fieldIdQueue.push_back(fieldDescriptor.GetId());
      }
   }
}

std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetColumnsForFieldTree(DescriptorId_t fieldId)
{
   std::vector<DescriptorId_t> colIds;
   std::deque<DescriptorId_t> fieldIdQueue{fieldId};

   while (!fieldIdQueue.empty()) {
      auto currId = fieldIdQueue.front();
      fieldIdQueue.pop_front();

      for (const auto &col : fDescriptor->GetColumnIterable(currId)) {
         if (col.IsAliasColumn()) {
            continue;
         }

         colIds.emplace_back(col.GetPhysicalId());
      }

      for (const auto &fld : fDescriptor->GetFieldIterable(currId)) {
         fieldIdQueue.push_back(fld.GetId());
      }
   }

   return colIds;
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
{
   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector(std::move(pageSource)));

   inspector->CollectColumnInfo();

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

ROOT::Experimental::RNTupleDescriptor *ROOT::Experimental::RNTupleInspector::GetDescriptor()
{
   return fDescriptor.get();
}

int ROOT::Experimental::RNTupleInspector::GetCompressionSettings()
{
   return fCompressionSettings;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetOnDiskSize()
{
   return fOnDiskSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetInMemorySize()
{
   return fInMemorySize;
}

float ROOT::Experimental::RNTupleInspector::GetCompressionFactor()
{
   return (float)fInMemorySize / (float)fOnDiskSize;
}

int ROOT::Experimental::RNTupleInspector::GetFieldTypeCount(const std::string typeName, bool includeSubFields)
{
   if (fFieldInfo.empty()) {
      CollectFieldInfo();
   }

   int typeCount = 0;

   for (const auto &fldInfo : fFieldInfo) {
      if (!includeSubFields && fldInfo.fFieldDescriptor->GetParentId() != fDescriptor->GetFieldZeroId()) {
         continue;
      }

      if (typeName == fldInfo.fFieldDescriptor->GetTypeName()) {
         typeCount++;
      }
   }

   return typeCount;
}

int ROOT::Experimental::RNTupleInspector::GetColumnTypeCount(ROOT::Experimental::EColumnType colType)
{
   int typeCount = 0;

   for (const auto &colInfo : fColumnInfo) {
      if (colInfo.fColumnDescriptor->GetModel().GetType() == colType) {
         ++typeCount;
      }
   }

   return typeCount;
}

ROOT::Experimental::RNTupleInspector::RColumnInfo
ROOT::Experimental::RNTupleInspector::GetColumnInfo(DescriptorId_t physicalColumnId)
{
   if (physicalColumnId > fDescriptor->GetNPhysicalColumns()) {
      throw RException(R__FAIL("No column with physical ID " + std::to_string(physicalColumnId) + " present"));
   }

   return fColumnInfo.at(physicalColumnId);
}

ROOT::Experimental::RNTupleInspector::RFieldInfo
ROOT::Experimental::RNTupleInspector::GetFieldInfo(DescriptorId_t fieldId)
{
   if (fFieldInfo.empty()) {
      CollectFieldInfo();
   }

   return fFieldInfo.at(fieldId);
}

ROOT::Experimental::RNTupleInspector::RFieldInfo
ROOT::Experimental::RNTupleInspector::GetFieldInfo(const std::string fieldName)
{
   DescriptorId_t fieldId = fDescriptor->FindFieldId(fieldName);

   if (fieldId == kInvalidDescriptorId) {
      throw RException(R__FAIL("Could not find field `" + fieldName + "`"));
   }

   return GetFieldInfo(fieldId);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RColumnDescriptor *ROOT::Experimental::RNTupleInspector::RColumnInfo::GetDescriptor()
{
   return fColumnDescriptor;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RColumnInfo::GetOnDiskSize()
{
   return fOnDiskSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RColumnInfo::GetInMemorySize()
{
   return fInMemorySize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RColumnInfo::GetElementSize()
{
   return fElementSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RColumnInfo::GetNElements()
{
   return fNElements;
}

ROOT::Experimental::EColumnType ROOT::Experimental::RNTupleInspector::RColumnInfo::GetType()
{
   if (!fColumnDescriptor) {
      return ROOT::Experimental::EColumnType::kUnknown;
   }

   return fColumnDescriptor->GetModel().GetType();
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldDescriptor *ROOT::Experimental::RNTupleInspector::RFieldInfo::GetDescriptor()
{
   return fFieldDescriptor;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RFieldInfo::GetOnDiskSize()
{
   return fOnDiskSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::RFieldInfo::GetInMemorySize()
{
   return fInMemorySize;
}
