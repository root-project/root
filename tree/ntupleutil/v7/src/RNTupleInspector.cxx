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

#include <algorithm>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>

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
   fCompressedSize = 0;
   fUncompressedSize = 0;

   for (const auto &colDesc : fDescriptor->GetColumnIterable()) {
      auto colId = colDesc.GetPhysicalId();

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements.
      auto colType = colDesc.GetModel().GetType();
      std::uint32_t elemSize = ROOT::Experimental::Detail::RColumnElementBase::Generate(colType)->GetSize();
      std::uint64_t nElems = 0;
      std::uint64_t compressedSize = 0;

      for (const auto &clusterDescriptor : fDescriptor->GetClusterIterable()) {
         if (!clusterDescriptor.ContainsColumn(colId)) {
            continue;
         }

         auto columnRange = clusterDescriptor.GetColumnRange(colId);
         nElems += columnRange.fNElements;

         if (fCompressionSettings == -1) {
            fCompressionSettings = columnRange.fCompressionSettings;
         } else if (fCompressionSettings != columnRange.fCompressionSettings) {
            // Note that currently all clusters and columns are compressed with the same settings and it is not yet
            // possible to do otherwise. This measn that currently, this exception should never be thrown, but this
            // could change in the future.
            throw RException(R__FAIL("compression setting mismatch between column ranges (" +
                                     std::to_string(fCompressionSettings) + " vs " +
                                     std::to_string(columnRange.fCompressionSettings) + ")"));
         }

         const auto &pageRange = clusterDescriptor.GetPageRange(colId);

         for (const auto &page : pageRange.fPageInfos) {
            compressedSize += page.fLocator.fBytesOnStorage;
            fCompressedSize += page.fLocator.fBytesOnStorage;
            fUncompressedSize += page.fNElements * elemSize;
         }
      }

      fColumnInfo.emplace(colId, RColumnInfo(colDesc, compressedSize, elemSize, nElems));
   }
}

ROOT::Experimental::RNTupleInspector::RFieldTreeInfo
ROOT::Experimental::RNTupleInspector::CollectFieldTreeInfo(DescriptorId_t fieldId)
{
   std::uint64_t compressedSize = 0;
   std::uint64_t uncompressedSize = 0;

   for (const auto &colDescriptor : fDescriptor->GetColumnIterable(fieldId)) {
      auto colInfo = GetColumnInfo(colDescriptor.GetPhysicalId());
      compressedSize += colInfo.GetCompressedSize();
      uncompressedSize += colInfo.GetUncompressedSize();
   }

   for (const auto &subFieldDescriptor : fDescriptor->GetFieldIterable(fieldId)) {
      DescriptorId_t subFieldId = subFieldDescriptor.GetId();

      auto subFieldInfo = CollectFieldTreeInfo(subFieldId);

      compressedSize += subFieldInfo.GetCompressedSize();
      uncompressedSize += subFieldInfo.GetUncompressedSize();
   }

   auto fieldInfo = RFieldTreeInfo(fDescriptor->GetFieldDescriptor(fieldId), compressedSize, uncompressedSize);
   fFieldTreeInfo.emplace(fieldId, fieldInfo);
   return fieldInfo;
}

std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetColumnsByFieldId(DescriptorId_t fieldId) const
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
   inspector->CollectFieldTreeInfo(inspector->GetDescriptor()->GetFieldZeroId());

   return inspector;
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(ROOT::Experimental::RNTuple *sourceNTuple)
{
   if (!sourceNTuple) {
      throw RException(R__FAIL("provided RNTuple is null"));
   }

   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource = sourceNTuple->MakePageSource();

   return ROOT::Experimental::RNTupleInspector::Create(std::move(pageSource));
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(std::string_view ntupleName, std::string_view sourceFileName)
{
   auto sourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!sourceFile || sourceFile->IsZombie()) {
      throw RException(R__FAIL("cannot open source file " + std::string(sourceFileName)));
   }
   auto ntuple = std::unique_ptr<ROOT::Experimental::RNTuple>(
      sourceFile->Get<ROOT::Experimental::RNTuple>(std::string(ntupleName).c_str()));
   if (!ntuple) {
      throw RException(
         R__FAIL("cannot read RNTuple " + std::string(ntupleName) + " from " + std::string(sourceFileName)));
   }

   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector(ntuple->MakePageSource()));
   inspector->fSourceFile = std::move(sourceFile);

   inspector->CollectColumnInfo();
   inspector->CollectFieldTreeInfo(inspector->GetDescriptor()->GetFieldZeroId());

   return inspector;
}

std::string ROOT::Experimental::RNTupleInspector::GetCompressionSettingsAsString() const
{
   int algorithm = fCompressionSettings / 100;
   int level = fCompressionSettings - (algorithm * 100);

   return RCompressionSetting::AlgorithmToString(static_cast<RCompressionSetting::EAlgorithm::EValues>(algorithm)) +
          " (level " + std::to_string(level) + ")";
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RNTupleInspector::RColumnInfo &
ROOT::Experimental::RNTupleInspector::GetColumnInfo(DescriptorId_t physicalColumnId) const
{
   if (physicalColumnId > fDescriptor->GetNPhysicalColumns()) {
      throw RException(R__FAIL("No column with physical ID " + std::to_string(physicalColumnId) + " present"));
   }

   return fColumnInfo.at(physicalColumnId);
}

size_t ROOT::Experimental::RNTupleInspector::GetColumnCountByType(ROOT::Experimental::EColumnType colType) const
{
   size_t typeCount = 0;

   for (auto &[colId, colInfo] : fColumnInfo) {
      if (colInfo.GetType() == colType) {
         ++typeCount;
      }
   }

   return typeCount;
}

const std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetColumnsByType(ROOT::Experimental::EColumnType colType)
{
   std::vector<DescriptorId_t> colIds;

   for (const auto &[colId, colInfo] : fColumnInfo) {
      if (colInfo.GetType() == colType)
         colIds.emplace_back(colId);
   }

   return colIds;
}

void ROOT::Experimental::RNTupleInspector::PrintColumnTypeInfo(ENTupleInspectorPrintFormat format, std::ostream &output)
{
   struct ColumnTypeInfo {
      std::uint32_t count;
      std::uint64_t nElems, compressedSize, uncompressedSize;

      void operator+=(const RColumnInfo &colInfo)
      {
         this->count++;
         this->nElems += colInfo.GetNElements();
         this->compressedSize += colInfo.GetCompressedSize();
         this->uncompressedSize += colInfo.GetUncompressedSize();
      }
   };

   std::map<EColumnType, ColumnTypeInfo> colTypeInfo;

   for (const auto &[colId, colInfo] : fColumnInfo) {
      colTypeInfo[colInfo.GetType()] += colInfo;
   }

   switch (format) {
   case ENTupleInspectorPrintFormat::kTable:
      output << " column type    | count   | # elements      | compressed bytes  | uncompressed bytes\n"
             << "----------------|---------|-----------------|-------------------|--------------------" << std::endl;
      for (const auto &[colType, typeInfo] : colTypeInfo) {
         output << std::setw(15) << Detail::RColumnElementBase::GetTypeName(colType) << " |" << std::setw(8)
                << typeInfo.count << " |" << std::setw(16) << typeInfo.nElems << " |" << std::setw(18)
                << typeInfo.compressedSize << " |" << std::setw(18) << typeInfo.uncompressedSize << " " << std::endl;
      }
      break;
   case ENTupleInspectorPrintFormat::kCSV:
      output << "columnType,count,nElements,compressedSize,uncompressedSize" << std::endl;
      for (const auto &[colType, typeInfo] : colTypeInfo) {
         output << Detail::RColumnElementBase::GetTypeName(colType) << "," << typeInfo.count << "," << typeInfo.nElems
                << "," << typeInfo.compressedSize << "," << typeInfo.uncompressedSize << std::endl;
      }
      break;
   default: throw RException(R__FAIL("Invalid print format"));
   }
}

std::unique_ptr<TH1D>
ROOT::Experimental::RNTupleInspector::GetColumnTypeInfoAsHist(ROOT::Experimental::ENTupleInspectorHist histKind,
                                                              std::string_view histName, std::string_view histTitle)
{
   if (histName == "") {
      switch (histKind) {
      case ENTupleInspectorHist::kCount: histName = "colTypeCountHist"; break;
      case ENTupleInspectorHist::kNElems: histName = "colTypeElemCountHist"; break;
      case ENTupleInspectorHist::kCompressedSize: histName = "colTypeCompSizeHist"; break;
      case ENTupleInspectorHist::kUncompressedSize: histName = "colTypeUncompSizeHist"; break;
      default: throw RException(R__FAIL("Unknown histogram type"));
      }
   }

   if (histTitle == "") {
      switch (histKind) {
      case ENTupleInspectorHist::kCount: histTitle = "Column count by type"; break;
      case ENTupleInspectorHist::kNElems: histTitle = "Number of elements by column type"; break;
      case ENTupleInspectorHist::kCompressedSize: histTitle = "Compressed size by column type"; break;
      case ENTupleInspectorHist::kUncompressedSize: histTitle = "Uncompressed size by column type"; break;
      default: throw RException(R__FAIL("Unknown histogram type"));
      }
   }

   auto hist = std::make_unique<TH1D>(std::string(histName).c_str(), std::string(histTitle).c_str(), 1, 0, 1);

   double data;
   for (const auto &[colId, colInfo] : fColumnInfo) {
      switch (histKind) {
      case ENTupleInspectorHist::kCount: data = 1.; break;
      case ENTupleInspectorHist::kNElems: data = colInfo.GetNElements(); break;
      case ENTupleInspectorHist::kCompressedSize: data = colInfo.GetCompressedSize(); break;
      case ENTupleInspectorHist::kUncompressedSize: data = colInfo.GetUncompressedSize(); break;
      default: throw RException(R__FAIL("Unknown histogram type"));
      }

      hist->AddBinContent(hist->GetXaxis()->FindBin(Detail::RColumnElementBase::GetTypeName(colInfo.GetType()).c_str()),
                          data);
   }

   return hist;
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RNTupleInspector::RFieldTreeInfo &
ROOT::Experimental::RNTupleInspector::GetFieldTreeInfo(DescriptorId_t fieldId) const
{
   if (fieldId >= fDescriptor->GetNFields()) {
      throw RException(R__FAIL("No field with ID " + std::to_string(fieldId) + " present"));
   }

   return fFieldTreeInfo.at(fieldId);
}

const ROOT::Experimental::RNTupleInspector::RFieldTreeInfo &
ROOT::Experimental::RNTupleInspector::GetFieldTreeInfo(std::string_view fieldName) const
{
   DescriptorId_t fieldId = fDescriptor->FindFieldId(fieldName);

   if (fieldId == kInvalidDescriptorId) {
      throw RException(R__FAIL("Could not find field `" + std::string(fieldName) + "`"));
   }

   return GetFieldTreeInfo(fieldId);
}

size_t ROOT::Experimental::RNTupleInspector::GetFieldCountByType(const std::regex &typeNamePattern,
                                                                 bool includeSubFields) const
{
   size_t typeCount = 0;

   for (auto &[fldId, fldInfo] : fFieldTreeInfo) {
      if (!includeSubFields && fldInfo.GetDescriptor().GetParentId() != fDescriptor->GetFieldZeroId()) {
         continue;
      }

      if (std::regex_match(fldInfo.GetDescriptor().GetTypeName(), typeNamePattern)) {
         typeCount++;
      }
   }

   return typeCount;
}

const std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetFieldsByName(const std::regex &fieldNamePattern, bool searchInSubFields) const
{
   std::vector<DescriptorId_t> fieldIds;

   for (auto &[fldId, fldInfo] : fFieldTreeInfo) {

      if (!searchInSubFields && fldInfo.GetDescriptor().GetParentId() != fDescriptor->GetFieldZeroId()) {
         continue;
      }

      if (std::regex_match(fldInfo.GetDescriptor().GetFieldName(), fieldNamePattern)) {
         fieldIds.emplace_back(fldId);
      }
   }

   return fieldIds;
}
