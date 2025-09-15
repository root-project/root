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

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RPageStorageFile.hxx>
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

using ROOT::Internal::RColumnElementBase;

ROOT::Experimental::RNTupleInspector::RNTupleInspector(std::unique_ptr<ROOT::Internal::RPageSource> pageSource)
   : fPageSource(std::move(pageSource))
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   fDescriptor = descriptorGuard->Clone();

   CollectColumnInfo();
   CollectFieldTreeInfo(fDescriptor.GetFieldZeroId());
}

// NOTE: outlined to avoid including RPageStorage in the header
ROOT::Experimental::RNTupleInspector::~RNTupleInspector() = default;

void ROOT::Experimental::RNTupleInspector::CollectColumnInfo()
{
   fCompressedSize = 0;
   fUncompressedSize = 0;

   for (const auto &colDesc : fDescriptor.GetColumnIterable()) {
      if (colDesc.IsAliasColumn())
         continue;

      auto colId = colDesc.GetPhysicalId();

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements.
      std::uint32_t elemSize = RColumnElementBase::Generate(colDesc.GetType())->GetSize();
      std::uint64_t nElems = 0;
      std::vector<std::uint64_t> compressedPageSizes{};

      for (const auto &clusterDescriptor : fDescriptor.GetClusterIterable()) {
         if (!clusterDescriptor.ContainsColumn(colId)) {
            continue;
         }

         auto columnRange = clusterDescriptor.GetColumnRange(colId);
         if (columnRange.IsSuppressed())
            continue;

         nElems += columnRange.GetNElements();

         if (!fCompressionSettings && columnRange.GetCompressionSettings()) {
            fCompressionSettings = *columnRange.GetCompressionSettings();
         } else if (fCompressionSettings && columnRange.GetCompressionSettings() &&
                    (*fCompressionSettings != *columnRange.GetCompressionSettings())) {
            // Note that currently all clusters and columns are compressed with the same settings and it is not yet
            // possible to do otherwise. This means that currently, this exception should never be thrown, but this
            // could change in the future.
            throw RException(R__FAIL("compression setting mismatch between column ranges (" +
                                     std::to_string(*fCompressionSettings) + " vs " +
                                     std::to_string(*columnRange.GetCompressionSettings()) +
                                     ") for column with physical ID " + std::to_string(colId)));
         }

         const auto &pageRange = clusterDescriptor.GetPageRange(colId);

         for (const auto &page : pageRange.GetPageInfos()) {
            compressedPageSizes.emplace_back(page.GetLocator().GetNBytesOnStorage());
            fUncompressedSize += page.GetNElements() * elemSize;
         }
      }

      fCompressedSize +=
         std::accumulate(compressedPageSizes.begin(), compressedPageSizes.end(), static_cast<std::uint64_t>(0));
      fColumnInfo.emplace(colId, RColumnInspector(colDesc, compressedPageSizes, elemSize, nElems));
   }
}

ROOT::Experimental::RNTupleInspector::RFieldTreeInspector
ROOT::Experimental::RNTupleInspector::CollectFieldTreeInfo(ROOT::DescriptorId_t fieldId)
{
   std::uint64_t compressedSize = 0;
   std::uint64_t uncompressedSize = 0;

   for (const auto &colDescriptor : fDescriptor.GetColumnIterable(fieldId)) {
      auto colInfo = GetColumnInspector(colDescriptor.GetPhysicalId());
      compressedSize += colInfo.GetCompressedSize();
      uncompressedSize += colInfo.GetUncompressedSize();
   }

   for (const auto &subFieldDescriptor : fDescriptor.GetFieldIterable(fieldId)) {
      auto subFieldId = subFieldDescriptor.GetId();

      auto subFieldInfo = CollectFieldTreeInfo(subFieldId);

      compressedSize += subFieldInfo.GetCompressedSize();
      uncompressedSize += subFieldInfo.GetUncompressedSize();
   }

   auto fieldInfo = RFieldTreeInspector(fDescriptor.GetFieldDescriptor(fieldId), compressedSize, uncompressedSize);
   fFieldTreeInfo.emplace(fieldId, fieldInfo);
   return fieldInfo;
}

std::vector<ROOT::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetAllColumnsOfField(ROOT::DescriptorId_t fieldId) const
{
   std::vector<ROOT::DescriptorId_t> colIds;
   std::deque<ROOT::DescriptorId_t> fieldIdQueue{fieldId};

   while (!fieldIdQueue.empty()) {
      auto currId = fieldIdQueue.front();
      fieldIdQueue.pop_front();

      for (const auto &col : fDescriptor.GetColumnIterable(currId)) {
         if (col.IsAliasColumn()) {
            continue;
         }

         colIds.emplace_back(col.GetPhysicalId());
      }

      for (const auto &fld : fDescriptor.GetFieldIterable(currId)) {
         fieldIdQueue.push_back(fld.GetId());
      }
   }

   return colIds;
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(const ROOT::RNTuple &sourceNTuple)
{
   auto pageSource = ROOT::Internal::RPageSourceFile::CreateFromAnchor(sourceNTuple);
   return std::unique_ptr<RNTupleInspector>(new RNTupleInspector(std::move(pageSource)));
}

std::unique_ptr<ROOT::Experimental::RNTupleInspector>
ROOT::Experimental::RNTupleInspector::Create(std::string_view ntupleName, std::string_view sourceFileName)
{
   auto pageSource = ROOT::Internal::RPageSource::Create(ntupleName, sourceFileName);
   return std::unique_ptr<RNTupleInspector>(new RNTupleInspector(std::move(pageSource)));
}

std::string ROOT::Experimental::RNTupleInspector::GetCompressionSettingsAsString() const
{
   if (!fCompressionSettings)
      return "unknown";

   int algorithm = *fCompressionSettings / 100;
   int level = *fCompressionSettings - (algorithm * 100);

   return RCompressionSetting::AlgorithmToString(static_cast<RCompressionSetting::EAlgorithm::EValues>(algorithm)) +
          " (level " + std::to_string(level) + ")";
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RNTupleInspector::RColumnInspector &
ROOT::Experimental::RNTupleInspector::GetColumnInspector(ROOT::DescriptorId_t physicalColumnId) const
{
   if (physicalColumnId > fDescriptor.GetNPhysicalColumns()) {
      throw RException(R__FAIL("No column with physical ID " + std::to_string(physicalColumnId) + " present"));
   }

   return fColumnInfo.at(physicalColumnId);
}

size_t ROOT::Experimental::RNTupleInspector::GetColumnCountByType(ROOT::ENTupleColumnType colType) const
{
   size_t typeCount = 0;

   for (auto &[colId, colInfo] : fColumnInfo) {
      if (colInfo.GetType() == colType) {
         ++typeCount;
      }
   }

   return typeCount;
}

std::vector<ROOT::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetColumnsByType(ROOT::ENTupleColumnType colType)
{
   std::vector<ROOT::DescriptorId_t> colIds;

   for (const auto &[colId, colInfo] : fColumnInfo) {
      if (colInfo.GetType() == colType)
         colIds.emplace_back(colId);
   }

   return colIds;
}

std::vector<ROOT::ENTupleColumnType> ROOT::Experimental::RNTupleInspector::GetColumnTypes()
{
   std::set<ROOT::ENTupleColumnType> colTypes;

   for (const auto &[colId, colInfo] : fColumnInfo) {
      colTypes.emplace(colInfo.GetType());
   }

   return std::vector(colTypes.begin(), colTypes.end());
}

void ROOT::Experimental::RNTupleInspector::PrintColumnTypeInfo(ENTupleInspectorPrintFormat format, std::ostream &output)
{
   struct ColumnTypeInfo {
      std::uint64_t nElems = 0;
      std::uint64_t compressedSize = 0;
      std::uint64_t uncompressedSize = 0;
      std::uint64_t nPages = 0;
      std::uint32_t count = 0;

      void operator+=(const RColumnInspector &colInfo)
      {
         this->count++;
         this->nElems += colInfo.GetNElements();
         this->compressedSize += colInfo.GetCompressedSize();
         this->uncompressedSize += colInfo.GetUncompressedSize();
         this->nPages += colInfo.GetNPages();
      }

      // Helper method to calculate compression factor
      float GetCompressionFactor() const
      {
         if (compressedSize == 0)
            return 1.0;
         return static_cast<float>(uncompressedSize) / static_cast<float>(compressedSize);
      }
   };

   std::map<ENTupleColumnType, ColumnTypeInfo> colTypeInfo;

   // Collect information for each column
   for (const auto &[colId, colInfo] : fColumnInfo) {
      colTypeInfo[colInfo.GetType()] += colInfo;
   }

   switch (format) {
   case ENTupleInspectorPrintFormat::kTable:
      output << " column type    | count   | # elements  | compressed bytes | uncompressed bytes | compression ratio | "
                "# pages \n"
             << "----------------|---------|-------------|------------------|--------------------|-------------------|-"
                "------"
             << std::endl;
      for (const auto &[colType, typeInfo] : colTypeInfo)
         output << std::setw(15) << RColumnElementBase::GetColumnTypeName(colType) << " |" << std::setw(8)
                << typeInfo.count << " |" << std::setw(12) << typeInfo.nElems << " |" << std::setw(17)
                << typeInfo.compressedSize << " |" << std::setw(19) << typeInfo.uncompressedSize << " |" << std::fixed
                << std::setprecision(3) << std::setw(18) << typeInfo.GetCompressionFactor() << " |" << std::setw(6)
                << typeInfo.nPages << " " << std::endl;
      break;
   case ENTupleInspectorPrintFormat::kCSV:
      output << "columnType,count,nElements,compressedSize,uncompressedSize,compressionFactor,nPages" << std::endl;
      for (const auto &[colType, typeInfo] : colTypeInfo) {
         output << RColumnElementBase::GetColumnTypeName(colType) << "," << typeInfo.count << "," << typeInfo.nElems
                << "," << typeInfo.compressedSize << "," << typeInfo.uncompressedSize << "," << std::fixed
                << std::setprecision(3) << typeInfo.GetCompressionFactor() << "," << typeInfo.nPages << std::endl;
      }
      break;
   default: R__ASSERT(false && "Invalid print format");
   }
}

std::unique_ptr<TH1D>
ROOT::Experimental::RNTupleInspector::GetColumnTypeInfoAsHist(ROOT::Experimental::ENTupleInspectorHist histKind,
                                                              std::string_view histName, std::string_view histTitle)
{
   if (histName.empty()) {
      switch (histKind) {
      case ENTupleInspectorHist::kCount: histName = "colTypeCountHist"; break;
      case ENTupleInspectorHist::kNElems: histName = "colTypeElemCountHist"; break;
      case ENTupleInspectorHist::kCompressedSize: histName = "colTypeCompSizeHist"; break;
      case ENTupleInspectorHist::kUncompressedSize: histName = "colTypeUncompSizeHist"; break;
      default: throw RException(R__FAIL("Unknown histogram type"));
      }
   }

   if (histTitle.empty()) {
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

      hist->AddBinContent(hist->GetXaxis()->FindBin(RColumnElementBase::GetColumnTypeName(colInfo.GetType())), data);
   }

   return hist;
}

std::unique_ptr<TH1D>
ROOT::Experimental::RNTupleInspector::GetPageSizeDistribution(ROOT::DescriptorId_t physicalColumnId,
                                                              std::string histName, std::string histTitle, size_t nBins)
{
   if (histTitle.empty())
      histTitle = "Page size distribution for column with ID " + std::to_string(physicalColumnId);

   return GetPageSizeDistribution({physicalColumnId}, histName, histTitle, nBins);
}

std::unique_ptr<TH1D> ROOT::Experimental::RNTupleInspector::GetPageSizeDistribution(ROOT::ENTupleColumnType colType,
                                                                                    std::string histName,
                                                                                    std::string histTitle, size_t nBins)
{
   if (histName.empty())
      histName = "pageSizeHistCol" + std::string{RColumnElementBase::GetColumnTypeName(colType)};
   if (histTitle.empty())
      histTitle =
         "Page size distribution for columns with type " + std::string{RColumnElementBase::GetColumnTypeName(colType)};

   auto perTypeHist = GetPageSizeDistribution({colType}, histName, histTitle, nBins);

   if (perTypeHist->GetNhists() < 1)
      return std::make_unique<TH1D>(histName.c_str(), histTitle.c_str(), 64, 0, 0);

   auto hist = std::unique_ptr<TH1D>(dynamic_cast<TH1D *>(perTypeHist->GetHists()->First()));

   hist->SetName(histName.c_str());
   hist->SetTitle(histTitle.c_str());
   hist->SetXTitle("Page size (B)");
   hist->SetYTitle("N_{pages}");
   return hist;
}

std::unique_ptr<TH1D>
ROOT::Experimental::RNTupleInspector::GetPageSizeDistribution(std::initializer_list<ROOT::DescriptorId_t> colIds,
                                                              std::string histName, std::string histTitle, size_t nBins)
{
   auto hist = std::make_unique<TH1D>();

   if (histName.empty())
      histName = "pageSizeHist";
   hist->SetName(histName.c_str());
   if (histTitle.empty())
      histTitle = "Page size distribution";
   hist->SetTitle(histTitle.c_str());
   hist->SetXTitle("Page size (B)");
   hist->SetYTitle("N_{pages}");

   std::vector<std::uint64_t> pageSizes;
   std::for_each(colIds.begin(), colIds.end(), [this, &pageSizes](const auto colId) {
      auto colInfo = GetColumnInspector(colId);
      pageSizes.insert(pageSizes.end(), colInfo.GetCompressedPageSizes().begin(),
                       colInfo.GetCompressedPageSizes().end());
   });

   if (!pageSizes.empty()) {
      auto histMinMax = std::minmax_element(pageSizes.begin(), pageSizes.end());
      hist->SetBins(nBins, *histMinMax.first,
                    *histMinMax.second + ((*histMinMax.second - *histMinMax.first) / static_cast<double>(nBins)));

      for (const auto pageSize : pageSizes) {
         hist->Fill(pageSize);
      }
   }

   return hist;
}

std::unique_ptr<THStack>
ROOT::Experimental::RNTupleInspector::GetPageSizeDistribution(std::initializer_list<ROOT::ENTupleColumnType> colTypes,
                                                              std::string histName, std::string histTitle, size_t nBins)
{
   if (histName.empty())
      histName = "pageSizeHist";
   if (histTitle.empty())
      histTitle = "Per-column type page size distribution";

   auto stackedHist = std::make_unique<THStack>(histName.c_str(), histTitle.c_str());

   double histMin = std::numeric_limits<double>::max();
   double histMax = 0;
   std::map<ROOT::ENTupleColumnType, std::vector<std::uint64_t>> pageSizes;

   std::vector<ROOT::ENTupleColumnType> colTypeVec = colTypes;
   if (std::empty(colTypes)) {
      colTypeVec = GetColumnTypes();
   }

   for (const auto colType : colTypeVec) {
      auto colIds = GetColumnsByType(colType);

      if (colIds.empty())
         continue;

      std::vector<std::uint64_t> pageSizesForColType;
      std::for_each(colIds.cbegin(), colIds.cend(), [this, &pageSizesForColType](const auto colId) {
         auto colInfo = GetColumnInspector(colId);
         pageSizesForColType.insert(pageSizesForColType.end(), colInfo.GetCompressedPageSizes().begin(),
                                    colInfo.GetCompressedPageSizes().end());
      });
      if (pageSizesForColType.empty())
         continue;

      pageSizes.emplace(colType, pageSizesForColType);

      auto histMinMax = std::minmax_element(pageSizesForColType.begin(), pageSizesForColType.end());
      histMin = std::min(histMin, static_cast<double>(*histMinMax.first));
      histMax = std::max(histMax, static_cast<double>(*histMinMax.second));
   }

   for (const auto &[colType, pageSizesForColType] : pageSizes) {
      auto hist = std::make_unique<TH1D>(
         TString::Format("%s%s", histName.c_str(), RColumnElementBase::GetColumnTypeName(colType)),
         RColumnElementBase::GetColumnTypeName(colType), nBins, histMin,
         histMax + ((histMax - histMin) / static_cast<double>(nBins)));

      for (const auto pageSize : pageSizesForColType) {
         hist->Fill(pageSize);
      }

      stackedHist->Add(hist.release());
   }

   return stackedHist;
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RNTupleInspector::RFieldTreeInspector &
ROOT::Experimental::RNTupleInspector::GetFieldTreeInspector(ROOT::DescriptorId_t fieldId) const
{
   if (fieldId >= fDescriptor.GetNFields()) {
      throw RException(R__FAIL("No field with ID " + std::to_string(fieldId) + " present"));
   }

   return fFieldTreeInfo.at(fieldId);
}

const ROOT::Experimental::RNTupleInspector::RFieldTreeInspector &
ROOT::Experimental::RNTupleInspector::GetFieldTreeInspector(std::string_view fieldName) const
{
   auto fieldId = fDescriptor.FindFieldId(fieldName);

   if (fieldId == kInvalidDescriptorId) {
      throw RException(R__FAIL("Could not find field `" + std::string(fieldName) + "`"));
   }

   return GetFieldTreeInspector(fieldId);
}

size_t ROOT::Experimental::RNTupleInspector::GetFieldCountByType(const std::regex &typeNamePattern,
                                                                 bool includeSubfields) const
{
   size_t typeCount = 0;

   for (auto &[fldId, fldInfo] : fFieldTreeInfo) {
      if (!includeSubfields && fldInfo.GetDescriptor().GetParentId() != fDescriptor.GetFieldZeroId()) {
         continue;
      }

      if (std::regex_match(fldInfo.GetDescriptor().GetTypeName(), typeNamePattern)) {
         typeCount++;
      }
   }

   return typeCount;
}

std::vector<ROOT::DescriptorId_t>
ROOT::Experimental::RNTupleInspector::GetFieldsByName(const std::regex &fieldNamePattern, bool searchInSubfields) const
{
   std::vector<ROOT::DescriptorId_t> fieldIds;

   for (auto &[fldId, fldInfo] : fFieldTreeInfo) {

      if (!searchInSubfields && fldInfo.GetDescriptor().GetParentId() != fDescriptor.GetFieldZeroId()) {
         continue;
      }

      if (std::regex_match(fldInfo.GetDescriptor().GetFieldName(), fieldNamePattern)) {
         fieldIds.emplace_back(fldId);
      }
   }

   return fieldIds;
}

void ROOT::Experimental::RNTupleInspector::PrintFieldTreeAsDot(const ROOT::RFieldDescriptor &fieldDescriptor,
                                                               std::ostream &output) const
{
   const auto &tupleDescriptor = GetDescriptor();
   const bool isZeroField = fieldDescriptor.GetParentId() == ROOT::kInvalidDescriptorId;
   if (isZeroField) {
      output << "digraph D {\n";
      output << "node[shape=box]\n";
   }
   const std::string &nodeId = (isZeroField) ? "0" : std::to_string(fieldDescriptor.GetId() + 1);
   const std::string &description = fieldDescriptor.GetFieldDescription();
   const std::uint32_t &version = fieldDescriptor.GetFieldVersion();

   auto htmlEscape = [&](const std::string &in) -> std::string {
      std::string out;
      out.reserve(in.size());
      for (const char &c : in) {
         switch (c) {
         case '&': out += "&amp;"; break;
         case '<': out += "&lt;"; break;
         case '>': out += "&gt;"; break;
         case '\"': out += "&quot;"; break;
         case '\'': out += "&#39;"; break;
         default: out += c; break;
         }
      }
      return out;
   };

   output << nodeId << "[label=<";
   if (!isZeroField) {
      output << "<b>Name: </b>" << htmlEscape(fieldDescriptor.GetFieldName()) << "<br></br>";
      output << "<b>Type: </b>" << htmlEscape(fieldDescriptor.GetTypeName()) << "<br></br>";
      output << "<b>ID: </b>" << std::to_string(fieldDescriptor.GetId()) << "<br></br>";
      if (description != "")
         output << "<b>Description: </b>" << htmlEscape(description) << "<br></br>";
      if (version != 0)
         output << "<b>Version: </b>" << version << "<br></br>";
   } else
      output << "<b>RFieldZero</b>";
   output << ">]\n";
   for (const auto &childFieldId : fieldDescriptor.GetLinkIds()) {
      const auto &childFieldDescriptor = tupleDescriptor.GetFieldDescriptor(childFieldId);
      output << nodeId + "->" + std::to_string(childFieldDescriptor.GetId() + 1) + "\n";
      PrintFieldTreeAsDot(childFieldDescriptor, output);
   }
   if (isZeroField)
      output << "}";
}
