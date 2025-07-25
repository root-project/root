/// \file RNTupleDescriptorFmt.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-25

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace {

struct ClusterInfo {
   std::uint64_t fFirstEntry = 0;
   std::uint64_t fNPhysicalPages = 0;
   std::uint64_t fNAliasedPages = 0;
   std::uint32_t fNEntries = 0;
   std::uint32_t fNBytesOnStorage = 0;
   std::uint32_t fNBytesInMemory = 0;

   bool operator==(const ClusterInfo &other) const { return fFirstEntry == other.fFirstEntry; }

   bool operator<(const ClusterInfo &other) const { return fFirstEntry < other.fFirstEntry; }
};

struct ColumnInfo {
   ROOT::DescriptorId_t fPhysicalColumnId = 0;
   ROOT::DescriptorId_t fLogicalColumnId = 0;
   ROOT::DescriptorId_t fFieldId = 0;
   std::uint64_t fNElements = 0;
   std::uint64_t fNPhysicalPages = 0;
   std::uint64_t fNAliasedPages = 0;
   std::uint64_t fNBytesOnStorage = 0;
   std::uint32_t fElementSize = 0;
   std::uint32_t fColumnIndex = 0;
   std::uint16_t fRepresentationIndex = 0;
   ROOT::ENTupleColumnType fType;
   std::string fFieldName;
   std::string fFieldDescription;

   bool operator<(const ColumnInfo &other) const
   {
      if (fFieldName == other.fFieldName) {
         if (fRepresentationIndex == other.fRepresentationIndex)
            return fColumnIndex < other.fColumnIndex;
         return fRepresentationIndex < other.fRepresentationIndex;
      }
      return fFieldName < other.fFieldName;
   }
};

std::string GetFieldName(ROOT::DescriptorId_t fieldId, const ROOT::RNTupleDescriptor &ntupleDesc)
{
   const auto &fieldDesc = ntupleDesc.GetFieldDescriptor(fieldId);
   if (fieldDesc.GetParentId() == ROOT::kInvalidDescriptorId)
      return fieldDesc.GetFieldName();
   return GetFieldName(fieldDesc.GetParentId(), ntupleDesc) + "." + fieldDesc.GetFieldName();
}

std::string GetFieldDescription(ROOT::DescriptorId_t fFieldId, const ROOT::RNTupleDescriptor &ntupleDesc)
{
   const auto &fieldDesc = ntupleDesc.GetFieldDescriptor(fFieldId);
   return fieldDesc.GetFieldDescription();
}

} // anonymous namespace

void ROOT::RNTupleDescriptor::PrintInfo(std::ostream &output) const
{
   std::vector<ColumnInfo> columns;
   std::vector<ClusterInfo> clusters;
   std::unordered_map<ROOT::DescriptorId_t, unsigned int> cluster2Idx;
   clusters.reserve(fClusterDescriptors.size());
   for (const auto &cluster : fClusterDescriptors) {
      ClusterInfo info;
      info.fFirstEntry = cluster.second.GetFirstEntryIndex();
      info.fNEntries = cluster.second.GetNEntries();
      cluster2Idx[cluster.first] = clusters.size();
      clusters.emplace_back(info);
   }

   std::uint64_t nBytesOnStorage = 0;
   std::uint64_t nBytesInMemory = 0;
   std::uint64_t nPhysicalPages = 0;
   std::uint64_t nAliasedPages = 0;
   std::unordered_set<std::uint64_t> seenPages{};
   int compression = -1;
   for (const auto &column : fColumnDescriptors) {
      // Alias columns (columns of projected fields) don't contribute to the storage consumption. Count them
      // but don't add the the page sizes to the overall volume.
      if (column.second.IsAliasColumn())
         continue;

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements
      auto elementSize = ROOT::Internal::RColumnElementBase::Generate(column.second.GetType())->GetSize();

      ColumnInfo info;
      info.fPhysicalColumnId = column.second.GetPhysicalId();
      info.fLogicalColumnId = column.second.GetLogicalId();
      info.fFieldId = column.second.GetFieldId();
      info.fColumnIndex = column.second.GetIndex();
      info.fElementSize = elementSize;
      info.fType = column.second.GetType();
      info.fRepresentationIndex = column.second.GetRepresentationIndex();

      for (const auto &cluster : fClusterDescriptors) {
         auto columnRange = cluster.second.GetColumnRange(column.second.GetPhysicalId());
         if (columnRange.IsSuppressed())
            continue;

         info.fNElements += columnRange.GetNElements();
         if (compression == -1 && columnRange.GetCompressionSettings()) {
            compression = *columnRange.GetCompressionSettings();
         }
         const auto &pageRange = cluster.second.GetPageRange(column.second.GetPhysicalId());
         auto idx = cluster2Idx[cluster.first];
         std::uint64_t locatorOffset;
         for (const auto &page : pageRange.GetPageInfos()) {
            locatorOffset = page.GetLocator().GetType() == ROOT::RNTupleLocator::ELocatorType::kTypeDAOS
                               ? page.GetLocator().GetPosition<RNTupleLocatorObject64>().GetLocation()
                               : page.GetLocator().GetPosition<std::uint64_t>();
            auto [_, pageAdded] = seenPages.emplace(locatorOffset);
            if (pageAdded) {
               nBytesOnStorage += page.GetLocator().GetNBytesOnStorage();
               nBytesInMemory += page.GetNElements() * elementSize;
               clusters[idx].fNBytesOnStorage += page.GetLocator().GetNBytesOnStorage();
               clusters[idx].fNBytesInMemory += page.GetNElements() * elementSize;
               ++clusters[idx].fNPhysicalPages;
               info.fNBytesOnStorage += page.GetLocator().GetNBytesOnStorage();
               ++info.fNPhysicalPages;
               ++nPhysicalPages;
            } else {
               ++clusters[idx].fNAliasedPages;
               ++info.fNAliasedPages;
               ++nAliasedPages;
            }
         }
      }
      columns.emplace_back(info);
   }
   auto headerSize = GetOnDiskHeaderSize();
   auto footerSize = GetOnDiskFooterSize();
   output << "============================================================\n";
   output << "NTUPLE:      " << GetName() << "\n";
   output << "Compression: " << compression << "\n";
   output << "------------------------------------------------------------\n";
   output << "  # Entries:        " << GetNEntries() << "\n";
   output << "  # Fields:         " << GetNFields() << "\n";
   output << "  # Columns:        " << GetNPhysicalColumns() << "\n";
   output << "  # Alias Columns:  " << GetNLogicalColumns() - GetNPhysicalColumns() << "\n";
   output << "  # Physical Pages: " << nPhysicalPages << "\n";
   output << "  # Aliased Pages:  " << nAliasedPages << "\n";
   output << "  # Clusters:       " << GetNClusters() << "\n";
   output << "  Size on storage:  " << nBytesOnStorage << " B" << "\n";
   output << "  Compression rate: " << std::fixed << std::setprecision(2)
          << float(nBytesInMemory) / float(nBytesOnStorage) << "\n";
   output << "  Header size:      " << headerSize << " B"
          << "\n";
   output << "  Footer size:      " << footerSize << " B"
          << "\n";
   output << "  Metadata / data:  " << std::fixed << std::setprecision(3)
          << float(headerSize + footerSize) / float(nBytesOnStorage) << "\n";
   output << "------------------------------------------------------------\n";
   output << "CLUSTER DETAILS\n";
   output << "------------------------------------------------------------" << std::endl;

   std::sort(clusters.begin(), clusters.end());
   for (unsigned int i = 0; i < clusters.size(); ++i) {
      output << "  # " << std::setw(5) << i << "   Entry range:      [" << clusters[i].fFirstEntry << ".."
             << clusters[i].fFirstEntry + clusters[i].fNEntries - 1 << "]  --  " << clusters[i].fNEntries << "\n";
      output << "         " << "   # Physical Pages: " << clusters[i].fNPhysicalPages << "\n";
      output << "         " << "   # Aliased Pages:  " << clusters[i].fNAliasedPages << "\n";
      output << "         " << "   Size on storage:  " << clusters[i].fNBytesOnStorage << " B\n";
      output << "         " << "   Compression:      " << std::fixed << std::setprecision(2);
      if (clusters[i].fNPhysicalPages > 0)
         output << float(clusters[i].fNBytesInMemory) / float(float(clusters[i].fNBytesOnStorage)) << std::endl;
      else
         output << "N/A" << std::endl;
   }

   output << "------------------------------------------------------------\n";
   output << "COLUMN DETAILS\n";
   output << "------------------------------------------------------------\n";
   for (auto &col : columns) {
      col.fFieldName = GetFieldName(col.fFieldId, *this).substr(1);
      col.fFieldDescription = GetFieldDescription(col.fFieldId, *this);
   }
   std::sort(columns.begin(), columns.end());
   for (const auto &col : columns) {
      auto avgPageSize = (col.fNPhysicalPages == 0) ? 0 : (col.fNBytesOnStorage / col.fNPhysicalPages);
      auto avgElementsPerPage = (col.fNPhysicalPages == 0) ? 0 : (col.fNElements / col.fNPhysicalPages);
      std::string nameAndType = std::string("  ") + col.fFieldName + " [#" + std::to_string(col.fColumnIndex);
      if (col.fRepresentationIndex > 0)
         nameAndType += " / R." + std::to_string(col.fRepresentationIndex);
      nameAndType += "]  --  " + std::string{ROOT::Internal::RColumnElementBase::GetColumnTypeName(col.fType)};
      std::string id = std::string("{id:") + std::to_string(col.fLogicalColumnId) + "}";
      if (col.fLogicalColumnId != col.fPhysicalColumnId)
         id += " --alias--> " + std::to_string(col.fPhysicalColumnId);
      output << nameAndType << std::setw(60 - nameAndType.length()) << id << "\n";
      if (!col.fFieldDescription.empty())
         output << "    Description:         " << col.fFieldDescription << "\n";
      output << "    # Elements:          " << col.fNElements << "\n";
      output << "    # Physical Pages:    " << col.fNPhysicalPages << "\n";
      output << "    # Aliased Pages:     " << col.fNAliasedPages << "\n";
      output << "    Avg elements / page: " << avgElementsPerPage << "\n";
      output << "    Avg page size:       " << avgPageSize << " B\n";
      output << "    Size on storage:     " << col.fNBytesOnStorage << " B\n";
      output << "    Compression:         " << std::fixed << std::setprecision(2);
      if (col.fNPhysicalPages > 0)
         output << float(col.fElementSize * col.fNElements) / float(col.fNBytesOnStorage) << std::endl;
      else
         output << "N/A" << std::endl;
      output << "............................................................" << std::endl;
   }
}
