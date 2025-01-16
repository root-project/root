/// \file RNTupleDescriptorFmt.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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
   std::uint32_t fNPages = 0;
   std::uint32_t fNEntries = 0;
   std::uint32_t fNBytesOnStorage = 0;
   std::uint32_t fNBytesInMemory = 0;

   bool operator==(const ClusterInfo &other) const { return fFirstEntry == other.fFirstEntry; }

   bool operator<(const ClusterInfo &other) const { return fFirstEntry < other.fFirstEntry; }
};

struct ColumnInfo {
   ROOT::Experimental::DescriptorId_t fPhysicalColumnId = 0;
   ROOT::Experimental::DescriptorId_t fLogicalColumnId = 0;
   ROOT::Experimental::DescriptorId_t fFieldId = 0;
   std::uint64_t fNElements = 0;
   std::uint64_t fNPages = 0;
   std::uint64_t fNBytesOnStorage = 0;
   std::uint32_t fElementSize = 0;
   std::uint32_t fColumnIndex = 0;
   std::uint16_t fRepresentationIndex = 0;
   ROOT::Experimental::ENTupleColumnType fType;
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

std::string
GetFieldName(ROOT::Experimental::DescriptorId_t fieldId, const ROOT::Experimental::RNTupleDescriptor &ntupleDesc)
{
   const auto &fieldDesc = ntupleDesc.GetFieldDescriptor(fieldId);
   if (fieldDesc.GetParentId() == ROOT::Experimental::kInvalidDescriptorId)
      return fieldDesc.GetFieldName();
   return GetFieldName(fieldDesc.GetParentId(), ntupleDesc) + "." + fieldDesc.GetFieldName();
}

std::string GetFieldDescription(ROOT::Experimental::DescriptorId_t fFieldId,
                                const ROOT::Experimental::RNTupleDescriptor &ntupleDesc)
{
   const auto &fieldDesc = ntupleDesc.GetFieldDescriptor(fFieldId);
   return fieldDesc.GetFieldDescription();
}

} // anonymous namespace

void ROOT::Experimental::RNTupleDescriptor::PrintInfo(std::ostream &output) const
{
   std::vector<ColumnInfo> columns;
   std::vector<ClusterInfo> clusters;
   std::unordered_map<DescriptorId_t, unsigned int> cluster2Idx;
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
   std::uint64_t nPages = 0;
   int compression = -1;
   for (const auto &column : fColumnDescriptors) {
      // Alias columns (columns of projected fields) don't contribute to the storage consumption. Count them
      // but don't add the the page sizes to the overall volume.
      if (column.second.IsAliasColumn())
         continue;

      // We generate the default memory representation for the given column type in order
      // to report the size _in memory_ of column elements
      auto elementSize = Internal::RColumnElementBase::Generate(column.second.GetType())->GetSize();

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
         if (columnRange.fIsSuppressed)
            continue;

         info.fNElements += columnRange.fNElements;
         if (compression == -1 && columnRange.fCompressionSettings) {
            compression = *columnRange.fCompressionSettings;
         }
         const auto &pageRange = cluster.second.GetPageRange(column.second.GetPhysicalId());
         auto idx = cluster2Idx[cluster.first];
         for (const auto &page : pageRange.fPageInfos) {
            nBytesOnStorage += page.fLocator.GetNBytesOnStorage();
            nBytesInMemory += page.fNElements * elementSize;
            clusters[idx].fNBytesOnStorage += page.fLocator.GetNBytesOnStorage();
            clusters[idx].fNBytesInMemory += page.fNElements * elementSize;
            ++clusters[idx].fNPages;
            info.fNBytesOnStorage += page.fLocator.GetNBytesOnStorage();
            ++info.fNPages;
            ++nPages;
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
   output << "  # Pages:          " << nPages << "\n";
   output << "  # Clusters:       " << GetNClusters() << "\n";
   output << "  Size on storage:  " << nBytesOnStorage << " B" << "\n";
   output << "  Compression rate: " << std::fixed << std::setprecision(2)
          << float(nBytesInMemory) / float(nBytesOnStorage) << "\n";
   output << "  Header size:      " << headerSize << " B"
          << "\n";
   output << "  Footer size:      " << footerSize << " B"
          << "\n";
   output << "  Meta-data / data: " << std::fixed << std::setprecision(3)
          << float(headerSize + footerSize) / float(nBytesOnStorage) << "\n";
   output << "------------------------------------------------------------\n";
   output << "CLUSTER DETAILS\n";
   output << "------------------------------------------------------------" << std::endl;

   std::sort(clusters.begin(), clusters.end());
   for (unsigned int i = 0; i < clusters.size(); ++i) {
      output << "  # " << std::setw(5) << i << "   Entry range:     [" << clusters[i].fFirstEntry << ".."
             << clusters[i].fFirstEntry + clusters[i].fNEntries - 1 << "]  --  " << clusters[i].fNEntries << "\n";
      output << "         " << "   # Pages:         " << clusters[i].fNPages << "\n";
      output << "         " << "   Size on storage: " << clusters[i].fNBytesOnStorage << " B\n";
      output << "         " << "   Compression:     " << std::fixed << std::setprecision(2)
             << float(clusters[i].fNBytesInMemory) / float(float(clusters[i].fNBytesOnStorage)) << std::endl;
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
      auto avgPageSize = (col.fNPages == 0) ? 0 : (col.fNBytesOnStorage / col.fNPages);
      auto avgElementsPerPage = (col.fNPages == 0) ? 0 : (col.fNElements / col.fNPages);
      std::string nameAndType = std::string("  ") + col.fFieldName + " [#" + std::to_string(col.fColumnIndex);
      if (col.fRepresentationIndex > 0)
         nameAndType += " / R." + std::to_string(col.fRepresentationIndex);
      nameAndType += "]  --  " + std::string{Internal::RColumnElementBase::GetColumnTypeName(col.fType)};
      std::string id = std::string("{id:") + std::to_string(col.fLogicalColumnId) + "}";
      if (col.fLogicalColumnId != col.fPhysicalColumnId)
         id += " --alias--> " + std::to_string(col.fPhysicalColumnId);
      output << nameAndType << std::setw(60 - nameAndType.length()) << id << "\n";
      if (!col.fFieldDescription.empty())
         output << "    Description:         " << col.fFieldDescription << "\n";
      output << "    # Elements:          " << col.fNElements << "\n";
      output << "    # Pages:             " << col.fNPages << "\n";
      output << "    Avg elements / page: " << avgElementsPerPage << "\n";
      output << "    Avg page size:       " << avgPageSize << " B\n";
      output << "    Size on storage:     " << col.fNBytesOnStorage << " B\n";
      output << "    Compression:         " << std::fixed << std::setprecision(2)
             << float(col.fElementSize * col.fNElements) / float(col.fNBytesOnStorage) << "\n";
      output << "............................................................" << std::endl;
   }
}
