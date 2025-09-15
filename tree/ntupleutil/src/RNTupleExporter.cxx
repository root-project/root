/// \file RNTupleExporter.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleExporter.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RLogger.hxx>
#include <fstream>
#include <sstream>

namespace ROOT::Experimental::Internal {

namespace {

ROOT::RLogChannel &RNTupleExporterLog()
{
   static RLogChannel sLog("ROOT.RNTupleExporter");
   return sLog;
}

struct RColumnExportInfo {
   const ROOT::RColumnDescriptor *fColDesc;
   const ROOT::RFieldDescriptor *fFieldDesc;
   std::string fQualifiedName;

   RColumnExportInfo(const ROOT::RNTupleDescriptor &desc, const ROOT::RColumnDescriptor &colDesc,
                     const ROOT::RFieldDescriptor &fieldDesc)
      : fColDesc(&colDesc),
        fFieldDesc(&fieldDesc),
        // NOTE: we don't need to keep the column representation index into account because exactly 1 representation
        // is active per page, so there is no risk of name collisions.
        fQualifiedName(desc.GetQualifiedFieldName(fieldDesc.GetId()) + '-' + std::to_string(colDesc.GetIndex()))
   {
   }
};

struct RAddColumnsResult {
   int fNColsTotal = 0;

   RAddColumnsResult &operator+=(const RAddColumnsResult &other)
   {
      fNColsTotal += other.fNColsTotal;
      return *this;
   }
};

template <typename T>
bool ItemIsFilteredOut(const RNTupleExporter::RFilter<T> &filter, const T &item)
{
   bool filterHasType = filter.fSet.find(item) != filter.fSet.end();
   bool isFiltered = (filter.fType == RNTupleExporter::EFilterType::kBlacklist) == filterHasType;
   return isFiltered;
}

RAddColumnsResult AddColumnsFromField(std::vector<RColumnExportInfo> &vec, const ROOT::RNTupleDescriptor &desc,
                                      const ROOT::RFieldDescriptor &fieldDesc,
                                      const RNTupleExporter::RPagesOptions &options)
{
   R__LOG_DEBUG(1, RNTupleExporterLog()) << "processing field \"" << desc.GetQualifiedFieldName(fieldDesc.GetId())
                                         << "\"";

   RAddColumnsResult res{};

   for (const auto &subfieldDesc : desc.GetFieldIterable(fieldDesc)) {
      if (subfieldDesc.IsProjectedField())
         continue;

      for (const auto &colDesc : desc.GetColumnIterable(subfieldDesc)) {
         // Filter columns by type
         bool typeIsFiltered = ItemIsFilteredOut(options.fColumnTypeFilter, colDesc.GetType());
         if (!typeIsFiltered)
            vec.emplace_back(desc, colDesc, subfieldDesc);
         res.fNColsTotal += 1;
      }
      res += AddColumnsFromField(vec, desc, subfieldDesc, options);
   }

   return res;
}

int CountPages(const ROOT::RNTupleDescriptor &desc, std::span<const RColumnExportInfo> columns)
{
   int nPages = 0;
   auto clusterId = desc.FindClusterId(0, 0);
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = desc.GetClusterDescriptor(clusterId);
      for (const auto &colInfo : columns) {
         const auto &pages = clusterDesc.GetPageRange(colInfo.fColDesc->GetPhysicalId());
         nPages += pages.GetPageInfos().size();
      }
      clusterId = desc.FindNextClusterId(clusterId);
   }
   return nPages;
}

} // namespace

RNTupleExporter::RPagesResult
RNTupleExporter::ExportPages(ROOT::Internal::RPageSource &source, const RPagesOptions &options)
{
   RPagesResult res = {};

   // make sure the source is attached
   source.Attach();

   auto desc = source.GetSharedDescriptorGuard();
   ROOT::Internal::RClusterPool clusterPool{source};

   // Collect column info
   std::vector<RColumnExportInfo> columnInfos;
   const RAddColumnsResult addColRes = AddColumnsFromField(columnInfos, desc.GetRef(), desc->GetFieldZero(), options);

   // Collect ColumnSet for the cluster pool query
   ROOT::Internal::RCluster::ColumnSet_t columnSet;
   columnSet.reserve(columnInfos.size());
   for (const auto &colInfo : columnInfos) {
      columnSet.emplace(colInfo.fColDesc->GetPhysicalId());
   }

   const auto nPages = CountPages(desc.GetRef(), columnInfos);

   const bool showProgress = (options.fFlags & RPagesOptions::kShowProgressBar) != 0;
   res.fExportedFileNames.reserve(nPages);

   // Iterate over the clusters in order and dump pages
   auto clusterId = nPages > 0 ? desc->FindClusterId(0, 0) : ROOT::kInvalidDescriptorId;
   int pagesExported = 0;
   int prevIntPercent = 0;
   while (clusterId != ROOT::kInvalidDescriptorId) {
      const auto &clusterDesc = desc->GetClusterDescriptor(clusterId);
      const ROOT::Internal::RCluster *cluster = clusterPool.GetCluster(clusterId, columnSet);
      for (const auto &colInfo : columnInfos) {
         auto columnId = colInfo.fColDesc->GetPhysicalId();
         const auto &pages = clusterDesc.GetPageRange(columnId);
         const auto &colRange = clusterDesc.GetColumnRange(columnId);
         std::uint64_t pageIdx = 0;

         R__LOG_DEBUG(0, RNTupleExporterLog())
            << "exporting column \"" << colInfo.fQualifiedName << "\" (" << pages.GetPageInfos().size() << " pages)";

         // We should never try to export a suppressed column range
         assert(!colRange.IsSuppressed() || pages.GetPageInfos().empty());

         for (const auto &pageInfo : pages.GetPageInfos()) {
            ROOT::Internal::ROnDiskPage::Key key{columnId, pageIdx};
            const ROOT::Internal::ROnDiskPage *onDiskPage = cluster->GetOnDiskPage(key);

            // dump the page
            const void *pageBuf = onDiskPage->GetAddress();
            const bool incChecksum = (options.fFlags & RPagesOptions::kIncludeChecksums) != 0 && pageInfo.HasChecksum();
            const std::size_t maybeChecksumSize = incChecksum * 8;
            const std::uint64_t pageBufSize = pageInfo.GetLocator().GetNBytesOnStorage() + maybeChecksumSize;
            std::ostringstream ss{options.fOutputPath, std::ios_base::ate};
            assert(colRange.GetCompressionSettings());
            ss << "/cluster_" << clusterDesc.GetId() << "_" << colInfo.fQualifiedName << "_page_" << pageIdx
               << "_elems_" << pageInfo.GetNElements() << "_comp_" << *colRange.GetCompressionSettings() << ".page";
            const auto outFileName = ss.str();
            std::ofstream outFile{outFileName, std::ios_base::binary};
            if (!outFile)
               throw ROOT::RException(
                  R__FAIL(std::string("output path ") + options.fOutputPath + " does not exist or is not writable!"));

            outFile.write(reinterpret_cast<const char *>(pageBuf), pageBufSize);

            res.fExportedFileNames.push_back(outFileName);

            ++pageIdx, ++pagesExported;

            if (showProgress) {
               int intPercent = static_cast<int>(100.f * pagesExported / res.fExportedFileNames.size());
               if (intPercent != prevIntPercent) {
                  fprintf(stderr, "\rExport progress: %02d%%", intPercent);
                  if (intPercent == 100)
                     fprintf(stderr, "\n");
                  prevIntPercent = intPercent;
               }
            }
         }
      }
      clusterId = desc->FindNextClusterId(clusterId);
   }

   assert(res.fExportedFileNames.size() == static_cast<size_t>(pagesExported));
   std::ostringstream ss;
   ss << "exported " << res.fExportedFileNames.size() << " pages (";
   if (options.fColumnTypeFilter.fSet.empty()) {
      ss << addColRes.fNColsTotal << " columns)";
   } else {
      auto nColsFilteredOut = addColRes.fNColsTotal - columnInfos.size();
      ss << nColsFilteredOut << "/" << addColRes.fNColsTotal << " columns filtered out)";
   }
   R__LOG_INFO(RNTupleExporterLog()) << ss.str();

   return res;
}

} // namespace ROOT::Experimental::Internal
