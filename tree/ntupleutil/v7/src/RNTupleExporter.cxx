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

ROOT::Experimental::RLogChannel &RNTupleExporterLog()
{
   static RLogChannel sLog("ROOT.RNTupleExporter");
   return sLog;
}

struct RColumnExportInfo {
   const RColumnDescriptor *fColDesc;
   const RFieldDescriptor *fFieldDesc;
   std::string fQualifiedName;

   RColumnExportInfo(const RNTupleDescriptor &desc, const RColumnDescriptor &colDesc, const RFieldDescriptor &fieldDesc)
      : fColDesc(&colDesc),
        fFieldDesc(&fieldDesc),
        // NOTE: we don't need to keep the column representation index into account because exactly 1 representation
        // is active per page, so there is no risk of name collisions.
        fQualifiedName(desc.GetQualifiedFieldName(fieldDesc.GetId()) + '-' + std::to_string(colDesc.GetIndex()))
   {
   }
};

void AddColumnsFromField(std::vector<RColumnExportInfo> &vec, const RNTupleDescriptor &desc,
                         const RFieldDescriptor &fieldDesc)
{
   R__LOG_DEBUG(1, RNTupleExporterLog()) << "processing field \"" << desc.GetQualifiedFieldName(fieldDesc.GetId())
                                         << "\"";

   for (const auto &subfieldDesc : desc.GetFieldIterable(fieldDesc)) {
      if (subfieldDesc.IsProjectedField())
         continue;

      for (const auto &colDesc : desc.GetColumnIterable(subfieldDesc)) {
         vec.emplace_back(desc, colDesc, subfieldDesc);
      }
      AddColumnsFromField(vec, desc, subfieldDesc);
   }
}

int CountPages(const RNTupleDescriptor &desc, std::span<const RColumnExportInfo> columns)
{
   int nPages = 0;
   DescriptorId_t clusterId = desc.FindClusterId(0, 0);
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = desc.GetClusterDescriptor(clusterId);
      for (const auto &colInfo : columns) {
         const auto &pages = clusterDesc.GetPageRange(colInfo.fColDesc->GetPhysicalId());
         nPages += pages.fPageInfos.size();
      }
      clusterId = desc.FindNextClusterId(clusterId);
   }
   return nPages;
}

} // namespace

RNTupleExporter::RPagesResult RNTupleExporter::ExportPages(RPageSource &source, const RPagesOptions &options)
{
   RPagesResult res = {};

   // make sure the source is attached
   source.Attach();

   auto desc = source.GetSharedDescriptorGuard();
   RClusterPool clusterPool{source};

   // Collect column info
   std::vector<RColumnExportInfo> columnInfos;
   AddColumnsFromField(columnInfos, desc.GetRef(), desc->GetFieldZero());

   // Collect ColumnSet for the cluster pool query
   RCluster::ColumnSet_t columnSet;
   columnSet.reserve(columnInfos.size());
   for (const auto &colInfo : columnInfos) {
      columnSet.emplace(colInfo.fColDesc->GetPhysicalId());
   }

   const auto nPages = CountPages(desc.GetRef(), columnInfos);

   const bool showProgress = (options.fFlags & RPagesOptions::kShowProgressBar) != 0;
   res.fExportedFileNames.reserve(nPages);

   // Iterate over the clusters in order and dump pages
   DescriptorId_t clusterId = desc->FindClusterId(0, 0);
   int pagesExported = 0;
   int prevIntPercent = 0;
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = desc->GetClusterDescriptor(clusterId);
      const RCluster *cluster = clusterPool.GetCluster(clusterId, columnSet);
      for (const auto &colInfo : columnInfos) {
         DescriptorId_t columnId = colInfo.fColDesc->GetPhysicalId();
         const auto &pages = clusterDesc.GetPageRange(columnId);
         const auto &colRange = clusterDesc.GetColumnRange(columnId);
         std::uint64_t pageIdx = 0;

         R__LOG_DEBUG(0, RNTupleExporterLog())
            << "exporting column \"" << colInfo.fQualifiedName << "\" (" << pages.fPageInfos.size() << " pages)";

         // We should never try to export a suppressed column range
         assert(!colRange.fIsSuppressed || pages.fPageInfos.empty());

         for (const auto &pageInfo : pages.fPageInfos) {
            ROnDiskPage::Key key{columnId, pageIdx};
            const ROnDiskPage *onDiskPage = cluster->GetOnDiskPage(key);

            // dump the page
            const void *pageBuf = onDiskPage->GetAddress();
            const bool incChecksum = (options.fFlags & RPagesOptions::kIncludeChecksums) != 0 && pageInfo.fHasChecksum;
            const std::size_t maybeChecksumSize = incChecksum * 8;
            const std::uint64_t pageBufSize = pageInfo.fLocator.fBytesOnStorage + maybeChecksumSize;
            std::ostringstream ss{options.fOutputPath, std::ios_base::ate};
            ss << "/cluster_" << clusterDesc.GetId() << "_" << colInfo.fQualifiedName << "_page_" << pageIdx
               << "_elems_" << pageInfo.fNElements << "_comp_" << colRange.fCompressionSettings << ".page";
            const auto outFileName = ss.str();
            std::ofstream outFile{outFileName, std::ios_base::binary};
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
   R__LOG_INFO(RNTupleExporterLog()) << "exported " << res.fExportedFileNames.size() << " pages.";

   return res;
}

} // namespace ROOT::Experimental::Internal
