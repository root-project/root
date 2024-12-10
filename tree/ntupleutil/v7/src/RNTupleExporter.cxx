/// \file RNTupleExporter.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
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

struct RColumnExportInfo {
   const RColumnDescriptor *fColDesc;
   const RFieldDescriptor *fFieldDesc;
   std::string fQualifiedName;

   RColumnExportInfo(const RNTupleDescriptor &desc, const RColumnDescriptor &colDesc, const RFieldDescriptor &fieldDesc)
      : fColDesc(&colDesc),
        fFieldDesc(&fieldDesc),
        fQualifiedName(desc.GetQualifiedFieldName(fieldDesc.GetId()) + '-' + std::to_string(colDesc.GetIndex()))
   {
   }
};

static void AddColumnsFromField(std::vector<RColumnExportInfo> &vec, const RNTupleDescriptor &desc,
                                const RFieldDescriptor &fieldDesc)
{
   R__LOG_DEBUG(1) << "processing field \"" << desc.GetQualifiedFieldName(fieldDesc.GetId()) << "\"";

   for (const auto &subfieldDesc : desc.GetFieldIterable(fieldDesc)) {
      if (subfieldDesc.IsProjectedField())
         continue;

      for (const auto &colDesc : desc.GetColumnIterable(subfieldDesc)) {
         vec.emplace_back(desc, colDesc, subfieldDesc);
      }
      AddColumnsFromField(vec, desc, subfieldDesc);
   }
}

RExportPagesResult RNTupleExporter::ExportPages(RPageSource &source, const RExportPagesOptions &options)
{
   RExportPagesResult res = {};

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

   bool reportFilenames = (options.fFlags & RExportPagesOptions::kReportExportedFileNames) != 0;

   // Iterate over the clusters in order and dump pages
   DescriptorId_t clusterId = desc->FindClusterId(0, 0);
   while (clusterId != kInvalidDescriptorId) {
      const auto &clusterDesc = desc->GetClusterDescriptor(clusterId);
      const RCluster *cluster = clusterPool.GetCluster(clusterId, columnSet);
      for (const auto &colInfo : columnInfos) {
         DescriptorId_t columnId = colInfo.fColDesc->GetPhysicalId();
         const auto &pages = clusterDesc.GetPageRange(columnId);
         const auto &colRange =clusterDesc.GetColumnRange(columnId);
         std::uint64_t pageIdx = 0;
         if (reportFilenames)
            res.fExportedFileNames.reserve(res.fExportedFileNames.size() + pages.fPageInfos.size());

         R__LOG_DEBUG(0) << "exporting column \"" << colInfo.fQualifiedName << "\" (" << pages.fPageInfos.size()
                         << " pages)";

         for (const auto &pageInfo : pages.fPageInfos) {
            ROnDiskPage::Key key{columnId, pageIdx};
            const ROnDiskPage *onDiskPage = cluster->GetOnDiskPage(key);

            // dump the page
            const void *pageBuf = onDiskPage->GetAddress();
            const auto maybeChecksumSize = (options.fFlags & RExportPagesOptions::kIncludeChecksums) ? 8 : 0;
            const std::uint64_t pageBufSize = pageInfo.fLocator.fBytesOnStorage + maybeChecksumSize;
            std::ostringstream ss{options.fOutputPath, std::ios_base::ate};
            ss << "/cluster_" << clusterDesc.GetId() << "_" << colInfo.fQualifiedName << "_page_" << pageIdx
               << "_elems_" << pageInfo.fNElements << "_comp_" << colRange.fCompressionSettings << ".page";
            const auto outFileName = ss.str();
            std::ofstream outFile{outFileName, std::ios_base::binary};
            outFile.write(reinterpret_cast<const char *>(pageBuf), pageBufSize);

            if (reportFilenames)
               res.fExportedFileNames.push_back(outFileName);

            ++pageIdx;
            ++res.fNPagesExported;
         }
      }
      clusterId = desc->FindNextClusterId(clusterId);
   }

   R__LOG_DEBUG(0) << "exported " << res.fNPagesExported << " pages.";

   return res;
}

} // namespace ROOT::Experimental::Internal
