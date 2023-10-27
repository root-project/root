/// \file RNTupleMerger.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch> & Max Orok <maxwellorok@gmail.com>
/// \date 2020-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleUtil.hxx>

Long64_t ROOT::Experimental::RNTuple::Merge(TCollection *inputs, TFileMergeInfo *mergeInfo)
{
   if (inputs == nullptr || mergeInfo == nullptr) {
      return -1;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<ROOT::Experimental::RFieldMerger>
ROOT::Experimental::RFieldMerger::Merge(const ROOT::Experimental::RFieldDescriptor &lhs,
                                        const ROOT::Experimental::RFieldDescriptor &rhs)
{
   return R__FAIL("couldn't merge field " + lhs.GetFieldName() + " with field " + rhs.GetFieldName() +
                  " (unimplemented!)");
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::RNTupleMerger::Merge(std::vector<std::unique_ptr<Detail::RPageSource>> &sources,
                                              std::unique_ptr<Detail::RPageSink> &destination)
{

   // Total entries written
   std::uint64_t nEntries{0};

   // Loop over all input sources
   bool firstSource = true;
   for (const auto &source : sources) {

      // Attach the current source
      source->Attach();

      // Collect all the columns
      // The column name : output column id map is only built once
      auto columns = CollectColumns(source, firstSource);

      // Get a handle on the descriptor (metadata)
      auto descriptor = source->GetSharedDescriptorGuard();

      // Create sink from the input model of the very first input file
      if(firstSource) {
         auto model = descriptor->GenerateModel();
         destination->Create(*model.get());
         firstSource = false;
      }

      // Now loop over all clusters in this file
      // descriptor->GetClusterIterable() doesn't guarantee any specific order...
      // Find the first cluster id and iterate from there...
      // Here we assume there is at least one entry in the ntuple, beware...
      auto clusterId = descriptor->FindClusterId(0, 0);

      while (clusterId != ROOT::Experimental::kInvalidDescriptorId) {

         // Get a hold on the current cluster descriptor
         auto &cluster = descriptor->GetClusterDescriptor(clusterId);

         // Now loop over all columns
         for (const auto &column : columns) {

            // See if this cluster contains this column
            // if not, there is nothing to read/do...
            auto columnId = column.fColumnInputId;
            if (!cluster.ContainsColumn(columnId)) {
               continue;
            }

            // Now get the pages for this column in this cluster
            const auto &pages = cluster.GetPageRange(columnId);
            size_t idx{0};

            // Loop over the pages
            for (const auto &pageInfo : pages.fPageInfos) {

               // Each page contains N elements that we are going to read together
               // LoadSealedPage reads packed/compressed bytes of a page into
               // a memory buffer provided by a sealed page
               RClusterIndex clusterIndex(clusterId, idx);
               Detail::RPageStorage::RSealedPage sealedPage;
               source->LoadSealedPage(columnId, clusterIndex, sealedPage);

               // The way LoadSealedPage works might require a double call
               // See the implementation. Here we do this in any case...
               auto buffer = std::make_unique<unsigned char[]>(sealedPage.fSize);
               sealedPage.fBuffer = buffer.get();
               source->LoadSealedPage(columnId, clusterIndex, sealedPage);

               // Now commit this page to the output
               // Can we do this w/ a CommitSealedPageV
               destination->CommitSealedPage(column.fColumnOutputId, sealedPage);

               // Move on to the next index
               idx += pageInfo.fNElements;

            } // end of loop over pages

         } // end of loop over columns

         // Commit the clusters
         nEntries += cluster.GetNEntries();
         destination->CommitCluster(nEntries);

         // Go to the next cluster
         clusterId = descriptor->FindNextClusterId(clusterId);

      } // end of loop over clusters

      // Commit all clusters for this input
      destination->CommitClusterGroup();

   } // end of loop over sources

   // Commit the output
   destination->CommitDataset();
}
