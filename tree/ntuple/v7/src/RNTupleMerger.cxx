/// \file RNTupleMerger.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>
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
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>

Long64_t ROOT::Experimental::RNTuple::Merge(TCollection *inputs, TFileMergeInfo *mergeInfo)
{
   if (inputs == nullptr || mergeInfo == nullptr) {
      return -1;
   }
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::BuildColumnIdMap(
   std::vector<ROOT::Experimental::Internal::RNTupleMerger::RColumnInfo> &columns)
{
   for (auto &column : columns) {
      column.fColumnOutputId = fOutputIdMap.size();
      fOutputIdMap[column.fColumnName + "." + column.fColumnTypeAndVersion] = column.fColumnOutputId;
   }
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::ValidateColumns(
   std::vector<ROOT::Experimental::Internal::RNTupleMerger::RColumnInfo> &columns)
{
   // First ensure that we have the same number of columns
   if (fOutputIdMap.size() != columns.size()) {
      throw RException(R__FAIL("Columns between sources do NOT match"));
   }
   // Then ensure that we have the same names of columns and assign the ids
   for (auto &column : columns) {
      try {
         column.fColumnOutputId = fOutputIdMap.at(column.fColumnName + "." + column.fColumnTypeAndVersion);
      } catch (const std::out_of_range &) {
         throw RException(R__FAIL("Column NOT found in the first source w/ name " + column.fColumnName +
                                  " type and version " + column.fColumnTypeAndVersion));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
std::vector<ROOT::Experimental::Internal::RNTupleMerger::RColumnInfo>
ROOT::Experimental::Internal::RNTupleMerger::CollectColumns(const RPageSource &source, bool firstSource)
{
   auto desc = source.GetSharedDescriptorGuard();
   std::vector<RColumnInfo> columns;
   // Here we recursively find the columns and fill the RColumnInfo vector
   AddColumnsFromField(columns, desc.GetRef(), desc->GetFieldZero());
   // Then we either build the internal map (first source) or validate the columns against it (remaning sources)
   // In either case, we also assign the output ids here
   if (firstSource) {
      BuildColumnIdMap(columns);
   } else {
      ValidateColumns(columns);
   }
   return columns;
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::AddColumnsFromField(
   std::vector<ROOT::Experimental::Internal::RNTupleMerger::RColumnInfo> &columns, const RNTupleDescriptor &desc,
   const RFieldDescriptor &fieldDesc, const std::string &prefix)
{
   for (const auto &field : desc.GetFieldIterable(fieldDesc)) {
      std::string name = prefix + field.GetFieldName() + ".";
      const std::string typeAndVersion = field.GetTypeName() + "." + std::to_string(field.GetTypeVersion());
      for (const auto &column : desc.GetColumnIterable(field)) {
         columns.emplace_back(name + std::to_string(column.GetIndex()), typeAndVersion, column.GetPhysicalId(),
                              kInvalidDescriptorId);
      }
      AddColumnsFromField(columns, desc, field, name);
   }
}

////////////////////////////////////////////////////////////////////////////////
void ROOT::Experimental::Internal::RNTupleMerger::Merge(std::span<RPageSource *> sources, RPageSink &destination)
{
   // Append the sources to the destination one-by-one
   bool isFirstSource = true;
   for (const auto &source : sources) {
      source->Attach();

      // Make sure the source contains events to be merged
      if (source->GetNEntries() == 0) {
         continue;
      }

      // Collect all the columns
      // The column name : output column id map is only built once
      auto columns = CollectColumns(*source, isFirstSource);

      // Get a handle on the descriptor (metadata)
      auto descriptor = source->GetSharedDescriptorGuard();

      // Create sink from the input model of the very first input file
      if (isFirstSource) {
         auto model = descriptor->CreateModel();
         destination.Init(*model.get());
         isFirstSource = false;
      }

      // Now loop over all clusters in this file
      // descriptor->GetClusterIterable() doesn't guarantee any specific order...
      // Find the first cluster id and iterate from there...
      auto clusterId = descriptor->FindClusterId(0, 0);

      while (clusterId != ROOT::Experimental::kInvalidDescriptorId) {
         auto &cluster = descriptor->GetClusterDescriptor(clusterId);

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
               Internal::RPageStorage::RSealedPage sealedPage;
               source->LoadSealedPage(columnId, clusterIndex, sealedPage);

               // The way LoadSealedPage works might require a double call
               // See the implementation. Here we do this in any case...
               auto buffer = std::make_unique<unsigned char[]>(sealedPage.fSize);
               sealedPage.fBuffer = buffer.get();
               source->LoadSealedPage(columnId, clusterIndex, sealedPage);

               // Now commit this page to the output
               // Can we do this w/ a CommitSealedPageV
               destination.CommitSealedPage(column.fColumnOutputId, sealedPage);

               // Move on to the next index
               idx += pageInfo.fNElements;

            } // end of loop over pages

         } // end of loop over columns

         // Commit the clusters
         destination.CommitCluster(cluster.GetNEntries());

         // Go to the next cluster
         clusterId = descriptor->FindNextClusterId(clusterId);

      } // end of loop over clusters

      // Commit all clusters for this input
      destination.CommitClusterGroup();

   } // end of loop over sources

   // Commit the output
   destination.CommitDataset();
}
