/// \file RPageStorage.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPageStorageRoot.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RTreeModel.hxx>

#include <TKey.h>

#include <utility>


ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view forestName, RSettings settings)
   : ROOT::Experimental::Detail::RPageSink(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
   , fSettings(settings)
{
}

ROOT::Experimental::Detail::RPageSinkRoot::~RPageSinkRoot()
{
   if (fSettings.fTakeOwnership) {
      fSettings.fFile->Close();
      delete fSettings.fFile;
   }
}

void ROOT::Experimental::Detail::RPageSinkRoot::AddColumn(RColumn* column)
{
   ROOT::Experimental::Internal::RColumnHeader columnHeader;
   columnHeader.fName = column->GetModel().GetName();
   columnHeader.fType = column->GetModel().GetType();
   columnHeader.fIsSorted = column->GetModel().GetIsSorted();
   fForestHeader.fColumns.emplace_back(columnHeader);
   fMapper.fColumn2Id[column] = fMapper.fColumn2Id.size();
   printf("Added column %s type %d\n", columnHeader.fName.c_str(), (int)columnHeader.fType);
}


void ROOT::Experimental::Detail::RPageSinkRoot::Create(RTreeModel *model)
{
   fDirectory = fSettings.fFile->mkdir(fForestName.c_str());

   for (auto& f : *model->GetRootField()) {
      ROOT::Experimental::Internal::RFieldHeader fieldHeader;
      fieldHeader.fName = f.GetName();
      fieldHeader.fType = f.GetType();
      if (f.GetParent()) fieldHeader.fParentName = f.GetParent()->GetName();
      fForestHeader.fFields.emplace_back(fieldHeader);

      f.GenerateColumns(this); // issues in turn one or several calls to AddColumn()
   }

   fPagePool = std::make_unique<RPagePool>(kPageSize, fMapper.fColumn2Id.size());
   for (auto column : fMapper.fColumn2Id) {
      SetHeadPage(fPagePool->ReservePage(column.first), column.first);
   }

   fDirectory->WriteObject(&fForestHeader, RMapper::kKeyForestHeader);
   fCurrentCluster.fPagesPerColumn.resize(fForestHeader.fColumns.size());
}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitPage(const RPage &page, RColumn *column)
{
   auto columnId = fMapper.fColumn2Id[column];
   ROOT::Experimental::Internal::RPagePayload pagePayload;
   pagePayload.fSize = page.GetSize();
   pagePayload.fContent = static_cast<unsigned char *>(page.GetBuffer());
   std::string key = std::string(RMapper::kKeyPagePayload) +
      std::to_string(fForestFooter.fNClusters) + RMapper::kKeySeparator +
      std::to_string(columnId) + RMapper::kKeySeparator +
      std::to_string(fCurrentCluster.fPagesPerColumn[columnId].fRangeStarts.size());
   fDirectory->WriteObject(&pagePayload, key.c_str());
   fCurrentCluster.fPagesPerColumn[columnId].fRangeStarts.push_back(page.GetRangeStart());
}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitCluster(ROOT::Experimental::TreeIndex_t nEntries)
{
   fCurrentCluster.fNEntries = nEntries;
   std::string key = RMapper::kKeyClusterFooter + std::to_string(fForestFooter.fNClusters);
   fDirectory->WriteObject(&fCurrentCluster, key.c_str());
   fForestFooter.fNClusters++;
   fForestFooter.fNEntries += nEntries;

   for (auto& pageInfo : fCurrentCluster.fPagesPerColumn) {
      pageInfo.fRangeStarts.clear();
   }
   fCurrentCluster.fEntryRangeStart = fForestFooter.fNEntries;
}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitDataset()
{
   fDirectory->WriteObject(&fForestFooter, RMapper::kKeyForestFooter);
}


//------------------------------------------------------------------------------


ROOT::Experimental::Detail::RPageSourceRoot::RPageSourceRoot(std::string_view forestName, RSettings settings)
   : ROOT::Experimental::Detail::RPageSource(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
   , fSettings(settings)
{
}


ROOT::Experimental::Detail::RPageSourceRoot::~RPageSourceRoot()
{
   if (fSettings.fTakeOwnership) {
      fSettings.fFile->Close();
      delete fSettings.fFile;
   }
}


void ROOT::Experimental::Detail::RPageSourceRoot::AddColumn(RColumn* column)
{
   auto& model = column->GetModel();
   auto columnId = fMapper.fColumnName2Id[model.GetName()];
   R__ASSERT(model == *fMapper.fId2ColumnModel[columnId]);
   fMapper.fColumn2Id[column] = columnId;
   printf("Attaching column %s id %d type %d\n", column->GetModel().GetName().c_str(), columnId, (int)(column->GetModel().GetType()));
}


void ROOT::Experimental::Detail::RPageSourceRoot::Attach()
{
   fDirectory = fSettings.fFile->GetDirectory(fForestName.c_str());
   auto keyForestHeader = fDirectory->GetKey(RMapper::kKeyForestHeader);
   auto forestHeader = keyForestHeader->ReadObject<ROOT::Experimental::Internal::RForestHeader>();
   printf("Number of fields %lu, of columns %lu\n", forestHeader->fFields.size(), forestHeader->fColumns.size());

   std::int32_t columnId = 0;
   for (auto &columnHeader : forestHeader->fColumns) {
      auto columnModel = std::make_unique<RColumnModel>(
         columnHeader.fName, columnHeader.fType, columnHeader.fIsSorted);
      fMapper.fId2ColumnModel[columnId] = std::move(columnModel);
      fMapper.fColumnName2Id[columnHeader.fName] = columnId;
      columnId++;
   }

   auto keyForestFooter = fDirectory->GetKey(RMapper::kKeyForestFooter);
   auto forestFooter = keyForestFooter->ReadObject<ROOT::Experimental::Internal::RForestFooter>();
   printf("Number of clusters: %d\n", forestFooter->fNClusters);
   auto nColumns = forestHeader->fColumns.size();
   fMapper.fColumnIndex.resize(nColumns);

   for (std::int32_t iCluster = 0; iCluster < forestFooter->fNClusters; ++iCluster) {
      auto keyClusterFooter = fDirectory->GetKey((RMapper::kKeyClusterFooter + std::to_string(iCluster)).c_str());
      auto clusterFooter = keyClusterFooter->ReadObject<ROOT::Experimental::Internal::RClusterFooter>();
      R__ASSERT(clusterFooter->fPagesPerColumn.size() == nColumns);
      for (unsigned iColumn = 0; iColumn < nColumns; ++iColumn) {
         for (auto rangeStart : clusterFooter->fPagesPerColumn[iColumn].fRangeStarts) {
            fMapper.fColumnIndex[iColumn].fRangeStarts.push_back(rangeStart);
         }
      }
      delete clusterFooter;
   }

   delete forestFooter;
   delete forestHeader;
}


std::unique_ptr<ROOT::Experimental::RTreeModel> ROOT::Experimental::Detail::RPageSourceRoot::GenerateModel()
{
   return nullptr;
}
