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
   fDirectory->WriteObject(&columnHeader,
      (RMapper::kKeyColumnHeader + std::to_string(fMapper.fColumn2Id.size())).c_str());
   fMapper.fColumn2Id[column] = fMapper.fColumn2Id.size();
   printf("Added column %s type %d\n", columnHeader.fName.c_str(), (int)columnHeader.fType);
}


void ROOT::Experimental::Detail::RPageSinkRoot::Create(RTreeModel *model)
{
   fDirectory = fSettings.fFile->mkdir(fForestName.c_str());
   ROOT::Experimental::Internal::RForestHeader forestHeader;

   for (auto& f : *model->GetRootField()) {
      ROOT::Experimental::Internal::RFieldHeader fieldHeader;
      fieldHeader.fName = f.GetName();
      fieldHeader.fType = f.GetType();
      if (f.GetParent()) fieldHeader.fParentName = f.GetParent()->GetName();

      fDirectory->WriteObject(&fieldHeader, (RMapper::kKeyFieldHeader + std::to_string(forestHeader.fNFields)).c_str());
      f.GenerateColumns(this); // issues in turn one or several calls to AddColumn()
      forestHeader.fNFields++;
   }

   forestHeader.fNColumns = fMapper.fColumn2Id.size();
   fPagePool = std::make_unique<RPagePool>(kPageSize, forestHeader.fNColumns);
   for (auto column : fMapper.fColumn2Id) {
      SetHeadPage(fPagePool->ReservePage(column.first), column.first);
   }

   fDirectory->WriteObject(&forestHeader, RMapper::kKeyForestHeader);
}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitPage(const RPage &page, RColumn *column)
{
   printf("WRITING OUT PAGE\n");
}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitCluster(ROOT::Experimental::TreeIndex_t /*nEntries*/)
{

}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitDataset(ROOT::Experimental::TreeIndex_t /*nEntries*/)
{

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
}


void ROOT::Experimental::Detail::RPageSourceRoot::AddColumn(RColumn* /*column*/)
{
}


void ROOT::Experimental::Detail::RPageSourceRoot::Attach()
{
   fDirectory = fSettings.fFile->GetDirectory(fForestName.c_str());
   auto keyForestHeader = fDirectory->GetKey(RMapper::kKeyForestHeader);
   fForestHeader = keyForestHeader->ReadObject<ROOT::Experimental::Internal::RForestHeader>();
   printf("Number of fields %d, of columns %d\n", fForestHeader->fNFields, fForestHeader->fNColumns);

   for (std::int32_t id = 0; id < fForestHeader->fNColumns; ++id) {
      ROOT::Experimental::Internal::RColumnHeader* columnHeader;
      auto keyColumnHeader = fDirectory->GetKey((RMapper::kKeyColumnHeader + std::to_string(id)).c_str());
      columnHeader = keyColumnHeader->ReadObject<ROOT::Experimental::Internal::RColumnHeader>();
      auto columnModel = std::make_unique<RColumnModel>(
         columnHeader->fName, columnHeader->fType, columnHeader->fIsSorted);
      fMapper.fId2ColumnModel[id] = std::move(columnModel);
      fMapper.fColumnName2Id[columnHeader->fName] = id;
   }
}


std::unique_ptr<ROOT::Experimental::RTreeModel> ROOT::Experimental::Detail::RPageSourceRoot::GenerateModel()
{
   return nullptr;
}
