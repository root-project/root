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

#include "ROOT/RPageStorageRoot.hxx"
#include "ROOT/RTreeModel.hxx"

#include "TKey.h"

namespace ROOT {
namespace Experimental {

class RSlotTFile : TKey {
public:

};

} // namespace Experimental
} // namespace ROOT


ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view forestName, RSettings settings)
   : ROOT::Experimental::Detail::RPageSink(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
   , fSettings(settings)
{
}

ROOT::Experimental::Detail::RPageSinkRoot::~RPageSinkRoot()
{
}

void ROOT::Experimental::Detail::RPageSinkRoot::AddColumn(RColumn* column)
{
   ROOT::Experimental::Internal::RColumnHeader columnHeader;
   columnHeader.fName = column->GetModel().GetName();
   columnHeader.fType = column->GetModel().GetType();
   columnHeader.fIsSorted = column->GetModel().GetIsSorted();
   fDirectory->WriteObject(&columnHeader, (kKeyColumnHeader + std::to_string(fColumn2Id.size())).c_str());
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

      fDirectory->WriteObject(&fieldHeader, (kKeyFieldHeader + std::to_string(forestHeader.fNFields)).c_str());
      f.GenerateColumns(this);
      forestHeader.fNFields++;
   }

   fDirectory->WriteObject(&forestHeader, kKeyForestHeader);
}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitPage(RPage* /*page*/)
{

}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitCluster(ROOT::Experimental::TreeIndex_t /*nEntries*/)
{

}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitDataset(ROOT::Experimental::TreeIndex_t /*nEntries*/)
{

}
