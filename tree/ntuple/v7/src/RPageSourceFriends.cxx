/// \file RPageSourceFriends.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RCluster.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageSourceFriends.hxx>

#include <utility>

ROOT::Experimental::Detail::RPageSourceFriends::RPageSourceFriends(
   std::string_view ntupleName, std::span<std::unique_ptr<RPageSource>> sources)
   : RPageSource(ntupleName, RNTupleReadOptions())
   , fMetrics(std::string(ntupleName))
{
   for (auto &s : sources) {
      fSources.emplace_back(std::move(s));
      fMetrics.ObserveMetrics(fSources.back()->GetMetrics());
   }
}

ROOT::Experimental::Detail::RPageSourceFriends::~RPageSourceFriends() = default;


void ROOT::Experimental::Detail::RPageSourceFriends::AddVirtualField(
   const RNTupleDescriptor &originDesc,
   std::size_t originIdx,
   const RFieldDescriptor &originField,
   DescriptorId_t virtualParent,
   const std::string &virtualName)
{
   auto virtualFieldId = fNextId++;
   auto virtualField = RDanglingFieldDescriptor(originField)
      .FieldId(virtualFieldId)
      .FieldName(virtualName)
      .MakeDescriptor().Unwrap();
   fBuilder.AddField(virtualField);
   fBuilder.AddFieldLink(virtualParent, virtualFieldId);

   for (const auto &f : originDesc.GetFieldRange(originField))
      AddVirtualField(originDesc, originIdx, f, virtualFieldId, f.GetFieldName());

   for (const auto &c: originDesc.GetColumnRange(originField)) {
      fBuilder.AddColumn(fNextId, virtualFieldId, c.GetVersion(), c.GetModel(), c.GetIndex());
      fVirtual2OriginColumn[fNextId] = {originIdx, c.GetId()};
      fNextId++;
   }
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceFriends::AttachImpl()
{
   fBuilder.SetNTuple(fNTupleName, "", "", RNTupleVersion(), RNTupleUuid());
   fBuilder.AddField(RDanglingFieldDescriptor()
      .FieldId(0)
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());

   for (std::size_t i = 0; i < fSources.size(); ++i) {
      fSources[i]->Attach();
      const auto &desc = fSources[i]->GetDescriptor();
      AddVirtualField(desc, i, desc.GetFieldZero(), 0, desc.GetName());

      for (const auto &c : desc.GetClusterRange()) {
         fBuilder.AddCluster(fNextId, c.GetVersion(), c.GetFirstEntryIndex(), c.GetNEntries());
         for (auto columnId : c.GetColumnIds()) {
            //auto pageRange = c.GetPageRange(columnId);
            //fBuilder.AddClusterPageRange(columnId, pageRange);
            fBuilder.AddClusterColumnRange(columnId, c.GetColumnRange(columnId));
         }
         fNextId++;
      }
   }

   fBuilder.EnsureValidDescriptor();
   return fBuilder.MoveDescriptor();
}


std::unique_ptr<ROOT::Experimental::Detail::RPageSource>
ROOT::Experimental::Detail::RPageSourceFriends::Clone() const
{
   // TODO
   return nullptr;
}


ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceFriends::PopulatePage(
   ColumnHandle_t /*columnHandle*/, NTupleSize_t /*globalIndex*/)
{
   return RPage();
   //return fSources[fVirtual2OriginColumn[columnHandle.fId].fColumnId].PopulatePage()
}


ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceFriends::PopulatePage(
   ColumnHandle_t /*columnHandle*/, const RClusterIndex &/*clusterIndex*/)
{
   return RPage();
}

void ROOT::Experimental::Detail::RPageSourceFriends::ReleasePage(RPage &/*page*/)
{

}


std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceFriends::LoadCluster(DescriptorId_t /* clusterId */,
                                                            const ColumnSet_t & /* columns */)
{
   // The virtual friends page source does not pre-load any clusters itself but the underlying page sources
   // that are combined may well do it.
   return nullptr;
}
