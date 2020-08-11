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
   , fMetrics("RPageSourceFriends")
{
   for (auto &s : sources) {
      fSources.emplace_back(std::move(s));
      fMetrics.ObserveMetrics(fSources.back()->GetMetrics());
   }
}

ROOT::Experimental::Detail::RPageSourceFriends::~RPageSourceFriends() = default;

void ROOT::Experimental::Detail::RPageSourceFriends::AddVirtualField(
   const RNTupleDescriptor &originDesc,
   const RFieldDescriptor &originChild,
   DescriptorId_t virtualParent)
{
   auto virtualFieldId = fNextId++;
   auto virtualField = RDanglingFieldDescriptor(originChild)
      .FieldId(virtualFieldId)
      .MakeDescriptor().Unwrap();
   fBuilder.AddField(virtualField);
   fBuilder.AddFieldLink(virtualParent, virtualFieldId);

   for (const auto &f : originDesc.GetFieldRange(originChild))
      AddVirtualField(originDesc, f, virtualFieldId);

   // TODO(jblomer): add column range and dangling columns
   std::uint32_t columnIdx = 0;
   DescriptorId_t columnId;
   while ((columnId = originDesc.FindColumnId(originChild.GetId(), columnIdx)) != kInvalidDescriptorId) {
      const auto &column = originDesc.GetColumnDescriptor(columnId);
      fBuilder.AddColumn(fNextId++, virtualFieldId, column.GetVersion(), column.GetModel(), columnIdx);
      columnIdx++;
   }
}

#include <iostream>
ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceFriends::AttachImpl()
{
   fBuilder.SetNTuple(fNTupleName, "", "", RNTupleVersion(), RNTupleUuid());
   fBuilder.AddField(RDanglingFieldDescriptor()
      .FieldId(0)
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());

   for (auto &s : fSources) {
      s->Attach();
      const auto &desc = s->GetDescriptor();
      for (const auto &f : desc.GetTopLevelFields())
         AddVirtualField(desc, f, 0);
   }

   fBuilder.EnsureValidDescriptor();
   fBuilder.GetDescriptor().PrintInfo(std::cout);
   return fBuilder.MoveDescriptor();
}


std::unique_ptr<ROOT::Experimental::Detail::RPageSource>
ROOT::Experimental::Detail::RPageSourceFriends::Clone() const
{

}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceFriends::PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{

}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceFriends::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{

}

void ROOT::Experimental::Detail::RPageSourceFriends::ReleasePage(RPage &page)
{

}


std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceFriends::LoadCluster(DescriptorId_t clusterId, const ColumnSet_t &columns)
{

}
