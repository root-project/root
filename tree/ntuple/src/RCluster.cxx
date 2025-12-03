/// \file RCluster.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2020-03-11
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

#include <TError.h>

#include <iterator>
#include <utility>

ROOT::Internal::ROnDiskPageMap::~ROnDiskPageMap() = default;

////////////////////////////////////////////////////////////////////////////////

ROOT::Internal::ROnDiskPageMapHeap::~ROnDiskPageMapHeap() = default;

////////////////////////////////////////////////////////////////////////////////

const ROOT::Internal::ROnDiskPage *ROOT::Internal::RCluster::GetOnDiskPage(const ROnDiskPage::Key &key) const
{
   const auto itr = fOnDiskPages.find(key);
   if (itr != fOnDiskPages.end())
      return &(itr->second);
   return nullptr;
}

void ROOT::Internal::RCluster::Adopt(std::unique_ptr<ROnDiskPageMap> pageMap)
{
   auto &pages = pageMap->fOnDiskPages;
   fOnDiskPages.insert(std::make_move_iterator(pages.begin()), std::make_move_iterator(pages.end()));
   pageMap->fOnDiskPages.clear();
   fPageMaps.emplace_back(std::move(pageMap));
}

void ROOT::Internal::RCluster::Adopt(RCluster &&other)
{
   R__ASSERT(fClusterId == other.fClusterId);

   auto &pages = other.fOnDiskPages;
   fOnDiskPages.insert(std::make_move_iterator(pages.begin()), std::make_move_iterator(pages.end()));
   other.fOnDiskPages.clear();

   auto &columns = other.fAvailPhysicalColumns;
   fAvailPhysicalColumns.insert(std::make_move_iterator(columns.begin()), std::make_move_iterator(columns.end()));
   other.fAvailPhysicalColumns.clear();
   std::move(other.fPageMaps.begin(), other.fPageMaps.end(), std::back_inserter(fPageMaps));
   other.fPageMaps.clear();
}

void ROOT::Internal::RCluster::SetColumnAvailable(ROOT::DescriptorId_t physicalColumnId)
{
   fAvailPhysicalColumns.insert(physicalColumnId);
}
