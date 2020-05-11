/// \file RCluster.cxx
/// \ingroup NTuple ROOT7
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


ROOT::Experimental::Detail::ROnDiskPageMap::ROnDiskPageMap(ROnDiskPageMap &&other)
   : fMemory(other.fMemory), fOnDiskPages(std::move(other.fOnDiskPages))
{
   other.fMemory = nullptr;
   other.fOnDiskPages.clear();
}


ROOT::Experimental::Detail::ROnDiskPageMap &
ROOT::Experimental::Detail::ROnDiskPageMap::operator =(ROnDiskPageMap &&other)
{
   fMemory = other.fMemory;
   other.fMemory = nullptr;
   fOnDiskPages = std::move(other.fOnDiskPages);
   other.fOnDiskPages.clear();
   return *this;
}


ROOT::Experimental::Detail::ROnDiskPageMap::~ROnDiskPageMap()
{
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::ROnDiskPageMapHeap::~ROnDiskPageMapHeap()
{
   delete[] static_cast<unsigned char *>(fMemory);
}


////////////////////////////////////////////////////////////////////////////////


const ROOT::Experimental::Detail::ROnDiskPage *
ROOT::Experimental::Detail::RCluster::GetOnDiskPage(const ROnDiskPage::Key &key) const
{
   const auto itr = fOnDiskPages.find(key);
   if (itr != fOnDiskPages.end())
      return &(itr->second);
   return nullptr;
}

void ROOT::Experimental::Detail::RCluster::MergePageMap(ROnDiskPageMap &&pageMap)
{
   for (const auto &entry : pageMap.fOnDiskPages) {
      fOnDiskPages.emplace(entry.first, entry.second);
   }
   pageMap.fOnDiskPages.clear();
   fPageMaps.emplace_back(std::move(pageMap));
}


void ROOT::Experimental::Detail::RCluster::CommitColumn(DescriptorId_t columnId)
{
   fAvailColumns.insert(columnId);
}


void ROOT::Experimental::Detail::RCluster::MergeCluster(RCluster &&other)
{
   R__ASSERT(fClusterId == other.fClusterId);

   for (const auto &entry : other.fOnDiskPages) {
      fOnDiskPages.emplace(entry.first, entry.second);
   }
   other.fOnDiskPages.clear();

   fAvailColumns.insert(other.fAvailColumns.begin(), other.fAvailColumns.end());
   other.fAvailColumns.clear();
   std::move(other.fPageMaps.begin(), other.fPageMaps.end(), std::back_inserter(fPageMaps));
   other.fPageMaps.clear();
}
