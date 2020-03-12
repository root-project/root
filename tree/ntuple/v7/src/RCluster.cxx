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

ROOT::Experimental::Detail::RCluster::~RCluster()
{
}

const ROOT::Experimental::Detail::ROnDiskPage *
ROOT::Experimental::Detail::RCluster::GetOnDiskPage(const ROnDiskPage::Key &key) const
{
   const auto itr = fOnDiskPages.find(key);
   if (itr != fOnDiskPages.end())
      return &(itr->second);
   return nullptr;
}


ROOT::Experimental::Detail::RHeapCluster::~RHeapCluster()
{
   delete[] static_cast<unsigned char *>(fMemory);
}


ROOT::Experimental::Detail::RMMapCluster::~RMMapCluster()
{
}
