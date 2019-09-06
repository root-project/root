/// \file RCluster.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-09-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RCluster.hxx>
#include <ROOT/RRawFile.hxx>

#include <cstdlib>

ROOT::Experimental::Detail::RCluster::~RCluster()
{
}

const ROOT::Experimental::Detail::RSheet *
ROOT::Experimental::Detail::RCluster::GetSheet(const RSheetKey &key) const
{
   const auto itr = fSheets.find(key);
   if (itr != fSheets.end())
      return &(itr->second);
   return nullptr;
}


ROOT::Experimental::Detail::RHeapCluster::~RHeapCluster()
{
   free(fHandle);
}

ROOT::Experimental::Detail::RMMapCluster::~RMMapCluster()
{
   fFile.Unmap(fHandle, fLength);
}
