/// \file RForestEntry.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REntry.hxx>
#include <ROOT/RFieldValue.hxx>

ROOT::Experimental::RForestEntry::~RForestEntry()
{
   for (auto idx : fManagedValues) {
      fValues[idx].GetField()->DestroyValue(fValues[idx]);
   }
}

void ROOT::Experimental::RForestEntry::AddValue(const Detail::RFieldValue& value)
{
   fManagedValues.emplace_back(fValues.size());
   fValues.push_back(value);
}

void ROOT::Experimental::RForestEntry::CaptureValue(const Detail::RFieldValue& value)
{
   fValues.push_back(value);
}
