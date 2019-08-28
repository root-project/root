/// \file RNTupleMetrics.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleMetrics.hxx>

#include <TError.h>

#include <algorithm>
#include <ctime>
#include <ostream>
#include <utility>

std::string ROOT::Experimental::Detail::RNTupleTickCounter::ToString() const {
   auto clocks = GetValue();
   return std::to_string(std::uint64_t(
      (double(clocks) / double(CLOCKS_PER_SEC)) / (1000. * 1000. * 1000.)));
}

ROOT::Experimental::Detail::RNTuplePerfCounter *
ROOT::Experimental::Detail::RNTupleMetrics::Lookup(const std::string &name) const
{
   const auto itr = std::find(fCounterInfos.begin(), fCounterInfos.end(), name);
   if (itr == fCounterInfos.end())
      return nullptr;
   return fCounters[std::distance(fCounterInfos.begin(), itr)].get();
}

void ROOT::Experimental::Detail::RNTupleMetrics::Print(std::ostream &output) const
{
   const unsigned int N = fCounters.size();
   for (unsigned int i = 0; i < N; ++i) {
      output << fName << "." << fCounterInfos[i].fName << "|" << fCounterInfos[i].fUnit << "|"
             << fCounterInfos[i].fDescription << "|" << fCounters[i]->ToString() << std::endl;
   }
}

void ROOT::Experimental::Detail::RNTupleMetrics::Activate()
{
   for (auto &c: fCounters)
      c->Activate();
   fIsActive = true;
}
