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

#include <ostream>

bool ROOT::Experimental::Detail::RNTupleMetrics::Contains(const std::string &name) const
{
   for (const auto &c : fCounters) {
      if (c->GetName() == name)
         return true;
   }
   return false;
}

void ROOT::Experimental::Detail::RNTupleMetrics::Print(std::ostream &output, const std::string &prefix) const
{

   for (const auto &c : fCounters) {
      output << prefix << fName << "." << c->GetName() << "|" << c->GetUnit() << "|" << c->GetDescription()
             << "|" << c->ToString() << std::endl;
   }
   for (const auto c : fObservedMetrics) {
      c->Print(output, prefix + fName + ".");
   }
}

void ROOT::Experimental::Detail::RNTupleMetrics::Enable()
{
   for (auto &c: fCounters)
      c->Enable();
   fIsEnabled = true;
   for (auto m: fObservedMetrics)
      m->Enable();
}

void ROOT::Experimental::Detail::RNTupleMetrics::ObserveMetrics(RNTupleMetrics &observee)
{
   fObservedMetrics.push_back(&observee);
}
