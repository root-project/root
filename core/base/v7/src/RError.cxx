/// \file RError.cxx
/// \ingroup Base ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RError.hxx"

#include <string>

std::string ROOT::Experimental::RError::GetReport() const
{
   auto report = fMessage + "\nAt:\n";
   for (const auto &loc : fStackTrace) {
      report += "  " + loc.fFunction + " [" + loc.fSourceFile + ":" + std::to_string(loc.fSourceLine) + "]\n";
   }
   return report;
}

ROOT::Experimental::RError::RError(
   const std::string &message, const std::string &func, const std::string &file, int line)
   : fMessage(message)

{
   AddFrame(func, file, line);
}

void ROOT::Experimental::RError::AddFrame(const std::string &func, const std::string &file, int line)
{
   fStackTrace.emplace_back(RLocation(func, file, line));
}
