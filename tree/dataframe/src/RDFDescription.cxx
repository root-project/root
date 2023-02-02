// Author: Ivan Kabadzhov, Enrico Guiraud CERN  01/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDFDescription.hxx"
#include <iostream>

namespace ROOT {
namespace RDF {

RDFDescription::RDFDescription(const std::string &briefDescription, const std::string &fullDescription)
   : fBriefDescription(briefDescription), fFullDescription(fullDescription){};

std::string RDFDescription::AsString(bool shortFormat /*= false*/) const
{
   if (shortFormat)
      return fBriefDescription;
   else
      return fBriefDescription + "\n\n" + fFullDescription;
}

void RDFDescription::Print(bool shortFormat /*= false*/) const
{
   std::cout << AsString(shortFormat);
}

std::ostream &operator<<(std::ostream &os, const RDFDescription &description)
{
   os << description.AsString();
   return os;
}

} // namespace RDF
} // namespace ROOT

namespace cling {
//////////////////////////////////////////////////////////////////////////
/// Print an RDFDescription at the prompt
std::string printValue(ROOT::RDF::RDFDescription *td)
{
   return td->AsString();
}

} // namespace cling
