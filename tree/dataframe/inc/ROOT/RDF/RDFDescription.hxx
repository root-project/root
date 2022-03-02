// Author: Ivan Kabadzhov, Enrico Guiraud CERN  01/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFDescription
#define ROOT_RDFDescription

#include <string>

namespace ROOT {
namespace RDF {

/**
\class ROOT::RDF::RDFDescription
\ingroup dataframe
\brief A DFDescription contains useful information about a given RDataFrame computation graph.

 A DFDescription is returned by the Describe() RDataFrame method.
 Each DFDescription object can output either a brief or full description.
*/
class RDFDescription {

   std::string fBriefDescription;
   std::string fFullDescription;

public:
   RDFDescription(const std::string &briefDescription, const std::string &fullDescription);

   std::string AsString(bool shortFormat = false) const;

   void Print(bool shortFormat = false) const;

   friend std::ostream &operator<<(std::ostream &os, const RDFDescription &description);
};

} // namespace RDF
} // namespace ROOT

/// Print an RDFDescription at the prompt
namespace cling {
std::string printValue(ROOT::RDF::RDFDescription *td);
} // namespace cling

#endif
