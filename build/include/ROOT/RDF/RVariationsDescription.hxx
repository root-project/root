// Author: Enrico Guiraud CERN  02/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RVARIATIONSDESCRIPTION
#define ROOT_RDF_RVARIATIONSDESCRIPTION

#include <string>
#include <vector>
namespace ROOT {
namespace Internal {
namespace RDF {
class RVariationBase;
}
} // namespace Internal
} // namespace ROOT

namespace ROOT {
namespace RDF {

/// A descriptor for the systematic variations known to a given RDataFrame node.
class RVariationsDescription {
   std::string fStringRepr;
   using Variations_t = std::vector<const ROOT::Internal::RDF::RVariationBase *>;

public:
   RVariationsDescription(const Variations_t &variations);
   void Print() const;
   std::string AsString() const { return fStringRepr; }
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RVARIATIONSDESCRIPTION
