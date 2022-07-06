// Author: Vincenzo Eduardo Padulano CERN/UPV 08/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_INTERNALUTILS
#define ROOT_RDF_INTERNALUTILS

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Detail {
namespace RDF {
class RNodeBase;
} // namespace RDF
} // namespace Detail
} // namespace ROOT

namespace ROOT {
namespace RDF {
template <typename T, typename V>
class RInterface;
using RNode = RInterface<::ROOT::Detail::RDF::RNodeBase, void>;
} // namespace RDF
} // namespace ROOT

namespace ROOT {
namespace Internal {
namespace RDF {

/**
\struct ROOT::Internal::RDF::RTreeInfo
\brief Information about the trees to be processed by an RDataFrame.
\ingroup dataframe
*/
struct RTreeInfo {
   std::string fTreeName;
   std::vector<std::string> fFileNames;
   std::vector<std::string> fTreeNamesInFiles;
};

std::unique_ptr<RTreeInfo> MakeTreeInfo(const ROOT::RDF::RNode &node);

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_INTERNALUTILS
