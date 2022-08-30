// Author: Vincenzo Eduardo Padulano CERN/UPV 08/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

#include "ROOT/InternalTreeUtils.hxx"
#include "ROOT/RDF/InternalUtils.hxx"
#include "ROOT/RDF/RInterface.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

////////////////////////////////////////////////////////////////////////////////
/// \brief Gets information about the internal tree in the dataframe.
/// \param[in] node A node of the computation graph.
/// \returns The name of the dataset, the filenames and the tree names inside
///          the files. If the RDataFrame is not processing a TTree-based
///          dataset, returns nullptr.
std::unique_ptr<RTreeInfo> MakeTreeInfo(const ROOT::RDF::RNode &node)
{
   const auto *tree = node.GetLoopManager()->GetTree();
   if (!tree)
      return nullptr;
   return std::unique_ptr<RTreeInfo>{new RTreeInfo{std::string{tree->GetName()},
                                                   ROOT::Internal::TreeUtils::GetFileNamesFromTree(*tree),
                                                   ROOT::Internal::TreeUtils::GetTreeFullPaths(*tree)}};
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
