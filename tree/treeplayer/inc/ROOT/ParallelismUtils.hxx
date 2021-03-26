/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup Parallelism Parallelism
Functions and classes that allow parallel execution of ROOT code.
*/

/**
 \file ROOT/ParallelismUtils.hxx
 \ingroup Parallelism
 \author Vincenzo Eduardo Padulano
 \date 2021-03
*/

#ifndef PARALLELISM_UTILS_H
#define PARALLELISM_UTILS_H

#include <utility> // std::pair
#include <vector>
#include <string>

class TTree;

namespace ROOT {
namespace Internal {
/**
\namespace ROOT::Internal::Parallelism
\ingroup Parallelism
\brief inline namespace hosting functions and classes that help in parallel workflows.
*/
inline namespace Parallelism {

using NameAlias = std::pair<std::string, std::string>; ///< A pair of name and alias of a TTree's friend tree.
/**
\struct ROOT::Internal::Parallelism::FriendInfo
\brief Information about friend trees of a certain TTree or TChain object.
\ingroup Parallelism
*/
struct FriendInfo {
   std::vector<NameAlias> fFriendNames;                    ///< Pairs of names and aliases of friend trees/chains
   std::vector<std::vector<std::string>> fFriendFileNames; ///< Names of the files where each friend is stored.
                                                           ///< fFriendFileNames[i] is the list of files for friend with
                                                           ///< name fFriendNames[i]
};

std::vector<std::string> GetTreeFullPaths(const TTree &tree);
FriendInfo GetFriendInfo(const TTree &tree);

} // namespace Parallelism
} // namespace Internal
} // namespace ROOT

#endif // PARALLELISM_UTILS_H