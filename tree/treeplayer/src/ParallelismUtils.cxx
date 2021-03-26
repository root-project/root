/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/ParallelismUtils.hxx"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TFriendElement.h"

#include <utility> // std::pair
#include <vector>
#include <string>

namespace ROOT {
namespace Internal {
inline namespace Parallelism {

////////////////////////////////////////////////////////////////////////////////
/// \fn std::vector<std::string> GetTreeFullPaths(const TTree &tree)
/// \ingroup Parallelism
/// \brief Retrieve the full path(s) to a TTree or the trees in a TChain.
/// \param[in] tree The tree or chain from which the paths will be retrieved.
/// \return If the input argument is a TChain, returns a vector of strings with
///         the name of the tree of each file in the chain. If the input
///         argument is a TTree, returns a vector with a single element that is
///         the full path of the tree in the file (e.g. the name of the tree
///         itself or the path with the directories inside the file). Finally,
///         the function returns just the name of the tree if it couldn't do any
///         better.
std::vector<std::string> GetTreeFullPaths(const TTree &tree)
{
   // Case 1: this is a TChain. For each file it contains, GetName returns the name of the tree in that file
   if (tree.IsA() == TChain::Class()) {
      auto &chain = static_cast<const TChain &>(tree);
      auto files = chain.GetListOfFiles();
      if (!files || files->GetEntries() == 0) {
         throw std::runtime_error("Input TChain does not contain any file");
      }
      std::vector<std::string> treeNames;
      for (TObject *f : *files)
         treeNames.emplace_back(f->GetName());

      return treeNames;
   }

   // Case 2: this is a TTree: we get the full path of it
   if (auto motherDir = tree.GetDirectory()) {
      // We have 2 subcases (ROOT-9948):
      // - 1. motherDir is a TFile
      // - 2. motherDir is a directory
      // If 1. we just return the name of the tree, if 2. we reconstruct the path
      // to the file.
      if (motherDir->InheritsFrom("TFile")) {
         return {tree.GetName()};
      }
      std::string fullPath = motherDir->GetPath();         // e.g. "file.root:/dir"
      fullPath = fullPath.substr(fullPath.find(":/") + 1); // e.g. "/dir"
      fullPath += "/";
      fullPath += tree.GetName(); // e.g. "/dir/tree"
      return {fullPath};
   }

   // We do our best and return the name of the tree
   return {tree.GetName()};
}

////////////////////////////////////////////////////////////////////////////////
/// \fn FriendInfo GetFriendInfo(const TTree &tree)
/// \ingroup Parallelism
/// \brief Get and store the names, aliases and file names of the friends of the tree.
/// \param[in] tree The tree from which friends information will be gathered.
///
/// \note "friends of friends" and circular references in the lists of friends
///       are not supported.
FriendInfo GetFriendInfo(const TTree &tree)
{
   std::vector<NameAlias> friendNames;
   std::vector<std::vector<std::string>> friendFileNames;

   // Typically, the correct way to call GetListOfFriends would be `tree.GetTree()->GetListOfFriends()`
   // (see e.g. the discussion at https://github.com/root-project/root/issues/6741).
   // However, in this case, in case we are dealing with a TChain we really only care about the TChain's
   // list of friends (which will need to be rebuilt in each processing task) while friends of the TChain's
   // internal TTree, if any, will be automatically loaded in each task just like they would be automatically
   // loaded here if we used tree.GetTree()->GetListOfFriends().
   const auto friends = tree.GetListOfFriends();
   if (!friends)
      return FriendInfo();

   for (auto fr : *friends) {
      const auto frTree = static_cast<TFriendElement *>(fr)->GetTree();
      const bool isChain = frTree->IsA() == TChain::Class();

      friendFileNames.emplace_back();
      auto &fileNames = friendFileNames.back();

      // Check if friend tree/chain has an alias
      const auto alias_c = tree.GetFriendAlias(frTree);
      const std::string alias = alias_c != nullptr ? alias_c : "";

      if (isChain) {
         // Note that each TChainElement returned by chain.GetListOfFiles has a name
         // equal to the tree name of this TChain and a title equal to the filename.
         // Accessing the information like this ensures that we get the correct
         // filenames and treenames if the treename is given as part of the filename
         // via chain.AddFile(file.root/myTree) and as well if the tree name is given
         // in the constructor via TChain(myTree) and a file is added later by chain.AddFile(file.root).

         // Get name of the trees building the chain
         const auto chainFiles = static_cast<TChain *>(frTree)->GetListOfFiles();
         const auto realName = chainFiles->First()->GetName();
         friendNames.emplace_back(std::make_pair(realName, alias));
         // Get filenames stored in the title member
         for (auto f : *chainFiles) {
            fileNames.emplace_back(f->GetTitle());
         }
      } else {
         // Get name of the tree
         const auto realName = GetTreeFullPaths(*frTree)[0];
         friendNames.emplace_back(std::make_pair(realName, alias));

         // Get filename
         const auto f = frTree->GetCurrentFile();
         if (!f)
            throw std::runtime_error("Friend trees with no associated file are not supported.");
         fileNames.emplace_back(f->GetName());
      }
   }

   return FriendInfo{std::move(friendNames), std::move(friendFileNames)};
}

} // namespace Parallelism
} // namespace Internal
} // namespace ROOT
