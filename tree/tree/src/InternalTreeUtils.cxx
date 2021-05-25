/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/InternalTreeUtils.hxx"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TFriendElement.h"

#include <utility> // std::pair
#include <vector>
#include <stdexcept> // std::runtime_error
#include <string>

namespace ROOT {
namespace Internal {
namespace TreeUtils {

////////////////////////////////////////////////////////////////////////////////
/// \fn std::vector<std::string> GetFileNamesFromTree(const TTree &tree)
/// \ingroup tree
/// \brief Get and store the file names associated with the input tree.
/// \param[in] tree The tree from which friends information will be gathered.
/// \throws std::runtime_error If no files could be associated with the input tree.
std::vector<std::string> GetFileNamesFromTree(const TTree &tree)
{
   std::vector<std::string> filenames;

   // If the input tree is a TChain, traverse its list of associated files.
   if (auto chain = dynamic_cast<const TChain *>(&tree)) {
      const auto *chainFiles = chain->GetListOfFiles();
      if (!chainFiles) {
         throw std::runtime_error("Could not retrieve a list of files from the input TChain.");
      }
      // Store this in a variable so it can be later used in `filenames.reserve`
      // if it passes the check.
      const auto nfiles = chainFiles->GetEntries();
      if (nfiles == 0) {
         throw std::runtime_error("The list of files associated with the input TChain is empty.");
      }
      filenames.reserve(nfiles);
      for (const auto *f : *chainFiles)
         filenames.emplace_back(f->GetTitle());
   } else {
      const TFile *f = tree.GetCurrentFile();
      if (!f) {
         throw std::runtime_error("The input TTree is not linked to any file, "
                                  "in-memory-only trees are not supported.");
      }

      filenames.emplace_back(f->GetName());
   }

   return filenames;
}

////////////////////////////////////////////////////////////////////////////////
/// \fn RFriendInfo GetFriendInfo(const TTree &tree)
/// \ingroup tree
/// \brief Get and store the names, aliases and file names of the direct friends of the tree.
/// \param[in] tree The tree from which friends information will be gathered.
/// \throws std::runtime_error If the input tree has a list of friends, but any
///         of them could not be associated with any file.
///
/// Calls TTree::GetListOfFriends and parses its result for the names, aliases
/// and file names, with different methodologies depending on whether the
/// parameter is a TTree or a TChain.
///
/// \note This function only retrieves information about <b>direct friends</b>
///       of the input tree. It will not recurse through friends of friends and
///       does not take into account circular references in the list of friends
///       of the input tree.
///
/// \returns An RFriendInfo struct, containing the information parsed from the
/// list of friends. The struct will contain three vectors, which elements at
/// position `i` represent the `i`-th friend of the input tree. If this friend
/// is a TTree, the `i`-th element of each of the three vectors will contain
/// respectively:
/// - A pair with the name and alias of the tree (the alias might not be
///   present, in which case it will be just an empty string).
/// - A vector with a single string representing the path to current file where
///   the tree is stored.
/// - An empty vector.
/// .
/// If the `i`-th friend is a TChain instead, the `i`-th element of each of the
/// three vectors will contain respectively:
/// - A pair with the name and alias of the chain (if present, both might be
///   empty strings).
/// - A vector with all the paths to the files contained in the chain.
/// - A vector with all the the names of the trees making up the chain,
///   associated with the file names of the previous vector.
RFriendInfo GetFriendInfo(const TTree &tree)
{
   std::vector<NameAlias> friendNames;
   std::vector<std::vector<std::string>> friendFileNames;
   std::vector<std::vector<std::string>> friendChainSubNames;

   // Typically, the correct way to call GetListOfFriends would be `tree.GetTree()->GetListOfFriends()`
   // (see e.g. the discussion at https://github.com/root-project/root/issues/6741).
   // However, in this case, in case we are dealing with a TChain we really only care about the TChain's
   // list of friends (which will need to be rebuilt in each processing task) while friends of the TChain's
   // internal TTree, if any, will be automatically loaded in each task just like they would be automatically
   // loaded here if we used tree.GetTree()->GetListOfFriends().
   const auto *friends = tree.GetListOfFriends();
   if (!friends)
      return RFriendInfo();

   for (auto fr : *friends) {
      // Can't pass fr as const TObject* because TFriendElement::GetTree is not const.
      // Also, we can't retrieve frTree as const TTree* because of TTree::GetFriendAlias(TTree *) a few lines later
      auto frTree = static_cast<TFriendElement *>(fr)->GetTree();

      // The vector of (name,alias) pairs of the current friend
      friendFileNames.emplace_back();
      auto &fileNames = friendFileNames.back();

      // The vector of names of sub trees of the current friend, if it is a TChain.
      // Otherwise, just an empty vector.
      friendChainSubNames.emplace_back();
      auto &chainSubNames = friendChainSubNames.back();

      // Check if friend tree/chain has an alias
      const auto *alias_c = tree.GetFriendAlias(frTree);
      const std::string alias = alias_c != nullptr ? alias_c : "";

      // If the current tree is a TChain
      if (auto frChain = dynamic_cast<const TChain *>(frTree)) {
         // Note that each TChainElement returned by TChain::GetListOfFiles has a name
         // equal to the tree name of this TChain and a title equal to the filename.
         // Accessing the information like this ensures that we get the correct
         // filenames and treenames if the treename is given as part of the filename
         // via chain.AddFile(file.root/myTree) and as well if the tree name is given
         // in the constructor via TChain(myTree) and a file is added later by chain.AddFile(file.root).
         // Caveat: The chain may be made of sub-trees with different names. All
         // tree names need to be retrieved separately, see below.

         // Get filelist of the current chain
         const auto *chainFiles = frChain->GetListOfFiles();
         if (!chainFiles || chainFiles->GetEntries() == 0) {
            throw std::runtime_error("A TChain in the list of friends does not contain any file. "
                                     "Friends with no associated files are not supported.");
         }

         // Retrieve the name of the chain and add a (name, alias) pair
         friendNames.emplace_back(std::make_pair(frChain->GetName(), alias));
         // Each file in the chain can contain a TTree with a different name wrt
         // the main TChain. Retrieve the name of the file through `GetTitle`
         // and the name of the tree through `GetName`
         for (const auto *f : *chainFiles) {
            chainSubNames.emplace_back(f->GetName());
            fileNames.emplace_back(f->GetTitle());
         }
      } else {
         // Get name of the tree
         const auto realName = GetTreeFullPaths(*frTree)[0];
         friendNames.emplace_back(std::make_pair(realName, alias));

         // Get filename
         const auto *f = frTree->GetCurrentFile();
         if (!f)
            throw std::runtime_error("A TTree in the list of friends is not linked to any file. "
                                     "Friends with no associated files are not supported.");
         fileNames.emplace_back(f->GetName());
      }
   }

   return RFriendInfo{std::move(friendNames), std::move(friendFileNames), std::move(friendChainSubNames)};
}

////////////////////////////////////////////////////////////////////////////////
/// \fn std::vector<std::string> GetTreeFullPaths(const TTree &tree)
/// \ingroup tree
/// \brief Retrieve the full path(s) to a TTree or the trees in a TChain.
/// \param[in] tree The tree or chain from which the paths will be retrieved.
/// \throws std::runtime_error If the input tree is a TChain but no files could
///         be found associated with it.
/// \return If the input argument is a TChain, returns a vector of strings with
///         the name of the tree of each file in the chain. If the input
///         argument is a TTree, returns a vector with a single element that is
///         the full path of the tree in the file (e.g. the name of the tree
///         itself or the path with the directories inside the file). Finally,
///         the function returns a vector with just the name of the tree if it
///         couldn't do any better.
std::vector<std::string> GetTreeFullPaths(const TTree &tree)
{
   // Case 1: this is a TChain. For each file it contains, GetName returns the name of the tree in that file
   if (auto chain = dynamic_cast<const TChain *>(&tree)) {
      const auto *chainFiles = chain->GetListOfFiles();
      if (!chainFiles || chainFiles->GetEntries() == 0) {
         throw std::runtime_error("The input TChain does not contain any file.");
      }
      std::vector<std::string> treeNames;
      for (const auto *f : *chainFiles)
         treeNames.emplace_back(f->GetName());

      return treeNames;
   }

   // Case 2: this is a TTree: we get the full path of it
   if (const auto *treeDir = tree.GetDirectory()) {
      // We have 2 subcases (ROOT-9948):
      // - 1. treeDir is a TFile: return the name of the tree.
      // - 2. treeDir is a directory: reconstruct the path to the tree in the directory.
      // Use dynamic_cast to check whether the directory is a TFile
      if (dynamic_cast<const TFile *>(treeDir)) {
         return {tree.GetName()};
      }
      std::string fullPath = treeDir->GetPath();           // e.g. "file.root:/dir"
      fullPath = fullPath.substr(fullPath.find(":/") + 1); // e.g. "/dir"
      fullPath += "/";
      fullPath += tree.GetName(); // e.g. "/dir/tree"
      return {fullPath};
   }

   // We do our best and return the name of the tree
   return {tree.GetName()};
}

} // namespace TreeUtils
} // namespace Internal
} // namespace ROOT
