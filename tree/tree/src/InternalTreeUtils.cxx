/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/InternalTreeUtils.hxx"
#include "TBranch.h" // Usage of TBranch in ClearMustCleanupBits
#include "TChain.h"
#include "TCollection.h" // TRangeStaticCast
#include "TFile.h"
#include "TFriendElement.h"
#include "TTree.h"
#include "TVirtualIndex.h"

#include <cstdint> // std::uint64_t
#include <limits>
#include <utility> // std::pair
#include <vector>
#include <stdexcept> // std::runtime_error
#include <string>

// Recursively get the top level branches from the specified tree and all of its attached friends.
static void GetTopLevelBranchNamesImpl(TTree &t, std::unordered_set<std::string> &bNamesReg, std::vector<std::string> &bNames,
                                       std::unordered_set<TTree *> &analysedTrees, const std::string friendName = "")
{
   if (!analysedTrees.insert(&t).second) {
      return;
   }

   auto branches = t.GetListOfBranches();
   if (branches) {
      for (auto branchObj : *branches) {
         const auto name = branchObj->GetName();
         if (bNamesReg.insert(name).second) {
            bNames.emplace_back(name);
         } else if (!friendName.empty()) {
            // If this is a friend and the branch name has already been inserted, it might be because the friend
            // has a branch with the same name as a branch in the main tree. Let's add it as <friendname>.<branchname>.
            const auto longName = friendName + "." + name;
            if (bNamesReg.insert(longName).second)
               bNames.emplace_back(longName);
         }
      }
   }

   auto friendTrees = t.GetListOfFriends();

   if (!friendTrees)
      return;

   for (auto friendTreeObj : *friendTrees) {
      auto friendElement = static_cast<TFriendElement *>(friendTreeObj);
      auto friendTree = friendElement->GetTree();
      const std::string frName(friendElement->GetName()); // this gets us the TTree name or the friend alias if any
      GetTopLevelBranchNamesImpl(*friendTree, bNamesReg, bNames, analysedTrees, frName);
   }
}

namespace ROOT {
namespace Internal {
namespace TreeUtils {

///////////////////////////////////////////////////////////////////////////////
/// Get all the top-level branches names, including the ones of the friend trees
std::vector<std::string> GetTopLevelBranchNames(TTree &t)
{
   std::unordered_set<std::string> bNamesSet;
   std::vector<std::string> bNames;
   std::unordered_set<TTree *> analysedTrees;
   GetTopLevelBranchNamesImpl(t, bNamesSet, bNames, analysedTrees);
   return bNames;
}

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
/// \param[in] retrieveEntries Whether to also retrieve the number of entries in
///            each tree of each friend: one if the friend is a TTree, more if
///            the friend is a TChain. In the latter case, this function
///            triggers the opening of all files in the chain.
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
/// list of friends. The struct will contain four vectors, which elements at
/// position `i` represent the `i`-th friend of the input tree. If this friend
/// is a TTree, the `i`-th element of each of the three vectors will contain
/// respectively:
///
/// - A pair with the name and alias of the tree (the alias might not be
///   present, in which case it will be just an empty string).
/// - A vector with a single string representing the path to current file where
///   the tree is stored.
/// - An empty vector.
/// - A vector with a single element, the number of entries in the tree.
///
/// If the `i`-th friend is a TChain instead, the `i`-th element of each of the
/// three vectors will contain respectively:
/// - A pair with the name and alias of the chain (if present, both might be
///   empty strings).
/// - A vector with all the paths to the files contained in the chain.
/// - A vector with all the names of the trees making up the chain,
///   associated with the file names of the previous vector.
/// - A vector with the number of entries of each tree in the previous vector or
///   an empty vector, depending on whether \p retrieveEntries is true.
ROOT::TreeUtils::RFriendInfo GetFriendInfo(const TTree &tree, bool retrieveEntries)
{
   // Typically, the correct way to call GetListOfFriends would be `tree.GetTree()->GetListOfFriends()`
   // (see e.g. the discussion at https://github.com/root-project/root/issues/6741).
   // However, in this case, in case we are dealing with a TChain we really only care about the TChain's
   // list of friends (which will need to be rebuilt in each processing task) while friends of the TChain's
   // internal TTree, if any, will be automatically loaded in each task just like they would be automatically
   // loaded here if we used tree.GetTree()->GetListOfFriends().
   const auto *friends = tree.GetListOfFriends();
   if (!friends || friends->GetEntries() == 0)
      return ROOT::TreeUtils::RFriendInfo();

   std::vector<std::pair<std::string, std::string>> friendNames;
   std::vector<std::vector<std::string>> friendFileNames;
   std::vector<std::vector<std::string>> friendChainSubNames;
   std::vector<std::vector<std::int64_t>> nEntriesPerTreePerFriend;
   std::vector<std::unique_ptr<TVirtualIndex>> treeIndexes;

   // Reserve space for all friends
   auto nFriends = friends->GetEntries();
   friendNames.reserve(nFriends);
   friendFileNames.reserve(nFriends);
   friendChainSubNames.reserve(nFriends);
   nEntriesPerTreePerFriend.reserve(nFriends);

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

      // The vector of entries in each tree of the current friend.
      nEntriesPerTreePerFriend.emplace_back();
      auto &nEntriesInThisFriend = nEntriesPerTreePerFriend.back();

      // Check if friend tree/chain has an alias
      const auto *alias_c = tree.GetFriendAlias(frTree);
      const std::string alias = alias_c != nullptr ? alias_c : "";

      auto *treeIndex = frTree->GetTreeIndex();
      treeIndexes.emplace_back(static_cast<TVirtualIndex *>(treeIndex ? treeIndex->Clone() : nullptr));

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

         // Reserve space for this friend
         auto nFiles = chainFiles->GetEntries();
         fileNames.reserve(nFiles);
         chainSubNames.reserve(nFiles);
         nEntriesInThisFriend.reserve(nFiles);

         // Retrieve the name of the chain and add a (name, alias) pair
         friendNames.emplace_back(std::make_pair(frChain->GetName(), alias));
         // Each file in the chain can contain a TTree with a different name wrt
         // the main TChain. Retrieve the name of the file through `GetTitle`
         // and the name of the tree through `GetName`
         for (const auto *f : *chainFiles) {

            auto thisTreeName = f->GetName();
            auto thisFileName = f->GetTitle();

            chainSubNames.emplace_back(thisTreeName);
            fileNames.emplace_back(thisFileName);

            if (retrieveEntries) {
               std::unique_ptr<TFile> thisFile{TFile::Open(thisFileName, "READ_WITHOUT_GLOBALREGISTRATION")};
               if (!thisFile || thisFile->IsZombie())
                  throw std::runtime_error(std::string("GetFriendInfo: Could not open file \"") + thisFileName + "\"");
               TTree *thisTree = thisFile->Get<TTree>(thisTreeName);
               if (!thisTree)
                  throw std::runtime_error(std::string("GetFriendInfo: Could not retrieve TTree \"") + thisTreeName +
                                           "\" from file \"" + thisFileName + "\"");
               nEntriesInThisFriend.emplace_back(thisTree->GetEntries());
            } else {
               // Avoid odr-using TTree::kMaxEntries which would require a
               // definition in C++14. In C++17, all constexpr static data
               // members are implicitly inline.
               static constexpr auto maxEntries = TTree::kMaxEntries;
               nEntriesInThisFriend.emplace_back(maxEntries);
            }
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
         // We already have a pointer to the file and the tree, we can get the
         // entries without triggering a re-open
         nEntriesInThisFriend.emplace_back(frTree->GetEntries());
      }
   }

   return ROOT::TreeUtils::RFriendInfo(std::move(friendNames), std::move(friendFileNames),
                                       std::move(friendChainSubNames), std::move(nEntriesPerTreePerFriend),
                                       std::move(treeIndexes));
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
      std::string fullPath = treeDir->GetPath();            // e.g. "file.root:/dir"
      fullPath = fullPath.substr(fullPath.rfind(":/") + 1); // e.g. "/dir"
      fullPath += "/";
      fullPath += tree.GetName(); // e.g. "/dir/tree"
      return {fullPath};
   }

   // We do our best and return the name of the tree
   return {tree.GetName()};
}

/// Reset the kMustCleanup bit of a TObjArray of TBranch objects (e.g. returned by TTree::GetListOfBranches).
///
/// In some rare cases, all branches in a TTree can have their kMustCleanup bit set, which causes a large amount
/// of contention at teardown due to concurrent calls to RecursiveRemove (which needs to take the global lock).
/// This helper function checks the first branch of the array and if it has the kMustCleanup bit set, it resets
/// it for all branches in the array, recursively going through sub-branches and leaves.
void ClearMustCleanupBits(TObjArray &branches)
{
   if (branches.GetEntries() == 0 || branches.At(0)->TestBit(kMustCleanup) == false)
      return; // we assume either no branches have the bit set, or all do. we never encountered an hybrid case

   for (auto *branch : ROOT::Detail::TRangeStaticCast<TBranch>(branches)) {
      branch->ResetBit(kMustCleanup);
      TObjArray *subBranches = branch->GetListOfBranches();
      ClearMustCleanupBits(*subBranches);
      TObjArray *leaves = branch->GetListOfLeaves();
      if (leaves->GetEntries() > 0 && leaves->At(0)->TestBit(kMustCleanup) == true) {
         for (TObject *leaf : *leaves)
            leaf->ResetBit(kMustCleanup);
      }
   }
}

/// \brief Create a TChain object with options that avoid common causes of thread contention.
///
/// In particular, set its kWithoutGlobalRegistration mode and reset its kMustCleanup bit.
std::unique_ptr<TChain> MakeChainForMT(const std::string &name, const std::string &title)
{
   auto c = std::make_unique<TChain>(name.c_str(), title.c_str(), TChain::kWithoutGlobalRegistration);
   c->ResetBit(TObject::kMustCleanup);
   return c;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create friends from the main TTree.
std::vector<std::unique_ptr<TChain>> MakeFriends(const ROOT::TreeUtils::RFriendInfo &finfo)
{
   std::vector<std::unique_ptr<TChain>> friends;
   const auto nFriends = finfo.fFriendNames.size();
   friends.reserve(nFriends);

   for (std::size_t i = 0u; i < nFriends; ++i) {
      const auto &thisFriendName = finfo.fFriendNames[i].first;
      const auto &thisFriendFileNames = finfo.fFriendFileNames[i];
      const auto &thisFriendChainSubNames = finfo.fFriendChainSubNames[i];
      const auto &thisFriendEntries = finfo.fNEntriesPerTreePerFriend[i];

      // Build a friend chain
      auto frChain = ROOT::Internal::TreeUtils::MakeChainForMT(thisFriendName);
      if (thisFriendChainSubNames.empty()) {
         // The friend is a TTree. It's safe to add to the chain the filename directly.
         frChain->Add(thisFriendFileNames[0].c_str(), thisFriendEntries[0]);
      } else {
         // Otherwise, the new friend chain needs to be built using the nomenclature
         // "filename?#treename" as argument to `TChain::Add`
         for (std::size_t j = 0u; j < thisFriendFileNames.size(); ++j) {
            frChain->Add((thisFriendFileNames[j] + "?#" + thisFriendChainSubNames[j]).c_str(), thisFriendEntries[j]);
         }
      }

      auto &treeIndex = finfo.fTreeIndexInfos[i];
      if (treeIndex) {
         auto *copyOfIndex = static_cast<TVirtualIndex *>(treeIndex->Clone());
         copyOfIndex->SetTree(frChain.get());
         frChain->SetTreeIndex(copyOfIndex);
      }

      friends.emplace_back(std::move(frChain));
   }

   return friends;
}

} // namespace TreeUtils
} // namespace Internal
} // namespace ROOT
