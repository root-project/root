// @(#)root/thread:$Id$
// Authors: Enric Tejedor, Enrico Guiraud CERN 05/06/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeProcessorMT
#define ROOT_TTreeProcessorMT

#include "TKey.h"
#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
#include "TTreeReader.h"
#include "TError.h"
#include "TEntryList.h"
#include "TFriendElement.h"
#include "ROOT/RMakeUnique.hxx"
#include "ROOT/TThreadedObject.hxx"

#include <string.h>
#include <functional>
#include <vector>


/** \class TTreeView
    \brief A helper class that encapsulates a file and a tree.

A helper class that encapsulates a TFile and a TTree, along with their names.
It is used together with TTProcessor and ROOT::TThreadedObject, so that
in the TTProcessor::Process method each thread can work on its own
<TFile,TTree> pair.

This class can also be used with a collection of file names or a TChain, in case
the tree is stored in more than one file. A view will always contain only the
current (active) tree and file objects.

A copy constructor is defined for TTreeView to work with ROOT::TThreadedObject.
The latter makes a copy of a model object every time a new thread accesses
the threaded object.
*/

namespace ROOT {
   namespace Internal {

      /// A cluster of entries
      struct EntryCluster {
         Long64_t start;
         Long64_t end;
      };

      /// Names, aliases, and file names of a TTree's or TChain's friends
      using NameAlias = std::pair<std::string, std::string>;
      struct FriendInfo {
         /// Pairs of names and aliases of friend trees/chains
         std::vector<Internal::NameAlias> fFriendNames;
         /// Names of the files where each friend is stored. fFriendFileNames[i] is the list of files for friend with
         /// name fFriendNames[i]
         std::vector<std::vector<std::string>> fFriendFileNames;
      };

      class TTreeView {
      private:
         using TreeReaderEntryListPair = std::pair<std::unique_ptr<TTreeReader>, std::unique_ptr<TEntryList>>;

         // NOTE: fFriends must come before fChain to be deleted after it, see ROOT-9281 for more details
         std::vector<std::unique_ptr<TChain>> fFriends; ///< Friends of the tree/chain
         std::unique_ptr<TChain> fChain;                ///< Chain on which to operate

         ////////////////////////////////////////////////////////////////////////////////
         /// Construct fChain, also adding friends if needed and injecting knowledge of offsets if available.
         void MakeChain(const std::string &treeName, const std::vector<std::string> &fileNames,
                        const FriendInfo &friendInfo, const std::vector<Long64_t> &nEntries,
                        const std::vector<std::vector<Long64_t>> &friendEntries)
         {
            const std::vector<NameAlias> &friendNames = friendInfo.fFriendNames;
            const std::vector<std::vector<std::string>> &friendFileNames = friendInfo.fFriendFileNames;

            fChain.reset(new TChain(treeName.c_str()));
            const auto nFiles = fileNames.size();
            for (auto i = 0u; i < nFiles; ++i) {
               fChain->Add(fileNames[i].c_str(), nEntries[i]);
            }
            fChain->ResetBit(TObject::kMustCleanup);

            fFriends.clear();
            const auto nFriends = friendNames.size();
            for (auto i = 0u; i < nFriends; ++i) {
               const auto &friendName = friendNames[i];
               const auto &name = friendName.first;
               const auto &alias = friendName.second;

               // Build a friend chain
               auto frChain = std::make_unique<TChain>(name.c_str());
               const auto nFileNames = friendFileNames[i].size();
               for (auto j = 0u; j < nFileNames; ++j)
                  frChain->Add(friendFileNames[i][j].c_str(), friendEntries[i][j]);

               // Make it friends with the main chain
               fChain->AddFriend(frChain.get(), alias.c_str());
               fFriends.emplace_back(std::move(frChain));
            }
         }

         TreeReaderEntryListPair MakeReaderWithEntryList(TEntryList &globalList, Long64_t start, Long64_t end)
         {
            // TEntryList and SetEntriesRange do not work together (the former has precedence).
            // We need to construct a TEntryList that contains only those entry numbers in our desired range.
            auto localList = std::make_unique<TEntryList>();
            Long64_t entry = globalList.GetEntry(0);
            do {
               if (entry >= end)
                  break;
               else if (entry >= start)
                  localList->Enter(entry);
            } while ((entry = globalList.Next()) >= 0);

            auto reader = std::make_unique<TTreeReader>(fChain.get(), localList.get());
            return std::make_pair(std::move(reader), std::move(localList));
         }

         std::unique_ptr<TTreeReader> MakeReader(Long64_t start, Long64_t end)
         {
            auto reader = std::make_unique<TTreeReader>(fChain.get());
            reader->SetEntriesRange(start, end);
            return reader;
         }

      public:
         TTreeView() {}

         // no-op, we don't want to copy the local TChains
         TTreeView(const TTreeView &) {}

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the current tree of this view.
         TreeReaderEntryListPair GetTreeReader(Long64_t start, Long64_t end, const std::string &treeName,
                                               const std::vector<std::string> &fileNames, const FriendInfo &friendInfo,
                                               TEntryList entryList, const std::vector<Long64_t> &nEntries,
                                               const std::vector<std::vector<Long64_t>> &friendEntries)
         {
            const bool usingLocalEntries = friendInfo.fFriendNames.empty() && entryList.GetN() == 0;
            if (fChain == nullptr || (usingLocalEntries && fileNames[0] != fChain->GetListOfFiles()->At(0)->GetTitle()))
               MakeChain(treeName, fileNames, friendInfo, nEntries, friendEntries);

            std::unique_ptr<TTreeReader> reader;
            std::unique_ptr<TEntryList> localList;
            if (entryList.GetN() > 0) {
               std::tie(reader, localList) = MakeReaderWithEntryList(entryList, start, end);
            } else {
               reader = MakeReader(start, end);
            }

            // we need to return the entry list too, as it needs to be in scope as long as the reader is
            return std::make_pair(std::move(reader), std::move(localList));
         }
      };
   } // End of namespace Internal

   class TTreeProcessorMT {
   private:
      const std::vector<std::string> fFileNames; ///< Names of the files
      const std::string fTreeName;               ///< Name of the tree
      /// User-defined selection of entry numbers to be processed, empty if none was provided
      const TEntryList fEntryList; // const to be sure to avoid race conditions among TTreeViews
      const Internal::FriendInfo fFriendInfo;

      ROOT::TThreadedObject<ROOT::Internal::TTreeView> fTreeView; ///<! Thread-local TreeViews

      Internal::FriendInfo GetFriendInfo(TTree &tree);
      std::string FindTreeName();
      static unsigned int fgMaxTasksPerFilePerWorker;
   public:
      TTreeProcessorMT(std::string_view filename, std::string_view treename = "");
      TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename = "");
      TTreeProcessorMT(TTree &tree, const TEntryList &entries);
      TTreeProcessorMT(TTree &tree);

      void Process(std::function<void(TTreeReader &)> func);
      static void SetMaxTasksPerFilePerWorker(unsigned int m);
      static unsigned int GetMaxTasksPerFilePerWorker();
   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
