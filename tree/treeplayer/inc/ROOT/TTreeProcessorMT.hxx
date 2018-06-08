// @(#)root/thread:$Id$
// Author: Enric Tejedor, CERN  12/09/2016

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

      using ClustersAndEntries = std::pair<std::vector<EntryCluster>, std::vector<Long64_t>>;
      ClustersAndEntries MakeClusters(const std::string &treename, const std::vector<std::string> &filenames);

      class TTreeView {
      private:
         using TreeReaderEntryListPair = std::pair<std::unique_ptr<TTreeReader>, std::unique_ptr<TEntryList>>;
         using NameAlias = std::pair<std::string, std::string>;

         // NOTE: fFriends must come before fChain to be deleted after it, see ROOT-9281 for more details
         std::vector<std::unique_ptr<TChain>> fFriends; ///< Friends of the tree/chain
         std::unique_ptr<TChain> fChain;                ///< Chain on which to operate
         std::vector<std::string> fFileNames;           ///< Names of the files
         std::string fTreeName;                         ///< Name of the tree
         TEntryList fEntryList; ///< User-defined selection of entry numbers to be processed, empty if none was provided
         std::vector<Long64_t> fLoadedEntries;          ///<! Per-task loaded entries (for task interleaving)
         std::vector<NameAlias> fFriendNames;           ///< <name,alias> pairs of the friends of the tree/chain
         std::vector<std::vector<std::string>> fFriendFileNames; ///< Names of the files where friends are stored

         ////////////////////////////////////////////////////////////////////////////////
         /// If not treeName was provided to the ctor, use the name of the first TTree in the first file, else throw.
         void GetTreeNameIfNeeded()
         {
            // If the tree name is empty, look for a tree in the file
            if (fTreeName.empty()) {
               ::TDirectory::TContext ctxt(gDirectory);
               std::unique_ptr<TFile> f(TFile::Open(fFileNames[0].c_str()));
               TIter next(f->GetListOfKeys());
               while (TKey *key = (TKey *)next()) {
                  const char *className = key->GetClassName();
                  if (strcmp(className, "TTree") == 0) {
                     fTreeName = key->GetName();
                     break;
                  }
               }
               if (fTreeName.empty()) {
                  auto msg = "Cannot find any tree in file " + fFileNames[0];
                  throw std::runtime_error(msg);
               }
            }
         }

         ////////////////////////////////////////////////////////////////////////////////
         /// Construct fChain, also adding friends if needed and injecting knowledge of offsets if available.
         void MakeChain(const std::vector<Long64_t> &nEntries, const std::vector<std::vector<Long64_t>> &friendEntries)
         {
            fChain.reset(new TChain(fTreeName.c_str()));
            const auto nFiles = fFileNames.size();
            for (auto i = 0u; i < nFiles; ++i) {
               fChain->Add(fFileNames[i].c_str(), nEntries[i]);
            }
            fChain->ResetBit(TObject::kMustCleanup);

            fFriends.clear();
            const auto nFriends = fFriendNames.size();
            for (auto i = 0u; i < nFriends; ++i) {
               const auto &friendName = fFriendNames[i];
               const auto &name = friendName.first;
               const auto &alias = friendName.second;

               // Build a friend chain
               auto frChain = std::make_unique<TChain>(name.c_str());
               const auto nFileNames = fFriendFileNames[i].size();
               for (auto j = 0u; j < nFileNames; ++j)
                  frChain->Add(fFriendFileNames[i][j].c_str(), friendEntries[i][j]);

               // Make it friends with the main chain
               fChain->AddFriend(frChain.get(), alias.c_str());
               fFriends.emplace_back(std::move(frChain));
            }
         }

         ////////////////////////////////////////////////////////////////////////////////
         /// Get and store the names, aliases and file names of the friends of the tree.
         void StoreFriends(const TTree &tree, bool isTree)
         {
            auto friends = tree.GetListOfFriends();
            if (!friends)
               return;

            for (auto fr : *friends) {
               auto frTree = static_cast<TFriendElement *>(fr)->GetTree();

               // Check if friend tree has an alias
               auto realName = frTree->GetName();
               auto alias = tree.GetFriendAlias(frTree);
               if (alias) {
                  fFriendNames.emplace_back(std::make_pair(realName, std::string(alias)));
               } else {
                  fFriendNames.emplace_back(std::make_pair(realName, ""));
               }

               // Store the file names of the friend tree
               fFriendFileNames.emplace_back();
               auto &fileNames = fFriendFileNames.back();
               if (isTree) {
                  auto f = frTree->GetCurrentFile();
                  fileNames.emplace_back(f->GetName());
               } else {
                  auto frChain = static_cast<TChain *>(frTree);
                  for (auto f : *(frChain->GetListOfFiles())) {
                     fileNames.emplace_back(f->GetTitle());
                  }
               }
            }
         }

         TreeReaderEntryListPair MakeReaderWithEntryList(Long64_t start, Long64_t end)
         {
            // TEntryList and SetEntriesRange do not work together (the former has precedence).
            // We need to construct a TEntryList that contains only those entry numbers in our desired range.
            auto elist = std::make_unique<TEntryList>();
            Long64_t entry = fEntryList.GetEntry(0);
            do {
               if (entry >= start && entry < end) // TODO can quit this loop early when entry >= end
                  elist->Enter(entry);
            } while ((entry = fEntryList.Next()) >= 0);

            auto reader = std::make_unique<TTreeReader>(fChain.get(), elist.get());
            return std::make_pair(std::move(reader), std::move(elist));
         }

         std::unique_ptr<TTreeReader> MakeReader(Long64_t start, Long64_t end)
         {
            auto reader = std::make_unique<TTreeReader>(fChain.get());
            fChain->LoadTree(start - 1);
            reader->SetEntriesRange(start, end);
            return reader;
         }

      public:
         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a file name.
         /// \param[in] fn Name of the file containing the tree to process.
         /// \param[in] tn Name of the tree to process. If not provided,
         ///               the implementation will automatically search for a
         ///               tree in the file.
         TTreeView(std::string_view fn, std::string_view tn) : fTreeName(tn)
         {
            fFileNames.emplace_back(fn);
            GetTreeNameIfNeeded();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a collection of file names.
         /// \param[in] fns Collection of file names containing the tree to process.
         /// \param[in] tn Name of the tree to process. If not provided,
         ///               the implementation will automatically search for a
         ///               tree in the collection of files.
         TTreeView(const std::vector<std::string_view>& fns, std::string_view tn) : fTreeName(tn)
         {
            if (fns.size() > 0) {
               for (auto& fn : fns)
                  fFileNames.emplace_back(fn);
            }
            else {
               auto msg = "The provided list of file names is empty, cannot process tree " + fTreeName;
               throw std::runtime_error(msg);
            }
            GetTreeNameIfNeeded();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a TTree.
         /// \param[in] tree Tree or chain of files containing the tree to process.
         TTreeView(TTree& tree) : fTreeName(tree.GetName())
         {
            static const TClassRef clRefTChain("TChain");
            if (clRefTChain == tree.IsA()) {
               TObjArray* filelist = static_cast<TChain&>(tree).GetListOfFiles();
               if (filelist->GetEntries() > 0) { 
                  for (auto f : *filelist)
                     fFileNames.emplace_back(f->GetTitle());
                  StoreFriends(tree, false);
               }
               else {
                  auto msg = "The provided chain of files is empty, cannot process tree " + fTreeName;
                  throw std::runtime_error(msg);
               }
            }
            else {
               TFile *f = tree.GetCurrentFile();
               if (f) {
                  fFileNames.emplace_back(f->GetName());
                  StoreFriends(tree, true);
               }
               else {
                  auto msg = "The specified TTree is not linked to any file, in-memory-only trees are not supported. Cannot process tree " + fTreeName;
                  throw std::runtime_error(msg);
               } 
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a TTree and a TEntryList.
         /// \param[in] tree Tree or chain of files containing the tree to process.
         /// \param[in] entries List of entry numbers to process.
         TTreeView(TTree& tree, TEntryList& entries) : TTreeView(tree)
         {
            Long64_t numEntries = entries.GetN();
            for (Long64_t i = 0; i < numEntries; ++i) {
               fEntryList.Enter(entries.GetEntry(i));
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Copy constructor.
         /// \param[in] view Object to copy.
         TTreeView(const TTreeView &view) : fTreeName(view.fTreeName), fEntryList(view.fEntryList)
         {
            for (auto& fn : view.fFileNames)
               fFileNames.emplace_back(fn);

            for (auto &fn : view.fFriendNames)
               fFriendNames.emplace_back(fn);

            for (auto &ffn : view.fFriendFileNames) {
               fFriendFileNames.emplace_back();
               auto &fileNames = fFriendFileNames.back();
               for (auto &name : ffn) {
                  fileNames.emplace_back(name);
               }
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the current tree of this view.
         TreeReaderEntryListPair GetTreeReader(Long64_t start, Long64_t end, const std::vector<Long64_t> &nEntries,
                                               const std::vector<std::vector<Long64_t>> &friendEntries)
         {
            if (fChain == nullptr)
               MakeChain(nEntries, friendEntries);

            std::unique_ptr<TTreeReader> reader;
            std::unique_ptr<TEntryList> elist;
            if (fEntryList.GetN() > 0) {
               std::tie(reader, elist) = MakeReaderWithEntryList(start, end);
            } else {
               reader = MakeReader(start, end);
            }

            // we need to return the entry list too, as it needs to be in scope as long as the reader is
            return std::make_pair(std::move(reader), std::move(elist));
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the filenames for this view.
         const std::vector<std::string> &GetFileNames() const
         {
            return fFileNames;
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the name of the tree of this view.
         std::string GetTreeName() const
         {
            return fTreeName;
         }

         //////////////////////////////////////////////////////////////////////////
         /// Push a new loaded entry to the stack.
         void PushTaskFirstEntry(Long64_t entry) { fLoadedEntries.push_back(entry); }

         //////////////////////////////////////////////////////////////////////////
         /// Restore the tree of the previous loaded entry, if any.
         void PopTaskFirstEntry()
         {
            fLoadedEntries.pop_back();
            if (fLoadedEntries.size() > 0) {
               fChain->LoadTree(fLoadedEntries.back());
            }
         }

         const std::vector<NameAlias> &GetFriendNames() const
         {
            return fFriendNames;
         }

         const std::vector<std::vector<std::string>> &GetFriendFileNames() const
         {
            return fFriendFileNames;
         }
      };
   } // End of namespace Internal


   class TTreeProcessorMT {
   private:
      ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView; ///<! Thread-local TreeViews

   public:
      TTreeProcessorMT(std::string_view filename, std::string_view treename = "");
      TTreeProcessorMT(const std::vector<std::string_view>& filenames, std::string_view treename = "");
      TTreeProcessorMT(TTree& tree);
      TTreeProcessorMT(TTree& tree, TEntryList& entries);
 
      void Process(std::function<void(TTreeReader&)> func);

   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
