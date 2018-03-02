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

      /// A cluster of entries as seen by TTreeView
      struct TreeViewCluster {
         Long64_t startEntry;
         Long64_t endEntry;
      };

      class TTreeView {
      private:
         typedef std::pair<std::string, std::string> NameAlias;

         std::unique_ptr<TChain> fChain;         ///< Chain on which to operate
         std::vector<std::string> fFileNames;    ///< Names of the files
         std::string fTreeName;                  ///< Name of the tree
         TEntryList fEntryList;                  ///< Entry numbers to be processed
         std::vector<Long64_t> fLoadedEntries;   ///<! Per-task loaded entries (for task interleaving)
         std::vector<NameAlias> fFriendNames;    ///< <name,alias> pairs of the friends of the tree/chain
         std::vector<std::vector<std::string>> fFriendFileNames; ///< Names of the files where friends are stored
         std::vector<std::unique_ptr<TChain>> fFriends;          ///< Friends of the tree/chain

         ////////////////////////////////////////////////////////////////////////////////
         /// Initialize TTreeView.
         void Init()
         {
            // If the tree name is empty, look for a tree in the file
            if (fTreeName.empty()) {
               ::TDirectory::TContext ctxt(gDirectory);
               std::unique_ptr<TFile> f(TFile::Open(fFileNames[0].c_str()));
               TIter next(f->GetListOfKeys());
               while (TKey *key = (TKey*)next()) {
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

            fChain.reset(new TChain(fTreeName.c_str()));
            for (auto &fn : fFileNames) {
               fChain->Add(fn.c_str());
            }
            fChain->ResetBit(TObject::kMustCleanup);

            auto friendNum = 0u;
            for (auto &na : fFriendNames) {
               auto &name = na.first;
               auto &alias = na.second;

               // Build a friend chain
               TChain *frChain = new TChain(name.c_str());
               auto &fileNames = fFriendFileNames[friendNum];
               for (auto &fn : fileNames)
                  frChain->Add(fn.c_str());

               // Make it friends with the main chain
               fFriends.emplace_back(frChain);
               fChain->AddFriend(frChain, alias.c_str());

               ++friendNum;
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
            Init();
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
               Init();
            }
            else {
               auto msg = "The provided list of file names is empty, cannot process tree " + fTreeName;
               throw std::runtime_error(msg);
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a TTree.
         /// \param[in] tree Tree or chain of files containing the tree to process.
         TTreeView(TTree& tree) : fTreeName(tree.GetName())
         {
            static const TClassRef clRefTChain("TChain");
            if (clRefTChain == tree.IsA()) {
               TObjArray* filelist = dynamic_cast<TChain&>(tree).GetListOfFiles();
               if (filelist->GetEntries() > 0) { 
                  for (auto f : *filelist)
                     fFileNames.emplace_back(f->GetTitle());
                  StoreFriends(tree, false);
                  Init();
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
                  Init();
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

            Init();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the current tree of this view.
         using TreeReaderEntryListPair = std::pair<std::unique_ptr<TTreeReader>, std::unique_ptr<TEntryList>>;
         TreeReaderEntryListPair GetTreeReader(Long64_t start, Long64_t end)
         {
            std::unique_ptr<TTreeReader> reader;
            std::unique_ptr<TEntryList> elist;
            if (fEntryList.GetN() > 0) {
               // TEntryList and SetEntriesRange do not work together (the former has precedence).
               // We need to construct a TEntryList that contains only those entry numbers
               // in our desired range.
               elist.reset(new TEntryList);
               Long64_t entry = fEntryList.GetEntry(0);
               do {
                  if (entry >= start && entry < end) // TODO can quit this loop early when entry >= end
                     elist->Enter(entry);
               } while ((entry = fEntryList.Next()) >= 0);

               reader.reset(new TTreeReader(fChain.get(), elist.get()));
            } else {
               // If no TEntryList is involved we can safely set the range in the reader
               reader.reset(new TTreeReader(fChain.get()));
               fChain->LoadTree(start - 1);
               reader->SetEntriesRange(start, end);
            }

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
         void PushLoadedEntry(Long64_t entry) { fLoadedEntries.push_back(entry); }

         //////////////////////////////////////////////////////////////////////////
         /// Restore the tree of the previous loaded entry, if any.
         void RestoreLoadedEntry()
         {
            fLoadedEntries.pop_back();
            if (fLoadedEntries.size() > 0) {
               fChain->LoadTree(fLoadedEntries.back());
            }
         }
      };
   } // End of namespace Internal


   class TTreeProcessorMT {
   private:
      ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView; ///<! Thread-local TreeViews

      std::vector<ROOT::Internal::TreeViewCluster> MakeClusters();
   public:
      TTreeProcessorMT(std::string_view filename, std::string_view treename = "");
      TTreeProcessorMT(const std::vector<std::string_view>& filenames, std::string_view treename = "");
      TTreeProcessorMT(TTree& tree);
      TTreeProcessorMT(TTree& tree, TEntryList& entries);
 
      void Process(std::function<void(TTreeReader&)> func);

   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
