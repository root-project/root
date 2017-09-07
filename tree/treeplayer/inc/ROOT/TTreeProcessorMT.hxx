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
         std::size_t filenameIdx;
      };

      /// Input data as seen by TTreeView.
      ///
      /// Each thread will contain a TTreeView that will perform bookkeping of a vector of (few) TreeViewInputs.
      /// This vector will contain a TreeViewInput for each task currently working on a different input file.
      struct TreeViewInput {
         std::unique_ptr<TFile> file;
         TTree *tree; // needs to be a raw pointer because file destructs this tree when deleted
         std::size_t filenameIdx; ///< The filename index of this file in the list of filenames contained in TTreeView
         unsigned int useCount;   ///< Number of tasks that are currently using this input
      };

      class TTreeView {
      private:
         std::vector<std::string> fFileNames;    ///< Names of the files
         std::vector<TEntryList> fEntryLists;    ///< Entry numbers to be processed per tree/file. 1:1 with fFileNames
         std::string fTreeName;                  ///< Name of the tree
         std::vector<TreeViewInput> fOpenInputs; ///< Input files currently open.

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
            static const TClassRef clRefTChain("TChain");
            if (clRefTChain == tree.IsA()) {
               // We need to convert the global entry numbers to per-tree entry numbers.
               // This will allow us to build a TEntryList for a given entry range of a tree of the chain.
               std::size_t nTrees = fFileNames.size();
               fEntryLists.resize(nTrees);

               TChain *chain = dynamic_cast<TChain*>(&tree);
               Long64_t currListEntry  = entries.GetEntry(0);
               Long64_t currTreeOffset = 0;

               for (unsigned int treeNum = 0; treeNum < nTrees && currListEntry >= 0; ++treeNum) {
                  chain->LoadTree(currTreeOffset);
                  TTree *currTree = chain->GetTree();
                  Long64_t currTreeEntries = currTree->GetEntries();
                  Long64_t nextTreeOffset = currTreeOffset + currTreeEntries;

                  while (currListEntry >= 0 && currListEntry < nextTreeOffset) {
                     fEntryLists[treeNum].Enter(currListEntry - currTreeOffset);
                     currListEntry = entries.Next();
                  }

                  currTreeOffset = nextTreeOffset;
               }
            }
            else {
               fEntryLists.emplace_back(entries);
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Copy constructor.
         /// \param[in] view Object to copy.
         TTreeView(const TTreeView& view) : fTreeName(view.fTreeName)
         {
            for (auto& fn : view.fFileNames)
               fFileNames.emplace_back(fn);

            for (auto& el : view.fEntryLists)
               fEntryLists.emplace_back(el);
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the current tree of this view.
         using TreeReaderEntryListPair = std::pair<std::unique_ptr<TTreeReader>, std::unique_ptr<TEntryList>>;
         TreeReaderEntryListPair GetTreeReader(std::size_t dataIdx, Long64_t start, Long64_t end)
         {
            std::unique_ptr<TTreeReader> reader;
            std::unique_ptr<TEntryList> elist;
            if (fEntryLists.size() > 0) {
               // TEntryList and SetEntriesRange do not work together (the former has precedence).
               // We need to construct a TEntryList that contains only those entry numbers
               // in our desired range.
               const auto filenameIdx = fOpenInputs[dataIdx].filenameIdx;
               elist.reset(new TEntryList);
               if (fEntryLists[filenameIdx].GetN() > 0) {
                  Long64_t entry = fEntryLists[filenameIdx].GetEntry(0);
                  do {
                     if (entry >= start && entry < end) // TODO can quit this loop early when entry >= end
                        elist->Enter(entry);
                  } while ((entry = fEntryLists[filenameIdx].Next()) >= 0);
               }
               reader.reset(new TTreeReader(fOpenInputs[dataIdx].tree, elist.get()));
            } else {
               // If no TEntryList is involved we can safely set the range in the reader
               reader.reset(new TTreeReader(fOpenInputs[dataIdx].tree));
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
         /// Search the open files for the filename with index i. If found, increment its "user counter", otherwise
         /// open it and add it to the vector of open files. Return the file's index in the vector of open files.
         std::size_t FindOrOpenFile(std::size_t filenameIdx)
         {
            const auto inputIt =
               std::find_if(fOpenInputs.begin(), fOpenInputs.end(),
                            [filenameIdx](const TreeViewInput &i) { return i.filenameIdx == filenameIdx; });
            if (inputIt != fOpenInputs.end()) {
               // requested file is already open
               inputIt->useCount++;
               return std::distance(fOpenInputs.begin(), inputIt); // return input's index in fOpenInputs
            } else {
               // requested file needs to be added to fOpenInputs
               TDirectory::TContext ctxt(gDirectory); // needed to restore the directory after opening the file
               std::unique_ptr<TFile> f(TFile::Open(fFileNames[filenameIdx].data()));
               // We must use TFile::Get instead of TFile::GetObject because this header will finish                 
               // in the PCH and the TFile::GetObject<TTree> specialization will be available there. PyROOT will then
               // not be able to specialize the method for types other that TTree.
               // A test that fails as a consequence of this issue is python-ttree-tree.
               TTree *t = static_cast<TTree*>(f->GetObjectChecked(fTreeName.c_str(), "TTree"));
               t->ResetBit(TObject::kMustCleanup);
               fOpenInputs.emplace_back(TreeViewInput{std::move(f), t, filenameIdx, /*useCount=*/1});
               return fOpenInputs.size() - 1;
            }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Decrease "use count" of the file at filenameIdx, delete corresponding TreeViewInput if use count is zero.
         void Cleanup(std::size_t dataIdx)
         {
            fOpenInputs[dataIdx].useCount--;
            if(fOpenInputs[dataIdx].useCount == 0)
               fOpenInputs.erase(fOpenInputs.begin() + dataIdx);
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
