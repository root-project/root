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
      class TTreeView {
      private:
         std::vector<std::string> fFileNames; ///< Names of the files
         std::string fTreeName;               ///< Name of the tree
         std::unique_ptr<TFile> fCurrentFile; ///<! Current file object of this view.
         TTree *fCurrentTree;                 ///<! Current tree object of this view.
         unsigned int fCurrentIdx;            ///<! Index of the current file.
         std::vector<TEntryList> fEntryLists; ///< Entry numbers to be processed per tree/file
         TEntryList fCurrentEntryList;        ///< Entry numbers for the current range being processed

         ////////////////////////////////////////////////////////////////////////////////
         /// Initialize the file and the tree for this view, first looking for a tree in
         /// the file if necessary.
         void Init()
         {
            fCurrentFile.reset(TFile::Open(fFileNames[fCurrentIdx].data()));

            // If the tree name is empty, look for a tree in the file
            if (fTreeName.empty()) {
               TIter next(fCurrentFile->GetListOfKeys());
               while (TKey *key = (TKey*)next()) {
                  const char *className = key->GetClassName();
                  if (strcmp(className, "TTree") == 0) {
                     fTreeName = key->GetName();
                     break;
                  }
               }
               if (fTreeName.empty())
                  ::Error("TreeView constructor", "Cannot find any tree in file %s", fFileNames[fCurrentIdx].data());
            }

            // We cannot use here the template method (TFile::GetObject) because the header will finish
            // in the PCH and the specialization will be available. PyROOT will not be able to specialize
            // the method for types other that TTree.
            fCurrentTree = (TTree*)fCurrentFile->Get(fTreeName.data());

            // Do not remove this tree from list of cleanups (thread unsafe)
            fCurrentTree->ResetBit(TObject::kMustCleanup);
         }

      public:
         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a file name.
         /// \param[in] fn Name of the file containing the tree to process.
         /// \param[in] tn Name of the tree to process. If not provided,
         ///               the implementation will automatically search for a
         ///               tree in the file.
         TTreeView(std::string_view fn, std::string_view tn) : fTreeName(tn), fCurrentIdx(0)
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
         TTreeView(const std::vector<std::string_view>& fns, std::string_view tn) : fTreeName(tn), fCurrentIdx(0)
         {
            if (fns.size() > 0) {
               for (auto& fn : fns)
                  fFileNames.emplace_back(fn);
               Init();
             }
             else {
                ::Error("TreeView constructor", "The provided list of file names is empty, cannot process tree %s", fTreeName.data());
             }
         }

         //////////////////////////////////////////////////////////////////////////
         /// Constructor based on a TTree.
         /// \param[in] tree Tree or chain of files containing the tree to process.
         TTreeView(TTree& tree) : fTreeName(tree.GetName()), fCurrentIdx(0)
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
                  ::Error("TreeView constructor", "The provided chain of files is empty, cannot process tree %s", fTreeName.data());
               }
            }
            else {
               fFileNames.emplace_back(tree.GetCurrentFile()->GetName());
               Init();
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
               size_t nTrees = fFileNames.size();
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
         TTreeView(const TTreeView& view) : fTreeName(view.fTreeName), fCurrentIdx(view.fCurrentIdx) 
         {
            for (auto& fn : view.fFileNames)
               fFileNames.emplace_back(fn);

            for (auto& el : view.fEntryLists)
               fEntryLists.emplace_back(el);

            Init();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the cluster iterator for the current tree of this view, starting
         /// from entry zero.
         TTree::TClusterIterator GetClusterIterator()
         {
            return fCurrentTree->GetClusterIterator(0);
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the current tree of this view.
         std::unique_ptr<TTreeReader> GetTreeReader(Long64_t start, Long64_t end)
         {
            TTreeReader *reader;
            if (fEntryLists.size() > 0 && fEntryLists[fCurrentIdx].GetN() > 0) {
               // TEntryList and SetEntriesRange do not work together (the former has precedence).
               // We need to construct a TEntryList that contains only those entry numbers
               // in our desired range.
               fCurrentEntryList.Reset();
               Long64_t entry = fEntryLists[fCurrentIdx].GetEntry(0);
               do {
                  if (entry >= start && entry < end) fCurrentEntryList.Enter(entry);
               } while ((entry = fEntryLists[fCurrentIdx].Next()) >= 0);

               reader = new TTreeReader(fCurrentTree, &fCurrentEntryList);

            }
            else {
               // If no TEntryList is involved we can safely set the range in the reader
               reader = new TTreeReader(fCurrentTree);
               reader->SetEntriesRange(start, end);
            }

            return std::unique_ptr<TTreeReader>(reader);
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the number of entries of the current tree of this view.
         Long64_t GetEntries() const
         {
            return fCurrentTree->GetEntries();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the number of files of this view.
         size_t GetNumFiles() const
         {
            return fFileNames.size();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Set the current file and tree of this view.
         void SetCurrent(unsigned int i)
         {
            if (i != fCurrentIdx) {
               fCurrentIdx = i;
               TFile *f = TFile::Open(fFileNames[fCurrentIdx].data());
               fCurrentTree = (TTree*)f->Get(fTreeName.data());
               fCurrentTree->ResetBit(TObject::kMustCleanup);
               fCurrentFile.reset(f);
            }
         }
      };
   } // End of namespace Internal


   class TTreeProcessorMT {
   private:
      ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView; ///<! Threaded object with <file,tree> per thread

   public:
      TTreeProcessorMT(std::string_view filename, std::string_view treename = "");
      TTreeProcessorMT(const std::vector<std::string_view>& filenames, std::string_view treename = "");
      TTreeProcessorMT(TTree& tree);
      TTreeProcessorMT(TTree& tree, TEntryList& entries);
 
      void Process(std::function<void(TTreeReader&)> func);

   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
