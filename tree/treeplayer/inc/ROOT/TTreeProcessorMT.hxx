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
         std::unique_ptr<TTree> fCurrentTree; ///<! Current tree object of this view.
         int fCurrentIdx;                     ///<! Index of the current file.

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
            TTree *tp = (TTree*)fCurrentFile->Get(fTreeName.data());
            fCurrentTree.reset(tp);
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
         /// Copy constructor.
         /// \param[in] view Object to copy.
         TTreeView(const TTreeView& view) : fTreeName(view.fTreeName), fCurrentIdx(view.fCurrentIdx) 
         {
            for (auto& fn : view.fFileNames)
               fFileNames.emplace_back(fn);
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
         std::unique_ptr<TTreeReader> GetTreeReader() const
         {
            return std::unique_ptr<TTreeReader>(new TTreeReader(fCurrentTree.get()));
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
         void SetCurrent(int i)
         {
            if (i != fCurrentIdx) {
               fCurrentIdx = i;
               TFile *f = TFile::Open(fFileNames[fCurrentIdx].data());
               TTree *t = (TTree*)f->Get(fTreeName.data());
               fCurrentTree.reset(t);
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
 
      void Process(std::function<void(TTreeReader&)> func);

   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
