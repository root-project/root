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

#ifndef ROOT_TKey
#include "TKey.h"
#endif

#ifndef ROOT_TTree
#include "TTree.h"
#endif

#ifndef ROOT_TFile
#include "TFile.h"
#endif

#ifndef ROOT_TTreeReader
#include "TTreeReader.h"
#endif

#ifndef ROOT_TThreadedObject
#include "ROOT/TThreadedObject.hxx"
#endif

#ifndef ROOT_TError
#include "TError.h"
#endif

#include <string.h>
#include <functional>


/** \class TTreeView
    \brief A helper class that encapsulates a file and a tree.

A helper class that encapsulates a TFile and a TTree, along with their names.
It is used together with TTProcessor and ROOT::TThreadedObject, so that
in the TTProcessor::Process method each thread can work on its own
<TFile,TTree> pair.

A copy constructor is defined for TTreeView to work with ROOT::TThreadedObject.
The latter makes a copy of a model object every time a new thread accesses
the threaded object.
*/

namespace ROOT {
   namespace Internal {
      class TTreeView {
      private:
         std::string_view fFileName;   ///< Name of the file
         std::string_view fTreeName;   ///< Name of the tree
         std::unique_ptr<TFile> fFile; ///<! File object of this view.
         std::unique_ptr<TTree> fTree; ///<! Tree object of this view.

         ////////////////////////////////////////////////////////////////////////////////
         /// Initialize the file and the tree for this view, first looking for a tree in
         /// the file if necessary.
         void Init()
         {
            fFile.reset(TFile::Open(fFileName.data()));

            // If the tree name is empty, look for a tree in the file
            if (fTreeName.empty()) {
               TIter next(fFile->GetListOfKeys());
               while (TKey *key = (TKey*)next()) {
                  const char *className = key->GetClassName();
                  if (strcmp(className, "TTree") == 0) {
                     fTreeName = key->GetName();
                     break;
                  }
               }
               if (fTreeName.empty())
                  ::Error("TreeView constructor", "Cannot find any tree in file %s", fFileName.data());
            }

            // We cannot use here the template method (TFile::GetObject) because the header will finish
            // in the PCH and the specialization will be available. PyROOT will not be able to specialize
            // the method for types other that TTree.
            TTree *tp = (TTree*)fFile->Get(fTreeName.data());
            fTree.reset(tp);
         }

      public:
         //////////////////////////////////////////////////////////////////////////
         /// Regular constructor.
         /// \param[in] fn Name of the file containing the tree to process.
         /// \param[in] tn Name of the tree to process. If not provided,
         ///               the implementation will automatically search for a
         ///                tree in the file.
         TTreeView(std::string_view fn, std::string_view tn) : fFileName(fn), fTreeName(tn)
         {
            Init();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Copy constructor.
         /// \param[in] view Object to copy.
         TTreeView(const TTreeView& view) : fFileName(view.fFileName), fTreeName(view.fTreeName)
         {
            Init();
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the cluster iterator for the tree of this view, starting from
         /// entry zero.
         TTree::TClusterIterator GetClusterIterator()
         {
            return fTree->GetClusterIterator(0);
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get a TTreeReader for the tree of this view.
         std::unique_ptr<TTreeReader> GetTreeReader() const
         {
            return std::unique_ptr<TTreeReader>(new TTreeReader(fTree.get()));
         }

         //////////////////////////////////////////////////////////////////////////
         /// Get the number of entries of the tree of this view.
         Long64_t GetEntries() const
         {
            return fTree->GetEntries();
         }
      };
   } // End of namespace Internal


   class TTreeProcessorMT {
   private:
      ROOT::TThreadedObject<ROOT::Internal::TTreeView> treeView; ///<! Threaded object with <file,tree> per thread

   public:
      TTreeProcessorMT(std::string_view filename, std::string_view treename = "");
 
      void Process(std::function<void(TTreeReader&)> func);

   };

} // End of namespace ROOT

#endif // defined TTreeProcessorMT
