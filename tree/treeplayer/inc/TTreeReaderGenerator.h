// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeReaderGenerator
#define ROOT_TTreeReaderGenerator

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeReaderGenerator                                                 //
//                                                                      //
// Generate a Selector using the TTreeReader interface                  //
// (TTreeReaderValue, TTreeReaderArray) to access the data in the tree. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Tlist
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TBranch;
class TBranchElement;
class TLeaf;
class TTree;

namespace ROOT {
   // 0 for the general case, 1 when this a split clases inside a TClonesArray,
   // 2 when this is a split classes inside an STL container.
   enum ELocation { kOut=0, kClones, kSTL };

   class TTreeReaderDescriptor : public TObject {
   public:
      enum ReaderType { kValue, kArray };
      ReaderType fType;    // Type of the reader: Value or Array
      TString fDataType;   // Data type of reader
      TString fName;       // Reader name
      TString fBranchName; // Branch corresponding to the reader

      TTreeReaderDescriptor(ReaderType type, TString dataType, TString name, TString branchName) :
         fType(type),
         fDataType(dataType),
         fName(name),
         fBranchName(branchName) { }
   };

   class TBranchDescriptor : public TNamed {
   public:
      ELocation             fIsClones;       // Type of container
      TString               fContainerName;  // Name of the container
      TString               fBranchName;     // Short name of the branch
      TString               fFullBranchName; // Full name of the branch
      TVirtualStreamerInfo *fInfo;           // Streamer info
      TBranchDescriptor    *fParent;         // Descriptor of the parent branch (NULL for topmost)

      TBranchDescriptor(const char *type, TVirtualStreamerInfo *info,
                        const char *branchname, ELocation isclones,
                        const TString &containerName, const char *prefix = 0, TBranchDescriptor *parent = 0) :
         TNamed(type,type),
         fIsClones(isclones),
         fContainerName(containerName),
         fBranchName(branchname),
         fFullBranchName(branchname),
         fInfo(info),
         fParent(parent)
         {
            // If there is a prefix, append to the beginning
            if (prefix) {
               fFullBranchName.Form("%s_%s", prefix, branchname);
            }
         }

      Bool_t IsClones() const { return fIsClones == kClones; }

      Bool_t IsSTL() const { return fIsClones == kSTL; }
   };

   class TTreeReaderGenerator
   {
      TTree                *fTree;              // Pointer to the tree
      TString               fClassname;         // Class name of the selector
      TList                 fListOfHeaders;     // List of included headers
      TList                 fListOfReaders;     // List of readers
      TString               fOptions;           // User options as a string
      Bool_t                fIncludeAllLeaves;  // Should all leaves be included
      Bool_t                fIncludeAllTopmost; // Should all topmost branches be included
      std::vector<TString>  fIncludeLeaves;     // Branches whose leaves should be included
      std::vector<TString>  fIncludeStruct;     // Branches whom should be included

      void   AddHeader(TClass *cl);
      void   AddReader(TTreeReaderDescriptor::ReaderType type, TString dataType, TString name,
                       TString branchName, TBranchDescriptor *parent = 0, Bool_t isLeaf = kTRUE);
      UInt_t AnalyzeBranches(TBranchDescriptor *desc, TBranchElement *branch, TVirtualStreamerInfo *info);
      UInt_t AnalyzeBranches(TBranchDescriptor *desc, TIter &branches, TVirtualStreamerInfo *info);
      UInt_t AnalyzeOldBranch(TBranch *branch);
      UInt_t AnalyzeOldLeaf(TLeaf *leaf, Int_t nleaves);
      Bool_t BranchNeedsReader(TString branchName, TBranchDescriptor *parent, Bool_t isLeaf);

      void   ParseOptions();
      void   AnalyzeTree(TTree *tree);
      void   WriteSelector();

   public:
      TTreeReaderGenerator(TTree* tree, const char *classname, Option_t *option);
   };
}

using ROOT::TTreeReaderGenerator;

#endif
