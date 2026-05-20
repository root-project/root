// @(#)root/tree:$Id$
// Author: Anna Kreshuk 17/03/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEntryListFromFile
#define ROOT_TEntryListFromFile

//////////////////////////////////////////////////////////////////////////
// TEntryListFromFile
//
// Manages entry lists from different files, when they are not loaded
// in memory at the same time.
//
// This entry list should only be used when processing a TChain (see
// TChain::SetEntryList() function). File naming convention:
// - by default, filename_elist.root is used, where filename is the
//   name of the chain element.
// - xxx$xxx.root - $ sign is replaced by the name of the chain element
// If the list name is not specified (by passing filename_elist.root/listname to
// the TChain::SetEntryList() function, the first object of class TEntryList
// in the file is taken.
// It is assumed that there are as many lists, as there are chain elements,
// and they are in the same order.
//
// If one of the list files can't be opened, or there is an error reading a list
// from the file, this list is skipped and the entry loop continues on the next
// list.

#include "TEntryList.h"

#include <limits>

class TFile;

class TEntryListFromFile: public TEntryList
{
protected:
   TString    fListFileName;  ///<  from this string names of all files can be found
   TString    fListName;      ///<  name of the list
   Int_t      fNFiles;        ///<  total number of files
   Long64_t   *fListOffset;   ///<[fNFiles] numbers of entries in ind. lists
   TFile      *fFile;         ///< currently open file
                              ///<  fCurrent points to the currently open list
   TObjArray *fFileNames;     ///<! points to the fFiles data member of the corresponding chain

   // Obsolete use TTree::kMaxEntries
   static constexpr auto kBigNumber = std::numeric_limits<Long64_t>::max();

private:
   TEntryListFromFile(const TEntryListFromFile&);            // Not implemented.
   TEntryListFromFile &operator=(const TEntryListFromFile&); // Not implemented.

public:

   TEntryListFromFile();
   TEntryListFromFile(const char *filename, const char *listname, Int_t nfiles);
   ~TEntryListFromFile() override;
   void        Add(const TEntryList * /* elist */) override {};
   Int_t       Contains(Long64_t /* entry */, TTree * /* tree = 0 */) override { return 0; };
   bool        Enter(Long64_t /* entry */, TTree * /* tree = 0 */) override { return false; };
   bool        Enter(Long64_t /* entry */, const char * /* treename */, const char * /* filename */) override { return false; };
   TEntryList *GetCurrentList() const override { return fCurrent; };
   TEntryList *GetEntryList(const char * /* treename */, const char * /* filename */, Option_t * /* opt="" */) override { return nullptr; };

   Long64_t    GetEntry(Long64_t index) override;
   Long64_t    GetEntryAndTree(Long64_t index, Int_t &treenum) override;
   virtual Long64_t    GetEntries();
   virtual Long64_t    GetEntriesFast() const { return fN; }

   Long64_t    GetN() const override { return fN; }
   const char *GetTreeName() const override { return fTreeName.Data(); }
   const char *GetFileName() const override { return fFileName.Data(); }
   Int_t       GetTreeNumber() const override { return fTreeNumber; }

   virtual Int_t       LoadList(Int_t listnumber);

   Int_t       Merge(TCollection * /*list*/) override{ return 0; }

   Long64_t    Next() override;
   void        OptimizeStorage() override {};
   bool        Remove(Long64_t /*entry*/, TTree * /*tree = nullptr */) override{ return false; }

   void        Print(const Option_t* option = "") const override;

   void        SetTree(const TTree * /*tree*/) override {}
   void        SetTree(const char * /*treename*/, const char * /*filename*/) override {}
   virtual void        SetFileNames(TObjArray *names) { fFileNames = names; }
   void        SetTreeNumber(Int_t index) override { fTreeNumber=index;  }
   virtual void        SetNFiles(Int_t nfiles) { fNFiles = nfiles; }
   void        Subtract(const TEntryList * /*elist*/) override {}

   ClassDefOverride(TEntryListFromFile, 1); //Manager for entry lists from different files
};
#endif
