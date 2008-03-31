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

class TFile;

class TEntryListFromFile: public TEntryList 
{
 protected:
   TString    fListFileName;  //from this string names of all files can be found
   TString    fListName;      //name of the list
   Int_t      fNFiles;        //total number of files
   Long64_t   *fListOffset;   //[fNFiles] numbers of entries in ind. lists
   TFile      *fFile;         //currently open file
                              //fCurrent points to the currently open list
   TObjArray *fFileNames;     //! points to the fFiles data member of the corresponding chain
   
 public:

   enum {
      kBigNumber = 1234567890
   };

   TEntryListFromFile();
   TEntryListFromFile(const char *filename, const char *listname, Int_t nfiles);
   virtual ~TEntryListFromFile();
   virtual void        Add(const TEntryList * /*elist*/){};
   virtual Int_t       Contains(Long64_t /*entry*/, TTree * /*tree = 0*/)  {return 0;};
   virtual Bool_t      Enter(Long64_t /*entry*/, TTree * /*tree = 0*/){return 0;};
   virtual TEntryList *GetCurrentList() const { return fCurrent; };
   virtual TEntryList *GetEntryList(const char * /*treename*/, const char * /*filename*/, Option_t * /*opt=""*/) {return 0;};
   
   virtual Long64_t    GetEntry(Int_t index);
   virtual Long64_t    GetEntryAndTree(Int_t index, Int_t &treenum);
   virtual Long64_t    GetEntries();
   virtual Long64_t    GetEntriesFast() const { return fN; };
   
   virtual Long64_t    GetN() const { return fN; }
   virtual const char *GetTreeName() const { return fTreeName.Data(); }
   virtual const char *GetFileName() const { return fFileName.Data(); }
   virtual Int_t       GetTreeNumber() const { return fTreeNumber; }
   
   virtual Int_t       LoadList(Int_t listnumber);
   
   virtual Int_t       Merge(TCollection * /*list*/){ return 0; };
   
   virtual Long64_t    Next();
   virtual void        OptimizeStorage() {};
   virtual Bool_t      Remove(Long64_t /*entry*/, TTree * /*tree = 0*/){ return 0; };
   
   virtual void        Print(const Option_t* option = "") const;

   virtual void        SetTree(const TTree * /*tree*/){};
   virtual void        SetTree(const char * /*treename*/, const char * /*filename*/){};
   virtual void        SetFileNames(TObjArray *names) { fFileNames = names; }
   virtual void        SetTreeNumber(Int_t index) { fTreeNumber=index;  }
   virtual void        SetNFiles(Int_t nfiles) { fNFiles = nfiles; }
   virtual void        Subtract(const TEntryList * /*elist*/) {};
   
   ClassDef(TEntryListFromFile, 1); //Manager for entry lists from different files
};
#endif
