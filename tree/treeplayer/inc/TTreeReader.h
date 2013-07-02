// @(#)root/tree:$Id$
// Author: Axel Naumann, 2010-08-02

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeReader
#define ROOT_TTreeReader


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTreeReader                                                            //
//                                                                        //
// A simple interface for reading trees or chains.                        //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBranchProxyDirector
#include "TBranchProxyDirector.h"
#endif
#ifndef ROOT_TBranchProxy
#include "TBranchProxy.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_THashTable
#include "THashTable.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TTreeReaderUtils
#include "TTreeReaderUtils.h"
#endif

class TDictionary;
class TDirectory;
class TFileCollection;

class TTreeReader: public TObject {
public:

   enum EEntryStatus {
      kEntryValid = 0, // data read okay
      kEntryNotLoaded, // no entry has been loaded yet
      kEntryNoTree, // the tree does not exist
      kEntryNotFound, // the tree entry number does not exist
      kEntryChainSetupError, // problem in accessing a chain element, e.g. file without the tree
      kEntryChainFileError, // problem in opening a chain's file
      kEntryDictionaryError, // problem reading dictionary info from tree
   };

   TTreeReader():
      fDirectory(0),
      fEntryStatus(kEntryNoTree)
   {}

   TTreeReader(TTree* tree);
   TTreeReader(const char* keyname, TDirectory* dir = NULL );
   TTreeReader(const char* /*keyname*/, TFileCollection* /*files*/) { Error("TTreeReader()", "Not Implemented!");};

   ~TTreeReader();

   void SetTree(TTree* tree);
   void SetTree(const char* /*keyname*/, TDirectory* /*dir = NULL*/ ) { Error("SetTree()", "Not Implemented!");};
   void SetChain(const char* /*keyname*/, TFileCollection* /*files*/ ) { Error("SetChain()", "Not Implemented!");};

   Bool_t IsChain() const { return TestBit(kBitIsChain); }

   Bool_t SetNextEntry() { return SetEntry(GetCurrentEntry() + 1) == kEntryValid; }
   EEntryStatus SetEntry(Long64_t entry);

   EEntryStatus GetEntryStatus() const { return fEntryStatus; }

   TTree* GetTree() const { return fTree; }
   Long64_t GetEntries(Bool_t force) const { return fTree ? (force ? fTree->GetEntries() : fTree->GetEntriesFast() ) : -1; }
   Long64_t GetCurrentEntry() const { return fDirector ? fDirector->GetReadEntry() : 0; }

protected:
   void Initialize();
   ROOT::TNamedBranchProxy* FindProxy(const char* branchname) const {
      return (ROOT::TNamedBranchProxy*) fProxies.FindObject(branchname); }
   TCollection* GetProxies() { return &fProxies; }

   void RegisterValueReader(ROOT::TTreeReaderValueBase* reader);
   void DeregisterValueReader(ROOT::TTreeReaderValueBase* reader);

private:

   enum EPropertyBits {
      kBitIsChain = BIT(14) // our tree is a chain
   };

   TTree* fTree; // tree that's read
   TDirectory* fDirectory; // directory (or current file for chains)
   EEntryStatus fEntryStatus; // status of most recent read request
   ROOT::TBranchProxyDirector* fDirector; // proxying director, owned
   TObjArray    fValues; // TTreeReaderValueBase objects that use our director
   THashTable   fProxies; //attached ROOT::TNamedBranchProxies; owned

   friend class ROOT::TTreeReaderValueBase;
   friend class ROOT::TTreeReaderArrayBase;

   ClassDef(TTreeReader, 0); // A simple interface to read trees
};

#endif // defined TTreeReader
