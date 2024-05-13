// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeSQL
#define ROOT_TTreeSQL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeSQL                                                             //
//                                                                      //
// A TTree object is a list of TBranch.                                 //
//   To Create a TTree object one must:                                 //
//    - Create the TTree header via the TTree constructor               //
//    - Call the TBranch constructor for every branch.                  //
//                                                                      //
//   To Fill this object, use member function Fill with no parameters.  //
//     The Fill function loops on all defined TBranch.                  //
//                                                                      //
// TTreeSQL is the TTree implementation interfacing with an SQL         //
// database                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TTree.h"

#include <vector>

class TSQLServer;
class TSQLRow;
class TBasketSQL;
class TSQLTableInfo;

class TTreeSQL : public TTree {

protected:
   Int_t                  fCurrentEntry;
   TString                fDB;
   TString                fInsertQuery;
   TString                fQuery;
   TString                fTable;
   TSQLResult            *fResult;
   TSQLRow               *fRow;
   TSQLServer            *fServer;
   bool                   fBranchChecked;
   TSQLTableInfo         *fTableInfo;

   void                   CheckBasket(TBranch *tb);
   bool                   CheckBranch(TBranch *tb);
   bool                   CheckTable(const TString &table) const;
   void                   CreateBranches();
   std::vector<Int_t>    *GetColumnIndice(TBranch *branch);
   void                   Init();
   void                   ResetQuery();
   TString                ConvertTypeName(const TString& typeName );
   virtual void           CreateBranch(const TString& branchName,const TString &typeName);
   bool                   CreateTable(const TString& table);
   TBasket               *CreateBasket(TBranch * br) override;

   TBranch               *BranchImp(const char *branchname, const char *classname, TClass *ptrClass, void *addobj, Int_t bufsize, Int_t splitlevel) override;
   TBranch               *BranchImp(const char *branchname, TClass *ptrClass, void *addobj, Int_t bufsize, Int_t splitlevel) override;

public:
   TTreeSQL(TSQLServer * server, TString DB, const TString& table);
   ~TTreeSQL() override;

   Int_t                  Branch(TCollection *list, Int_t bufsize=32000, Int_t splitlevel=99, const char *name="") override;
   Int_t                  Branch(TList *list, Int_t bufsize=32000, Int_t splitlevel=99) override;
   Int_t                  Branch(const char *folder, Int_t bufsize=32000, Int_t splitlevel=99) override;
   TBranch               *Bronch(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=99) override;
   TBranch               *BranchOld(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=1) override;
   TBranch               *Branch(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=99) override;

   TBranch               *Branch(const char *name, void *address, const char *leaflist, Int_t bufsize) override;

   Int_t          Fill() override;
   Int_t          GetEntry(Long64_t entry=0, Int_t getall=0) override;
   Long64_t       GetEntries() const override;
   Long64_t               GetEntries(const char *sel) override { return TTree::GetEntries(sel); }
   Long64_t               GetEntriesFast()const override;
   TString                GetTableName(){ return fTable; }
   Long64_t       LoadTree(Long64_t entry) override;
   virtual Long64_t       PrepEntry(Long64_t entry);
   void                   Refresh() override;

   ClassDefOverride(TTreeSQL,2);  // TTree Implementation read and write to a SQL database.
};


#endif
