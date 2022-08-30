// @(#)root/tree:$Id$
// Author: Bruno Lenzi 12/07/2011

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEntryListArray
#define ROOT_TEntryListArray

#include "TEntryList.h"

class TTree;
class TDirectory;
class TObjArray;
class TString;

class TList;
class TCollection;
class TIter;

class TEntryListArray : public TEntryList {

private:
   TEntryListArray& operator=(const TEntryListArray&); // Not implemented

protected:
   TList *fSubLists;                     ///<  a list of underlying entry lists for each event of a TEntryList
   Long64_t         fEntry;              ///<  the entry number, when the list is used for subentries
   TEntryListArray *fLastSubListQueried; ///<! last sublist checked by GetSubListForEntry
   TIter *fSubListIter;                  ///<! to iterate over fSubLists and keep last one checked

   void Init();
   virtual void AddEntriesAndSubLists(const TEntryList *elist);
   virtual void ConvertToTEntryListArray(TEntryList *e);
//    virtual TList* GetSubLists() const {
//       return fSubLists;
//    };
   virtual Bool_t      RemoveSubList(TEntryListArray *e, TTree *tree = nullptr);
   virtual Bool_t      RemoveSubListForEntry(Long64_t entry, TTree *tree = nullptr);
   virtual TEntryListArray* SetEntry(Long64_t entry, TTree *tree = nullptr);


public:
   TEntryListArray();
   TEntryListArray(const char *name, const char *title);
   TEntryListArray(const char *name, const char *title, const TTree *tree);
   TEntryListArray(const char *name, const char *title, const char *treename, const char *filename);
   TEntryListArray(const TTree *tree);
   TEntryListArray(const TEntryListArray& elist);
   TEntryListArray(const TEntryList& elist); // to convert TEL to TELA
   ~TEntryListArray() override;

   void                Add(const TEntryList *elist) override;
   virtual Int_t       Contains(Long64_t entry, TTree *tree, Long64_t subentry);
   Int_t       Contains(Long64_t entry, TTree *tree = nullptr) override {
      return TEntryList::Contains(entry, tree);
   };
   virtual Bool_t      Enter(Long64_t entry, TTree *tree, Long64_t subentry);
   virtual Bool_t      Enter(Long64_t entry, const char *treename, const char *filename, Long64_t subentry);
   Bool_t      Enter(Long64_t entry, TTree *tree = nullptr) override {
      return Enter(entry, tree, -1);
   };
   Bool_t              Enter(Long64_t entry, const char *treename, const char *filename) override
   {
      return Enter(entry, treename, filename, -1);
   };
//    virtual Bool_t      Enter(Long64_t entry, TTree *tree, const TEntryList *e);
   virtual TEntryListArray* GetSubListForEntry(Long64_t entry, TTree *tree = nullptr);
   void        Print(const Option_t* option = "") const override;
   virtual Bool_t      Remove(Long64_t entry, TTree *tree, Long64_t subentry);
   Bool_t      Remove(Long64_t entry, TTree *tree = nullptr) override {
      return Remove(entry, tree, -1);
   };
   void        Reset() override;

   void        SetTree(const char *treename, const char *filename) override;
   void        SetTree(const TTree *tree) override {
      TEntryList::SetTree(tree);   // will take treename and filename from the tree and call the method above
   }
   void        Subtract(const TEntryList *elist) override;
   virtual TList* GetSubLists() const {
      return fSubLists;
   };

   ClassDefOverride(TEntryListArray, 1);  //A list of entries and subentries in a TTree
};
#endif
