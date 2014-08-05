// @(#)root/tree:$Id$
// Author: Bruno Lenzi 12/07/2011

/*************************************************************************
* Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

//______________________________________________________________________________
/* Begin_Html
<center><h2>TEntryListArray: a list of entries and subentries in a TTree or TChain</h2></center>

TEntryListArray is an extension of TEntryList, used to hold selected entries and subentries (sublists) for when the user has a TTree with containers (vectors, arrays, ...).
End_Html

Begin_Html
<h4> Usage with TTree::Draw to select entries and subentries </h4>
<ol>
<li> <b>To fill a list <i>elist</i> </b>:
    <pre>
     tree->Draw(">> elist", "x > 0", "entrylistarray");
    </pre>
<li> <b>To use a list to select entries and subentries:</b>
  <pre>
     tree->SetEntryList(elist);
     tree->Draw("y");
     tree->Draw("z");
  </pre>
</ol>

Its main purpose is to improve the performance of a code that needs to apply complex cuts on TTree::Draw multiple times. After the first call above to TTree::Draw, a TEntryListArray is created and filled with the entries and the indices of the arrays that satisfied the selection cut (x > 0). In the subsequent calls to TTree::Draw, only these entries / subentries are used to fill histograms.
End_Html

Begin_Html
<h4> About the class </h4>

The class derives from TEntryList and can be used basically in the same way. This same class is used to keep entries and subentries, so there are two types of TEntryListArray's:

<ol>
<li> The ones that only hold subentries
  <ul><li> fEntry is set to the entry# for which the subentries correspond
  <li> fSubLists must be 0</ul>
<li> The ones that hold entries and eventually lists with subentries in fSubLists.
  <ul><li> fEntry = -1 for those
  <li> If there are no sublists for a given entry, all the subentries will be used in the selection </ul>
</ol>

<h4> Additions with respect to TEntryList </h4>
<ol><li> Data members:
 <ul><li> fSubLists: a container to hold the sublists
 <li> fEntry: the entry number if the list is used to hold subentries
 <li> fLastSubListQueried and fSubListIter: a pointer to the last sublist queried and an iterator to resume the loop from the last sublist queried (to speed up selection and insertion in TTree::Draw) </ul>
<li> Public methods:
  <ul><li> Contains, Enter and Remove with subentry as argument
  <li> GetSubListForEntry: to return the sublist corresponding to the given entry </ul>
<li> Protected methods:
  <ul><li> AddEntriesAndSubLists: called by Add when adding two TEntryList arrays with sublists
  <li> ConvertToTEntryListArray: convert TEntryList to TEntryListArray
  <li> RemoveSubList: to remove the given sublist
  <li> RemoveSubListForEntry: to remove the sublist corresponding to the given entry
  <li> SetEntry: to get / set a sublist for the given entry </ul>
</ol>
End_Html */


#include "TEntryListArray.h"
#include "TEntryListBlock.h"
#include "TTree.h"
#include "TFile.h"
#include "TSystem.h"
#include <iostream>

ClassImp(TEntryListArray)

//______________________________________________________________________________
void TEntryListArray::Init()
{
   // Initialize data members, called by Reset
   fSubLists = 0;
   fEntry = -1;
   fLastSubListQueried = 0;
   fSubListIter = 0;
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray() : TEntryList(), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //default c-tor
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const char *name, const char *title): TEntryList(name, title), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //c-tor with name and title
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const char *name, const char *title, const TTree *tree): TEntryList(name, title, tree), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //constructor with name and title, which also sets the tree
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const char *name, const char *title, const char *treename, const char *filename): TEntryList(name, title, treename, filename), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //c-tor with name and title, which also sets the treename and the filename
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const TTree *tree) : TEntryList(tree), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //c-tor, which sets the tree
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const TEntryListArray &elist) : TEntryList(), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //copy c-tor
   fEntry = elist.fEntry;
   Add(&elist);
}

//______________________________________________________________________________
TEntryListArray::TEntryListArray(const TEntryList& elist) : TEntryList(elist), fSubLists(0), fEntry(-1), fLastSubListQueried(0), fSubListIter(0)
{
   //c-tor, from TEntryList
}


//______________________________________________________________________________
TEntryListArray::~TEntryListArray()
{
   // d-tor
   if (fSubLists) {
      fSubLists->Delete();
      delete fSubLists;
   }
   fSubLists = 0;
   delete fSubListIter;
   fSubListIter = 0;
}

//______________________________________________________________________________
void TEntryListArray::Add(const TEntryList *elist)
{
   //Add 2 entry lists

   if (!elist) return;

   if (fEntry != -1) {
      TEntryList::Add(elist);
      return;
   }

   // Include in this list all the trees present in elist, so the sublists can be added
   // This would happen in any case when calling TEntryList::Add
   if (elist->GetLists()) { // the other list has lists to hold mutiple trees, add one by one
      TIter next(elist->GetLists());
      const TEntryList *e = 0;
      while ((e = (const TEntryList*)next())) {
         SetTree(e->GetTreeName(), e->GetFileName());
      }
   } else {
      SetTree(elist->GetTreeName(), elist->GetFileName());
   }

   AddEntriesAndSubLists(elist);
}

//______________________________________________________________________________
void TEntryListArray::AddEntriesAndSubLists(const TEntryList *elist)
{
   // The method that really adds two entry lists with sublists
   // If lists are splitted (fLists != 0), look for the ones whose trees match and call the method for those lists.
   // Add first the sublists, and then use TEntryList::Add to deal with the entries

   // WARNING: cannot call TEntryList::Add in the beginning:
   // - Need to know which entries are present in each list when adding the sublists
   // - TEL::Add is recursive, so it will call this guy after the first iteration

   // Add to the entries and sublists of this list, the ones from the other list
   if (!elist) return;

   if (fLists) { // This list is splitted
      TEntryListArray* e = 0;
      TIter next(fLists);
      fN = 0; // reset fN to set it to the sum of fN in each list
      // Only need to do it here and the next condition will be called only from here
      while ((e = (TEntryListArray*) next())) {
         e->AddEntriesAndSubLists(elist);
         fN += e->GetN();
      }
   } else if (elist->GetLists()) { // The other list is splitted --> will be called only from the previous if
      TIter next(elist->GetLists());
      TEntryList *e = 0;
      while ((e = (TEntryList*) next())) {
         AddEntriesAndSubLists(e);
      }
   } else { // None of the lists are splitted
      if (strcmp(elist->GetTreeName(), fTreeName.Data()) || strcmp(elist->GetFileName(), fFileName.Data()))
         return; // Lists are for different trees
      const TEntryListArray *elist_array = dynamic_cast< const TEntryListArray *>(elist);
      if (!fSubLists && (!elist_array || !elist_array->GetSubLists())) {  // no sublists in neither
         TEntryList::Add(elist);
         return;
      }
      // Deal with the sublists: Loop over both fSubLists
      // - If the sublists are for the same entry, Add the sublists
      // - For sublists only in this list, check if entry is in elist, and remove the sublist if so
      // - For sublists only in the other list, insert them in fSubLists
      if (!fSubLists && elist_array->GetSubLists()) {
         fSubLists = new TList();
      }
      TEntryListArray *el1;
      const TEntryListArray *el2;
      TCollection *other_sublists = 0;
      if (elist_array) {
         other_sublists = elist_array->GetSubLists();
      }
      TIter next1(fSubLists);
      TIter next2(other_sublists); // should work even if elist->fSubLists is null

      for (el1 = (TEntryListArray*) next1(), el2 = (const TEntryListArray*) next2(); el1 || el2;)  {
         if (el1 && el2 && el1->fEntry == el2->fEntry) { // sublists for the same entry, Add them
            el1->TEntryList::Add(el2);
            el1 = (TEntryListArray*) next1();
            el2 = (const TEntryListArray*) next2();
         } else if (el1 && (!el2 || el1->fEntry < el2->fEntry)) { // el1->fEntry is not in elist->fSubLists
            if ((const_cast<TEntryList*>(elist))->Contains(el1->fEntry)) {
               RemoveSubList(el1);
            }
            el1 = (TEntryListArray*) next1();
         } else { // el2->fEntry is not in fSubLists --> make a copy and add it
            if (!Contains(el2->fEntry)) {
               if (!el1) {
                  fSubLists->AddLast(new TEntryListArray(*el2));
               } else {
                  fSubLists->AddBefore(el1, new TEntryListArray(*el2));
               }
            }
            el2 = (const TEntryListArray*) next2();
         }
      }
      TEntryList::Add(elist);
   }
}

//______________________________________________________________________________
Int_t TEntryListArray::Contains(Long64_t entry, TTree *tree, Long64_t subentry)
{
   //When tree = 0, returns from the current list
   //When tree != 0, finds the list corresponding to this tree
   //When tree is a chain, the entry is assumed to be global index and the local
   //entry is recomputed from the treeoffset information of the chain

   //When subentry != -1, return true if the enter is present and not splitted
   //or if the subentry list is found and contains #subentry

   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray) {
         return currentArray->Contains(localentry, 0, subentry);
      }
      return 0;
   }
   // tree = 0
   Int_t result = TEntryList::Contains(entry);
   if (result && fSubLists) {
      TEntryListArray *t = GetSubListForEntry(entry);
      if (t) {
         result = t->TEntryList::Contains(subentry);
      }
   }
   return result;
}

//______________________________________________________________________________
void TEntryListArray::ConvertToTEntryListArray(TEntryList *e)
{
   // Create a TEntryListArray based on the given TEntryList
   // Called by SetTree when the given list is added to fLists
   // Replace it by a TEntryListArray and delete the given list

   // TODO: Keep the blocks and the number of entries to transfer without copying?
   //    TObjArray *blocks = e->fBlocks;
   //    Int_t NBlocks = e->fNBlocks;
   //    Long64_t N = e->fN;
   //    e->fBlocks = 0;
   //    e->fNBlocks = 0;
   //    e->fN = 0;

   TEntryListArray *earray = new TEntryListArray(*e);
//    earray->fBlocks = blocks;
//    earray->fNBlocks = NBlocks;
//    earray->fN = N;

   if (e == fCurrent) {
      fCurrent = earray;
   }
   // If the list has just been splitted, earray will be the first one
   // and must keep the current sublists
   if (fSubLists) {
      earray->fSubLists = fSubLists;
      fSubLists = 0;
   }
   if (e == fLists->First()) {
      fLists->AddFirst(earray);
   } else {
      fLists->Add(earray);
   }
   fLists->Remove(e);
   delete e;
   e = 0;
}

//________________________________________________________________________
Bool_t TEntryListArray::Enter(Long64_t entry, TTree *tree, Long64_t subentry)
{
   //Add entry #entry (, #subentry) to the list
   //When tree = 0, adds to the current list
   //When tree != 0, finds the list corresponding to this tree (or add a new one)
   //When tree is a chain, the entry is assumed to be global index and the local
   //entry is recomputed from the treeoffset information of the chain

   //When subentry = -1, add all subentries (remove the sublist if it exists)
   //When subentry != -1 and the entry is not present,
   //add only the given subentry, creating a TEntryListArray to hold the subentries for the given entry
   //Return true only if the entry is new (not the subentry)

   Bool_t result = 0;

   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray) {
         if ((result = currentArray->Enter(localentry, 0, subentry)))
            if (fLists) ++fN;
      }
      return result;
   }
   if (fLists) {
      if (!fCurrent) fCurrent = (TEntryList*)fLists->First();
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray && (result = currentArray->Enter(entry, 0, subentry))) {
         ++fN;
      }
      return result;
   }
   // tree = 0 && !fLists
   // Sub entries were already present ?
   TEntryListArray *t = GetSubListForEntry(entry);
   if (t) { // Sub entries were already present
      if (subentry != -1) {
         t->TEntryList::Enter(subentry);
      } else { // remove the sub entries
         RemoveSubList(t);
      }
   } else {
      result = TEntryList::Enter(entry);
      if (subentry != -1 && result) { // a sub entry was given and the entry was not present
         t = SetEntry(entry);
         if (t) t->TEntryList::Enter(subentry);
      }
   }
   return result;
}

//______________________________________________________________________________
TEntryListArray* TEntryListArray::GetSubListForEntry(Long64_t entry, TTree *tree)
{
   // Return the list holding the subentries for the given entry or 0

   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      if (fCurrent) {
         TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
         if (currentArray) {
            return currentArray->GetSubListForEntry(localentry);
         }
      }
      return 0;
   }
   // tree = 0

   if (!fSubLists || !fSubLists->GetEntries()) {
      return 0;
   }

   if (!fSubListIter) {
      fSubListIter = new TIter(fSubLists);
      fLastSubListQueried = (TEntryListArray*) fSubListIter->Next();
   }
   else if (!fLastSubListQueried || entry < fLastSubListQueried->fEntry) {
      // Restart the loop: fLastSubListQueried should point to the newest entry
      // or where we stoped the last search
      // (it is 0 only if we reached the end of the loop)
      fSubListIter->Reset();
      fLastSubListQueried = (TEntryListArray*) fSubListIter->Next();
   }

   if (entry == fLastSubListQueried->fEntry) {
      return fLastSubListQueried;
   }

   while ((fLastSubListQueried = (TEntryListArray*) fSubListIter->Next())) {
      if (fLastSubListQueried->fEntry == entry) {
         return fLastSubListQueried;
      }
      if (fLastSubListQueried->fEntry > entry) {
         break;
      }
   }
   return 0;
}

//______________________________________________________________________________
void TEntryListArray::Print(const Option_t* option) const
{
   //Print this list
   //option = "" - default - print the name of the tree and file
   //option = "all" - print all the entry numbers
   //option = "subentries" - print all the entry numbers and associated subentries
   TString opt = option;
   opt.ToUpper();
   Bool_t new_line = !opt.Contains("EOL");

   if (!opt.Contains("S") && new_line) {
      TEntryList::Print(option);
      return;
   }

   if (fLists) {
      TIter next(fLists);
      TEntryListArray *e = 0;
      while ((e = (TEntryListArray*)next())) {
         std::cout << e->fTreeName << ":" << std::endl;
         e->Print(option);
      }
      return;
   }

   // Print all subentries
   TEntryListArray *tmp = const_cast<TEntryListArray *>(this);
   TIter next(fSubLists);
   TEntryListArray *e = (TEntryListArray*)next();
   for (Int_t i = 0; i < tmp->fN; ++i) {
      Long64_t entry = tmp->GetEntry(i);
      std::cout << entry << " ";
      if (fSubLists) {
         std::cout << " : ";
      }
      if (e && e->fEntry == entry) {
         e->Print("all,EOL");
         e = (TEntryListArray*)next();
      }
      if (new_line) {
         std::cout << std::endl;
      }
   }
}

//______________________________________________________________________________
Bool_t TEntryListArray::Remove(Long64_t entry, TTree *tree, Long64_t subentry)
{
   //Remove entry #entry (, #subentry)  from the list
   //When tree = 0, removes from the current list
   //When tree != 0, finds the list, corresponding to this tree
   //When tree is a chain, the entry is assumed to be global index and the local
   //entry is recomputed from the treeoffset information of the chain
   //If subentry != -1, only the given subentry is removed

   Bool_t result = 0;

   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray && (result = currentArray->Remove(localentry, 0, subentry))) {
         if (fLists) {
            --fN;
         }
      }
      return result;
   }
   if (fLists) {
      if (!fCurrent) fCurrent = (TEntryList*)fLists->First();
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray && (result = currentArray->Remove(entry, 0, subentry)) && fLists) {
         --fN;
      }
      return result;
   }

   // tree = 0 && !fLists
   TEntryListArray *e = GetSubListForEntry(entry);
   if (e) {
      if (subentry != -1) {
         e->TEntryList::Remove(subentry);
      }
      if (subentry == -1 || !e->GetN()) {
         RemoveSubList(e, tree);
         return TEntryList::Remove(entry);
      }
   } else if (subentry == -1) {
      return TEntryList::Remove(entry);
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TEntryListArray::RemoveSubList(TEntryListArray *e, TTree *tree)
{
   // Remove the given sublist and return true if succeeded
   if (!e) return 0;
   if (tree) {
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray) {
         return currentArray->RemoveSubList(e);
      }
   }

   if (!fSubLists->Remove(e)) {
      return 0;
   }
   // fSubLists->Sort(); --> for TObjArray
   delete e;
   e = 0;
   if (!fSubLists->GetEntries()) {
      delete fSubLists;
      fSubLists = 0;
   }
   return 1;
}

//______________________________________________________________________________
Bool_t TEntryListArray::RemoveSubListForEntry(Long64_t entry, TTree *tree)
{
   // Remove the sublists for the given entry --> not being used...

   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray) {
         return currentArray->RemoveSubListForEntry(localentry);
      }
   }
   return RemoveSubList(GetSubListForEntry(entry));
}

//______________________________________________________________________________
void TEntryListArray::Reset()
{
   // Reset all entries and remove all sublists
   TEntryList::Reset();
   if (fSubLists) {
      if (!((TEntryListArray*)fSubLists->First())->GetDirectory()) {
         fSubLists->Delete();
      }
      delete fSubLists;
   }
   delete fSubListIter;
   Init();
}

//______________________________________________________________________________
TEntryListArray* TEntryListArray::SetEntry(Long64_t entry, TTree *tree)
{
   //Create a sublist for the given entry and returns it --> should be called after calling GetSubListForEntry

   if (entry < 0) return 0;

   // If tree is given, switch to the list that contains tree
   if (tree) {
      Long64_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      TEntryListArray *currentArray = dynamic_cast<TEntryListArray*>(fCurrent);
      if (currentArray) {
         return currentArray->SetEntry(localentry);
      }
      return 0;
   }
   // tree = 0
   if (!fSubLists) {
      fSubLists = new TList();
   }
   TEntryListArray *newlist = new TEntryListArray();
   newlist->fEntry = entry;
   if (fLastSubListQueried) {
      fSubLists->AddBefore(fLastSubListQueried, newlist);
      fSubListIter->Reset(); // Reset the iterator to avoid missing the entry next to the new one (bug in TIter?)
   } else {
      fSubLists->AddLast(newlist);
   }
   fLastSubListQueried = newlist;
   return newlist;
}

//______________________________________________________________________________
void TEntryListArray::Subtract(const TEntryList *elist)
{
   //Remove all the entries (and subentries) of this entry list that are contained in elist
   //If for a given entry present in both lists, one has subentries and the other does not, the whole entry is removed

   if (!elist) return;

   if (fLists) { // This list is splitted
      TEntryListArray* e = 0;
      TIter next(fLists);
      fN = 0; // reset fN to set it to the sum of fN in each list
      while ((e = (TEntryListArray*) next())) {
         e->Subtract(elist);
         fN += e->GetN();
      }
   } else if (elist->GetLists()) { // The other list is splitted
      TIter next(elist->GetLists());
      TEntryList *e = 0;
      while ((e = (TEntryList*) next())) {
         Subtract(e);
      }
   } else { // None of the lists are splitted
      if (strcmp(elist->GetTreeName(), fTreeName.Data()) || strcmp(elist->GetFileName(), fFileName.Data()))
         return; // Lists are for different trees
      const TEntryListArray *elist_array = dynamic_cast< const TEntryListArray *>(elist);
      if (!fSubLists || !elist_array || !elist_array->GetSubLists()) {  // there are no sublists in one of the lists
         TEntryList::Subtract(elist);
         if (fSubLists) {
            TEntryListArray *e = 0;
            TIter next(fSubLists);
            while ((e = (TEntryListArray*) next())) {
               if (!Contains(e->fEntry))
                  RemoveSubList(e);
            }
         }
      } else { // Both lists have subentries, will have to loop over them
         TEntryListArray *el1, *el2;
         TIter next1(fSubLists);
         TIter next2(elist_array->GetSubLists());
         el1 = (TEntryListArray*) next1();
         el2 = (TEntryListArray*) next2();

         Long64_t n2 = elist->GetN();
         Long64_t entry;
         for (Int_t i = 0; i < n2; ++i) {
            entry = (const_cast<TEntryList*>(elist))->GetEntry(i);
            // Try to find the sublist for this entry in list
            while (el1 && el1->fEntry < entry) { // && el2
               el1 = (TEntryListArray*) next1();
            }
            while (el2 && el2->fEntry < entry) { // && el1
               el2 = (TEntryListArray*) next2();
            }

            if (el1 && el2 && entry == el1->fEntry && entry == el2->fEntry) { // both lists have sublists for this entry
               el1->Subtract(el2);
               if (!el1->fN) {
                  Remove(entry);
               }
            } else {
               Remove(entry);
            }
         }
      }
   }
}

//______________________________________________________________________________
void TEntryListArray::SetTree(const char *treename, const char *filename)
{
   //If a list for a tree with such name and filename exists, sets it as the current sublist
   //If not, creates this list and sets it as the current sublist

   //  ! the filename is taken as provided, no extensions to full path or url !

   // Uses the method from the base class: if the tree is new, the a new TEntryList will be created (and stored in fLists) and needs to be converted to a TEntryListArray

   Int_t nLists = -1;
   if (fLists) {
      nLists = fLists->GetEntries();
   }
   TEntryList::SetTree(treename, filename);
   if (fLists && fLists->GetEntries() != nLists) { // fList was created and/or has new additions
      if (nLists == -1) {
         // The list has just been splitted (fList was created)
         // There should be two TEntryLists in fLists:
         // must convert both to TEntryListArray
         // and transfer the sublists to the first one
         ConvertToTEntryListArray((TEntryList*) fLists->First());
      }
      ConvertToTEntryListArray((TEntryList*) fLists->Last());
   }
}
