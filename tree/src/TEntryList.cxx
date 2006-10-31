// @(#)root/tree:$Name:  $:$Id: TEntryList.cxx,v 1.1 2006/10/27 09:58:02 brun Exp $
// Author: Anna Kreshuk 27/10/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
// TEntryList
//
// Stores entry numbers. 
//////////////////////////////////////////////////////////////////////////


#include "TEntryList.h"
#include "TEntryListBlock.h"
#include "TTree.h"
#include "TFile.h"

ClassImp(TEntryList)

//______________________________________________________________________________
TEntryList::TEntryList()
{
   //default c-tor

   fLists = 0;
   fCurrent = this;
   fBlocks = 0;
   fN = 0;
   fNBlocks = 0;
   fTreeName = "";
   fFileName = "";
   fTreeNumber = -1;
   fDirectory = 0;

   fLastIndexQueried = -1;
   fLastIndexReturned = 0;
}

//______________________________________________________________________________
TEntryList::TEntryList(const char *name, const char *title):TNamed(name, title)
{
   //c-tor with name and title

   fLists = 0;
   fCurrent = this;
   fBlocks = 0;
   fN = 0;
   fNBlocks = 0;
   fTreeName = "";
   fFileName = "";
   fTreeNumber = -1;
   fDirectory  = gDirectory;
   gDirectory->Append(this);

   fLastIndexQueried = -1;
   fLastIndexReturned = 0;

}

//______________________________________________________________________________
TEntryList::TEntryList(const char *name, const char *title, const TTree *tree):TNamed(name, title)
{
   //constructor with name and title, which also sets the tree

   fLists = 0;
   fCurrent = this;
   fBlocks = 0;
   fN = 0;
   fNBlocks = 0;
   fTreeNumber = -1;
   SetTree(tree);
   fTreeName = tree->GetName();
   fFileName = tree->GetCurrentFile()->GetName();

   fDirectory  = gDirectory;
   gDirectory->Append(this);

   fLastIndexQueried = -1;
   fLastIndexReturned = 0;

}

//______________________________________________________________________________
TEntryList::TEntryList(const char *name, const char *title, const char *treename, const char *filename):TNamed(name, title)
{
   //c-tor with name and title, which also sets the treename and the filename

   fLists = 0;
   fCurrent = this;
   fBlocks = 0;
   fNBlocks = 0;
   fN = 0;
   SetTree(treename, filename);
   fTreeNumber = -1;
   fDirectory  = gDirectory;
   gDirectory->Append(this);

   fLastIndexQueried = -1;
   fLastIndexReturned = 0;
}

//______________________________________________________________________________
TEntryList::TEntryList(const TTree *tree)
{
   //c-tor, which sets the tree

   fLists = 0;
   fCurrent = this;
   fBlocks = 0;
   fNBlocks = 0;
   fN = 0;

   SetTree(tree);
   fTreeNumber = -1;
   fDirectory  = gDirectory;
   gDirectory->Append(this);

   fLastIndexQueried = -1;
   fLastIndexReturned = 0;
}

//______________________________________________________________________________
TEntryList::TEntryList(const TEntryList &elist) : TNamed(elist)
{
   //copy c-tor

   fNBlocks = elist.fNBlocks;
   fTreeName = elist.fTreeName;
   fFileName = elist.fFileName;
   fTreeNumber = elist.fTreeNumber;
   fN = elist.fN;
   fLists = 0;
   fBlocks = 0;
   if (elist.fLists){
      fLists = new TList();
      TEntryList *el1 = 0;
      TEntryList *el2 = 0;
      TIter next(elist.fLists);
      while((el1 = (TEntryList*)next())){
         el2 = new TEntryList(*el1);
         if (el1==elist.fCurrent)
            fCurrent = el2;
         fLists->Add(el2);
      }
   } else {
      if (elist.fBlocks){
         TEntryListBlock *block1 = 0;
         TEntryListBlock *block2 = 0;
         //or just copy it as a TObjArray??
         fBlocks = new TObjArray();
         for (Int_t i=0; i<fNBlocks; i++){
            block1 = (TEntryListBlock*)elist.fBlocks->UncheckedAt(i);
            block2 = new TEntryListBlock(*block1);
            fBlocks->Add(block2);
         }
      }
      fCurrent = this;
   }
   fDirectory  = 0;

}


//______________________________________________________________________________
TEntryList::~TEntryList()
{
// d-tor
// !!! check for memory leaks!!!

   if (fBlocks){
      fBlocks->Delete();
      delete fBlocks;
      
   }
   fBlocks = 0;
   //if (fLists){
      //printf("deleting lists\n");
   // fLists->Delete();
   // delete fLists;
   //}
   
   //fLists = 0;
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fDirectory  = 0;
   fLists = 0;
}

//______________________________________________________________________________
void TEntryList::Add(const TEntryList *elist)
{
   //Add 2 entry lists

   if (fN==0){
      //this list is empty. copy the other list completely ??
      fNBlocks = elist->fNBlocks;
      fTreeName = elist->fTreeName;
      fFileName = elist->fFileName;
      fTreeNumber = elist->fTreeNumber;
      fN = elist->fN;
      if (elist->fLists){
         fLists = new TList();
         TEntryList *el1 = 0;
         TEntryList *el2 = 0;
         TIter next(elist->fLists);
         while((el1 = (TEntryList*)next())){
            el2 = new TEntryList(*el1);
            if (el1==elist->fCurrent)
               fCurrent = el2;
            fLists->Add(el2);
         }
      } else {
         if (elist->fBlocks){
            TEntryListBlock *block1 = 0;
            TEntryListBlock *block2 = 0;
            //or just copy it as a TObjArray??
            fBlocks = new TObjArray();
            for (Int_t i=0; i<fNBlocks; i++){
               block1 = (TEntryListBlock*)elist->fBlocks->UncheckedAt(i);
               block2 = new TEntryListBlock(*block1);
               fBlocks->Add(block2);
            }
         }
         fCurrent = this;

      }
      return;
   }

   if (!fLists){
      if (!strcmp(elist->fTreeName.Data(),fTreeName.Data()) && !strcmp(elist->fFileName.Data(),fFileName.Data())){
         //entry lists are for the same tree
         if (!elist->fBlocks)
            //the other list is empty list
            return;
         if (!fBlocks){
            //this entry list is empty
            fBlocks = new TObjArray(*elist->fBlocks);
            return;
         }
         //both not empty, merge block by block
         TEntryListBlock *block1=0;
         TEntryListBlock *block2=0;
         Int_t i;
         Int_t nmin = TMath::Min(fNBlocks, elist->fNBlocks);
         Int_t nnew;
         for (i=0; i<nmin; i++){
            block1 = (TEntryListBlock*)fBlocks->UncheckedAt(i);
            block2 = (TEntryListBlock*)elist->fBlocks->UncheckedAt(i);
            nnew = block1->Merge(block2);
            fN=nnew;
         }
         if (fNBlocks<elist->fNBlocks){
            Int_t nmax = elist->fNBlocks; 
            for (i=nmin; i<nmax; i++){
               block2 = (TEntryListBlock*)elist->fBlocks->UncheckedAt(i);
               block1 = new TEntryListBlock(*block2);
               fBlocks->Add(block1);
               fN+=block1->GetNPassed();
               fNBlocks++;
            }
         }
      } else {
      //entry lists are for different trees. create a chain entry list with
      //2 sub lists for the first and second entry lists
            fLists = new TList();
            TEntryList *el = new TEntryList();
            el->fTreeName = fTreeName;
            el->fFileName = fFileName;
            el->fBlocks = fBlocks;
            fBlocks = 0;
            el->fNBlocks = fNBlocks;
            fLists->Add(el);
            el = new TEntryList(*elist);
            fLists->Add(el);
            fCurrent = el;
            fN+=el->GetN();
      }
   } else {
      //there are already some sublists in this list, just add another one
      if (!elist->fLists){
         //the other list doesn't have sublists
         TIter next(fLists);
         TEntryList *el = 0; 
         Bool_t found = kFALSE;
         while ((el = (TEntryList*)next())){
            if (!strcmp(el->fTreeName.Data(), elist->fTreeName.Data()) && 
                !strcmp(el->fFileName.Data(), elist->fFileName.Data())){
               //found a list for the same tree
               el->Add(elist);
               found = kTRUE;
               break;
            }
         }
         if (!found){       
            el = new TEntryList(*elist);
            fLists->Add(el);
            fN+=el->GetN();
         }
      } else {
         //add all sublists from the other list
         TEntryList *el = 0;
         TIter next(elist->fLists);
         while ((el = (TEntryList*)next())){
            Add(el);
         }
      }
   }

}


//______________________________________________________________________________
Int_t TEntryList::Contains(Long64_t entry, TTree *tree)
{
//When tree = 0, returns from the current list
//When tree != 0, finds the list, corresponding to this tree
//When tree is a chain, the entry is assumed to be global index and the local
//entry is recomputed from the treeoffset information of the chain

   if (!tree){
      if (fBlocks) {
         //this entry list doesn't contain any sub-lists
         TEntryListBlock *block = 0;
         Int_t nblock = entry/kBlockSize;
         if (nblock >= fNBlocks) return 0;
         block = (TEntryListBlock*)fBlocks->UncheckedAt(nblock);
         return block->Contains(entry-nblock*kBlockSize);
      }
      if (fLists) {
         return fCurrent->Contains(entry);
      }
   } else {
      tree->LoadTree(entry);
      SetTree(tree->GetTree());
      if (fCurrent)
         return fCurrent->Contains(entry);
   }
   return 0;

}

//________________________________________________________________________
Bool_t TEntryList::Enter(Long64_t entry, TTree *tree)
{
//Add entry #entry to the list
//When tree = 0, returns from the current list
//When tree != 0, finds the list, corresponding to this tree
//When tree is a chain, the entry is assumed to be global index and the local
//entry is recomputed from the treeoffset information of the chain

   if (!tree){
      if (!fLists) {
         if (!fBlocks) fBlocks = new TObjArray();
         TEntryListBlock *block = 0;
         Int_t nblock = entry/kBlockSize;
         if (nblock >= fNBlocks) {
            if (fNBlocks>0){
               block = (TEntryListBlock*)fBlocks->UncheckedAt(fNBlocks-1);
               if (!block) return 0;
               block->OptimizeStorage();
            }
            for (Int_t i=fNBlocks; i<=nblock; i++){
               block = new TEntryListBlock();
               fBlocks->Add(block);
            }
         fNBlocks = nblock+1;
         }
         block = (TEntryListBlock*)fBlocks->UncheckedAt(nblock);
         if (block->Enter(entry-nblock*kBlockSize)) {
            fN++;
            return 1;
         }
      } else {
         //the entry in the current entry list
         if (fCurrent->Enter(entry)) {
            fN++;
            return 1;
         }
      }
   } else {
      Int_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      if (fCurrent){
         if (fCurrent->Enter(localentry)) {
            fN++;
            return 1;
         }
      }
   }
   return 0;

}

//______________________________________________________________________________
Bool_t TEntryList::Remove(Long64_t entry, TTree *tree)
{
//Remove entry #entry from the list
//When tree = 0, returns from the current list
//When tree != 0, finds the list, corresponding to this tree
//When tree is a chain, the entry is assumed to be global index and the local
//entry is recomputed from the treeoffset information of the chain


   if (!tree){
      if (!fLists) {
         if (!fBlocks) return 0;
         TEntryListBlock *block = 0;
         Int_t nblock = entry/kBlockSize;
         block = (TEntryListBlock*)fBlocks->UncheckedAt(nblock);
         if (!block) return 0;
         Long64_t blockindex = entry - nblock*kBlockSize;
         if (block->Remove(blockindex)){
            fN--;
            return 1;
         }
      } else {
         if (fCurrent->Remove(entry)){
            fN--;
            return 1;
         }
      }
   } else {
      Int_t localentry = tree->LoadTree(entry);
      SetTree(tree->GetTree());
      if (fCurrent){
         if (fCurrent->Remove(localentry)) {
            fN--;
            return 1;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Long64_t TEntryList::GetEntry(Int_t index)
{
   //return the number of the entry #index of this TEntryList in the TTree or TChain
   //See also Next().

   if (index>=fN){
      return -1;
   }
   if (index==fLastIndexQueried+1){
      //in a loop
      return Next();
   } else {
      if (fBlocks) {
         TEntryListBlock *block = 0;
         Long64_t total_passed = 0;
         Int_t i=0;
         while (total_passed<=index && i<fNBlocks){
            block=(TEntryListBlock*)fBlocks->UncheckedAt(i);
            total_passed+=block->GetNPassed();
            i++;
         }
         i--;
         total_passed-=block->GetNPassed();
         if (i!=fLastIndexReturned/kBlockSize){
            block = (TEntryListBlock*)fBlocks->UncheckedAt(fLastIndexReturned/kBlockSize);
            block->ResetIndices();
            block = (TEntryListBlock*)fBlocks->UncheckedAt(i);
         }

         Long64_t localindex = index - total_passed;
         Long64_t blockindex = block->GetEntry(localindex);
         if (blockindex < 0) return -1;
         Long64_t res = i*kBlockSize + blockindex;
         fLastIndexQueried = index;
         fLastIndexReturned = res;
         return res;
      } else {
         //find the corresponding list
         TIter next(fLists);
         TEntryList *templist;
         Long64_t ntotal = 0;
         if (fCurrent){
            //reset all indices of the current list
            Int_t currentblock = (fCurrent->fLastIndexReturned)/kBlockSize;
            TEntryListBlock *block = (TEntryListBlock*)fCurrent->fBlocks->UncheckedAt(currentblock);
            block->ResetIndices();
            fCurrent->fLastIndexReturned = 0;
            fCurrent->fLastIndexQueried = -1;

         }
         while ((templist = (TEntryList*)next())){
            ntotal += templist->GetN();
            if (ntotal > index)
              break;
         }
         fCurrent = templist;
         Long64_t localentry = index - (ntotal - fCurrent->GetN());
         fLastIndexQueried = index;
         fLastIndexReturned = fCurrent->GetEntry(localentry);
         return fLastIndexReturned;
      }

   }
   return -1;
}

//______________________________________________________________________________
Long64_t TEntryList::GetEntryAndTree(Int_t index, Int_t &treenum)
{
//return the index of "index"-th non-zero entry in the TTree or TChain
//and the # of the corresponding tree in the chain

   Long64_t result = GetEntry(index);
   treenum = fCurrent->fTreeNumber;
   return result;
}

//______________________________________________________________________________
TEntryList *TEntryList::GetEntryList(const char *treename, const char *filename)
{
   //return the entry list, correspoding to treename and filename

   if (!fLists){
      if (!strcmp(treename, fTreeName.Data()) && !(strcmp(filename, fFileName.Data()))){
         return this;
      } else {
         return 0;
      }
   }
   TIter next(fLists);
   TEntryList *templist;
   while ((templist = (TEntryList*)next())){
      if (!strcmp(treename, templist->GetTreeName()) && !(strcmp(filename, templist->GetFileName()))){
         return templist;
      }
   }
   return 0;
}
      
//______________________________________________________________________________
Int_t TEntryList::Merge(TCollection *list)
{
   //Merge this list with the lists from the collection

   if (!list) return -1;
   TIter next(list);
   TEntryList *elist = 0;
   while ((elist = (TEntryList*)next())) {
      if (!elist->InheritsFrom(TEntryList::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s",elist->ClassName(),this->ClassName());
         return -1;
      }
      Add(elist);
   }
   return 0;   
}

//______________________________________________________________________________
Long64_t TEntryList::Next()
{
   //return the next non-zero entry index (next after fLastIndexQueried)
   //this function is faster than GetEntry()

   Long64_t result;
   if (fN == fLastIndexQueried+1 || fN==0){
      return -1;
   }
   if (fBlocks){
      Int_t iblock = fLastIndexReturned/kBlockSize;
      TEntryListBlock *current_block = (TEntryListBlock*)fBlocks->UncheckedAt(iblock);
      result = current_block->Next();
      if (result>=0) {
         fLastIndexQueried++;
         fLastIndexReturned = result+kBlockSize*iblock;
         return fLastIndexReturned;
      }
      else {
         while (result<0 && iblock<fNBlocks-1) {
            current_block->ResetIndices();
            iblock++;
            current_block = (TEntryListBlock*)fBlocks->UncheckedAt(iblock);
            current_block->ResetIndices();
            result = current_block->Next();
         }
         if (result<0) {
            fLastIndexQueried = -1;
            fLastIndexReturned = 0;
            return -1;
         }
         fLastIndexQueried++;
         fLastIndexReturned = result+kBlockSize*iblock;

         return fLastIndexReturned;
      }
   } else {
      result = fCurrent->Next();
      if (result>=0) {
         fLastIndexQueried++;
         fLastIndexReturned = result;
         return result;
      } else {
         if (fCurrent){
            //reset all indices of the current list
            Int_t currentblock = (fCurrent->fLastIndexReturned)/kBlockSize;
            TEntryListBlock *block = (TEntryListBlock*)fCurrent->fBlocks->UncheckedAt(currentblock);
            block->ResetIndices();
            fCurrent->fLastIndexReturned = 0;
            fCurrent->fLastIndexQueried = -1;

         }

         //find the list with the next non-zero entry
         while (result<0 && fCurrent!=((TEntryList*)fLists->Last())){
            fCurrent->fLastIndexQueried = -1;
            fCurrent->fLastIndexReturned = 0;
            fCurrent = (TEntryList*)fLists->After(fCurrent);
            result = fCurrent->Next();
         }
         fLastIndexQueried++;
         fLastIndexReturned = result;
         return result;
      }
   }
}


//______________________________________________________________________________
void TEntryList::OptimizeStorage()
{
   //Checks if the array representation is more economical and if so, switches to it

   if (fBlocks){
      TEntryListBlock *block = 0;
      for (Int_t i=0; i<fNBlocks; i++){
         block = (TEntryListBlock*)fBlocks->UncheckedAt(i);
         block->OptimizeStorage();
      }
   }
}


//______________________________________________________________________________
void TEntryList::Print(const Option_t* option) const
{
   //Print this list
   //option = "" - default - print the name of the tree and file
   //option = "all" - print all the entry numbers

   TString opt = option;
   opt.ToUpper();
   if (fBlocks) {
      printf("%s %s\n", fTreeName.Data(), fFileName.Data());
      if (opt.Contains("A")){
         TEntryListBlock* block = 0;
         for (Int_t i=0; i<fNBlocks; i++){
            block = (TEntryListBlock*)fBlocks->UncheckedAt(i);
            Int_t shift = i*kBlockSize;
            block->PrintWithShift(shift);
         }
      }
   }
   else{
      TEntryList *elist = 0;
      TIter next(fLists);
      while((elist = (TEntryList*)next())){
         elist->Print(option);
      }
   }
   
}

//______________________________________________________________________________
void TEntryList::Reset()
{
   //Reset this list

   //Maybe not delete, but just reset the number of blocks to 0????

   if (fBlocks){
      fBlocks->Delete();
      delete fBlocks;
      fBlocks = 0;
   }
   if (fLists){
      if (!((TEntryList*)fLists->First())->GetDirectory()){
         fLists->Delete();
         printf("not in the curren directory\n");
      }
      delete fLists;
      fLists = 0;
   }
   fCurrent = this;
   fBlocks = 0;
   fNBlocks = 0;
   fN = 0;
   fTreeName = "";
   fFileName = "";
   fTreeNumber = -1;
   fLastIndexQueried = -1;
   fLastIndexReturned = 0;
}

//______________________________________________________________________________
void TEntryList::SetTree(const char *treename, const char *filename)
{
   //If a list for a tree with such name and filename exists, sets it as the current sublist
   //If not, creates this list and sets it as the current sublist

   TEntryList *elist = 0;
   if (fCurrent->fTreeName==treename && fCurrent->fFileName==filename){
      //this tree's entry list is current, do nothing
      return;
   }
   if (fLists) {
      //find the corresponding entry list and make it current

      TIter next(fLists);
      while ((elist = (TEntryList*)next())){
         if (elist->fTreeName==treename && elist->fFileName== filename){
            fCurrent = elist;
            return;
         }
      }
      //didn't find an entry list for this tree, create a new one
      elist = new TEntryList("", "", treename, filename);
      fLists->Add(elist);
      fCurrent = elist;
      return;
   } else {
      if (fBlocks){
         if(fTreeName!=treename || fFileName!=filename){
         //we have a chain and already have an entry list for the first tree
         //move the first entry list to the fLists
            fLists = new TList();
            elist = new TEntryList();
            elist->fTreeName = fTreeName;
            elist->fFileName = fFileName;
            elist->fN = fN;
            elist->fTreeNumber = fTreeNumber;
            elist->fBlocks = fBlocks;
            fBlocks = 0;
            elist->fNBlocks = fNBlocks;
            fLists->Add(elist);
            elist = new TEntryList("", "", treename, filename);
            fLists->Add(elist);
            fCurrent = elist;
         }
         else {
            //same tree as in the current entry list, don't do anything
            return;
         }
      } else {
         fTreeName = treename;
         fFileName = filename;
      }
   }
}

//______________________________________________________________________________
void TEntryList::SetTree(const TTree *tree)
{
   //If a list for a tree with such name and filename exists, sets it as the current sublist
   //If not, creates this list and sets it as the current sublist

   TString treename = tree->GetTree()->GetName();
   TString filename = tree->GetTree()->GetCurrentFile()->GetName();
   SetTree(treename, filename);

}

//______________________________________________________________________________
void TEntryList::Subtract(const TEntryList *elist)
{
   //remove all the entries of this entry list, that are contained in elist

   TEntryList *templist = 0;
   if (!fLists){
      if (!fBlocks) return;
      //check if lists are for the same tree
      if (!elist->fLists){
         //second list is also only for 1 tree
         if (!strcmp(elist->fTreeName.Data(),fTreeName.Data()) && 
             !strcmp(elist->fFileName.Data(),fFileName.Data())){
            //same tree
            Long64_t n2 = elist->GetN();
            Long64_t entry;
            for (Int_t i=0; i<n2; i++){
               entry = (const_cast<TEntryList*>(elist))->GetEntry(i);
               Remove(entry);
            }
         } else {
            //different trees
            return;
         }
      } else {
         //second list has sublists, try to find one for the same tree as this list
         TIter next1(elist->GetLists());
         templist = 0;
         Bool_t found = kFALSE;
         while ((templist = (TEntryList*)next1())){
            if (!strcmp(templist->fTreeName.Data(),fTreeName.Data()) && 
                !strcmp(templist->fFileName.Data(),fFileName.Data())){
               found = kTRUE;
               break;
            }
         }
         if (found) {
            Subtract(templist);
         }
      } 
   } else {
      //this list has sublists
      TIter next2(fLists);
      templist = 0;
      while ((templist = (TEntryList*)next2())){
         templist->Subtract(elist);
      }
   }
   return;


}

//______________________________________________________________________________
TEntryList operator||(TEntryList &elist1, TEntryList &elist2)
{
   TEntryList eresult = elist1;
   //eresult = elist1;
   // printf("internal in operator1\n");
   eresult.Print("all");
   eresult.Add(&elist2);
   // printf("internal in operator2\n");
   eresult.Print("all");

   return eresult;
}


