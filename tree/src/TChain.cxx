// @(#)root/tree:$Name:  $:$Id: TChain.cxx,v 1.10 2000/10/09 13:56:40 brun Exp $
// Author: Rene Brun   03/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChain                                                               //
//                                                                      //
// A chain is a collection of files containg TTree objects.             //
// When the chain is created, the first parameter is the default name   //
// for the Tree to be processed later on.                               //
//                                                                      //
// Enter a new element in the chain via the TChain::Add function.       //
// Once a chain is defined, one can use the normal TTree functions      //
// to Draw,Scan,etc.                                                    //
//                                                                      //
// Use TChain::SetBranchStatus to activate one or more branches for all //
// the trees in the chain.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TChain.h"
#include "TTree.h"
#include "TFile.h"
#include "TSelector.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBrowser.h"
#include "TChainElement.h"

#include <math.h>
#include <float.h>

ClassImp(TChain)

//______________________________________________________________________________
TChain::TChain(): TTree()
{
//*-*-*-*-*-*Default constructor for Chain*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==============================

   fTreeOffsetLen  = 100;
   fNtrees         = 0;
   fTreeNumber     = -1;
   fTreeOffset     = 0;
   fTree           = 0;
   fFile           = 0;
   fFiles          = new TObjArray(fTreeOffsetLen );
   fStatus         = new TList();
   fNotify         = 0;
}

//______________________________________________________________________________
TChain::TChain(const char *name, const char *title)
       :TTree(name,title)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a Chain*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============
//
//   A TChain is a collection of TFile objects.
//    the first parameter "name" is the name of the TTree object
//    in the files added with Add.
//   Use TChain::Add to add a new element to this chain.
//
//    Example:
//  Suppose we have 3 files f1.root, f2.root and f3.root. Each file
//  contains a TTree object named "T".
//     TChain ch("T");  creates a chain to process a Tree called "T"
//     ch.Add("f1.root");
//     ch.Add("f2.root");
//     ch.Add("f3.root");
//     ch.Draw("x");
//       The Draw function above will process the variable "x" in Tree "T"
//       reading sequentially the 3 files in the chain ch.
//
//*-*

   fTreeOffsetLen  = 100;
   fNtrees         = 0;
   fTreeNumber     = -1;
   fTreeOffset     = new Int_t[fTreeOffsetLen];
   fTree           = 0;
   fFile           = 0;
   fFiles          = new TObjArray(fTreeOffsetLen );
   fStatus         = new TList();
   fTreeOffset[0]  = 0;
   TChainElement *element = new TChainElement("*","");
   fStatus->Add(element);
   gDirectory->GetList()->Remove(this);
   gROOT->GetListOfSpecials()->Add(this);
   fDirectory = 0;
   fNotify    = 0;
}

//______________________________________________________________________________
TChain::~TChain()
{
//*-*-*-*-*-*Default destructor for a Chain*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==============================

   fDirectory = 0;
   if (fFile) {delete fFile; fFile = 0;}
   gROOT->GetListOfSpecials()->Remove(this);
   delete [] fTreeOffset;
   fFiles->Delete();
   delete fFiles;
   fStatus->Delete();
   delete fStatus;
}


//______________________________________________________________________________
void TChain::Add(const char *name, Int_t nentries)
{
//       Add a new element to this chain.
//       An element can be the name of another chain or the name of a file
//       containing a tree.
//    name may have the following format:
//       //machine/file_name.root/subdir/tree_name
//      machine, subdir and tree_name are optional. If tree_name is missing,
//      the chain name will be assumed.
//
//    if nentries < 0, the file is connected and the tree header read in memory
//    to get the number of entries.
//    if (nentries >= 0, the file is not connected, nentries is assumed to be
//    the number of entries in the file. In this case, no check is made that
//    the file exists and the Tree existing in the file. This second mode
//    is interesting in case the number of entries in the file is already stored
//    in a run data base for example.

   TDirectory *cursav = gDirectory;
   char *treename = (char*)GetName();
   char *dot = (char*)strstr(name,".root");
   if (!dot) {
      Error("Add","a chain element name must contain the string .root");
      return;
   }

   //Check enough space in fTreeOffset
   if (fNtrees+1 >= fTreeOffsetLen) {
      fTreeOffsetLen *= 2;
      Int_t *trees = new Int_t[fTreeOffsetLen];
      for (Int_t i=0;i<=fNtrees;i++) trees[i] = fTreeOffset[i];
      delete [] fTreeOffset;
      fTreeOffset = trees;
   }

   //Search for a a slash between the .root and the end
   Int_t nch = strlen(name) + strlen(treename);
   char *filename = new char[nch+1];
   strcpy(filename,name);
   char *pos = (char*)strstr(filename,".root") + 5;
   while (*pos) {
      if (*pos == '/') {
         treename = pos+1;
         *pos = 0;
         break;
      }
      pos++;
   }

   //Connect the file to get the number of entries
   Int_t pksize = 0;
   if (nentries <= 0) {
      TFile *file = TFile::Open(filename);
      if (file->IsZombie()) {
         delete file;
         delete [] filename;
         return;
      }

   //Check that tree with the right name exists in the file
      TObject *obj = file->Get(treename);
      if (obj && !obj->InheritsFrom("TTree") ) {
         strcat(treename,"/");
         strcat(treename,(char*)GetName());
         obj = file->Get(treename);
      }
      if (!obj || !obj->InheritsFrom("TTree") ) {
         Error("Add","cannot find tree with name %s", treename);
         delete file;
         delete [] filename;
         return;
      }
      TTree *tree = (TTree*)obj;
      nentries = (Int_t)tree->GetEntries();
      pksize   = tree->GetPacketSize();
      delete tree;
      delete file;
   }

   fTreeOffset[fNtrees+1] = fTreeOffset[fNtrees] + nentries;
   fNtrees++;
   fEntries += nentries;

   TChainElement *element = new TChainElement(treename,filename);
   element->SetPacketSize(pksize);
   element->SetNumberEntries(nentries);
   fFiles->Add(element);

   delete [] filename;
   if (cursav) cursav->cd();
}

//______________________________________________________________________________
void TChain::Browse(TBrowser *)
{

}

//_______________________________________________________________________
void TChain::CreatePackets()
{
//*-*-*-*-*-*-*-*-*Initialize the packet descriptor string*-*-*-*-*-*-*-*-*-*
//*-*              =======================================

   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      element->CreatePackets();
   }
}

//______________________________________________________________________________
Int_t TChain::Draw(const char *varexp, TCut selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
   // Draw expression varexp for selected entries.
   //
   // This function accepts TCut objects as arguments.
   // Useful to use the string operator +, example:
   //    ntuple.Draw("x",cut1+cut2+cut3);
   //

   return TChain::Draw(varexp, selection.GetTitle(), option, nentries, firstentry);
}

//______________________________________________________________________________
Int_t TChain::Draw(const char *varexp, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
{
   // Process all entries in this chain and draw histogram
   // corresponding to expression varexp.

   if (LoadTree(firstentry) < 0) return 0;
   return TTree::Draw(varexp,selection,option,nentries,firstentry);
}


//______________________________________________________________________________
TBranch *TChain::GetBranch(const char *name)
{
//*-*-*-*-*-*-*-*-*Return pointer to the branch name*-*-*-*-*
//*-*              ==========================================

   if (fTree) return fTree->GetBranch(name);
   LoadTree(0);
   if (fTree) return fTree->GetBranch(name);
   return 0;
}

//______________________________________________________________________________
Int_t TChain::GetChainEntryNumber(Int_t entry)
{
// return absolute entry number in the chain
// the input parameter entry is the entry number in the current Tree of this chain

  return entry + fTreeOffset[fTreeNumber];
}

//______________________________________________________________________________
Int_t TChain::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*-*-*-*-*Return entry in memory*-*-*-*-*-*-*-*-*-*
//*-*              ======================
//     getall = 0 : get only active branches
//     getall = 1 : get all branches

   if (LoadTree(entry) < 0) return 0;
   return fTree->GetEntry(fReadEntry,getall);
}


//______________________________________________________________________________
TLeaf *TChain::GetLeaf(const char *name)
{
//*-*-*-*-*-*-*-*-*Return pointer to the leaf name*-*-*-*-*
//*-*              ==========================================

   if (fTree) return fTree->GetLeaf(name);
   LoadTree(0);
   if (fTree) return fTree->GetLeaf(name);
   return 0;
}


//______________________________________________________________________________
TObjArray *TChain::GetListOfBranches()
{
//*-*-*-*-*-*-*-*-*Return pointer to list of branches of current tree*-*-*-*-*
//*-*              ================================================

   if (fTree) return fTree->GetListOfBranches();
   LoadTree(0);
   if (fTree) return fTree->GetListOfBranches();
   return 0;
}


//______________________________________________________________________________
TObjArray *TChain::GetListOfLeaves()
{
//*-*-*-*-*-*-*-*-*Return pointer to list of leaves of current tree*-*-*-*-*
//*-*              ================================================

   if (fTree) return fTree->GetListOfLeaves();
   LoadTree(0);
   if (fTree) return fTree->GetListOfLeaves();
   return 0;
}

//______________________________________________________________________________
Double_t TChain::GetMaximum(const char *columname)
{
//*-*-*-*-*-*-*-*-*Return maximum of column with name columname*-*-*-*-*-*-*
//*-*              ============================================

   Double_t theMax = -FLT_MAX;  //in float.h
   for (Int_t file=0;file<fNtrees;file++) {
      Int_t first = fTreeOffset[file];
      LoadTree(first);
      Double_t curmax = fTree->GetMaximum(columname);;
      if (curmax > theMax) theMax = curmax;
   }
   return theMax;
}


//______________________________________________________________________________
Double_t TChain::GetMinimum(const char *columname)
{
//*-*-*-*-*-*-*-*-*Return minimum of column with name columname*-*-*-*-*-*-*
//*-*              ============================================

   Double_t theMin = FLT_MAX; //in float.h
   for (Int_t file=0;file<fNtrees;file++) {
      Int_t first = fTreeOffset[file];
      LoadTree(first);
      Double_t curmin = fTree->GetMinimum(columname);;
      if (curmin < theMin) theMin = curmin;
   }
   return theMin;
}


//______________________________________________________________________________
Int_t TChain::GetNbranches()
{
//*-*-*-*-*-*-*-*-*Return number of branches of current tree*-*-*-*-*
//*-*              =========================================

   if (fTree) return fTree->GetNbranches();
   LoadTree(0);
   if (fTree) return fTree->GetNbranches();
   return 0;
}


//______________________________________________________________________________
Int_t TChain::LoadTree(Int_t entry)
{
//  The input argument entry is the entry serial number in the whole chain.
//  The function finds the corresponding Tree and returns the entry number
//  in this tree.
   
   if (!fNtrees) return 1;
   if (entry < 0 || entry > fEntries) return -2;

   //Find in which tree this entry belongs to
   Int_t t;
   for (t=0;t<fNtrees;t++) {
      if (entry < fTreeOffset[t+1]) break;
   }

   fReadEntry = entry - fTreeOffset[t];
   // If entry belongs to the current tree return entry
   if (t == fTreeNumber) {
      return fReadEntry;
   }

   //Delete current tree and connect new tree
   TDirectory *cursav = gDirectory;
   delete fFile; fFile = 0;
   TChainElement *element = (TChainElement*)fFiles->At(t);
   if (!element) return -4;
   fFile = TFile::Open(element->GetTitle());
   if (fFile->IsZombie()) {
      delete fFile; fFile = 0;
      return -3;
   }
   fTree = (TTree*)fFile->Get(element->GetName());
   fTreeNumber = t;
   fDirectory = fFile;

   //Set the branches status and address for the newly connected file
   fTree->SetMaxVirtualSize(fMaxVirtualSize);
   SetChainOffset(fTreeOffset[t]);
   TIter next(fStatus);
   Int_t status;
   while ((element = (TChainElement*)next())) {
      status = element->GetStatus();
      if (status >=0) fTree->SetBranchStatus(element->GetName(),status);
      void *add = element->GetBaddress();
      if (add) fTree->SetBranchAddress(element->GetName(),add);
   }

   if (cursav) cursav->cd();

   //Notify user if requested
   if (fNotify) fNotify->Notify();

   return fReadEntry;
}

//______________________________________________________________________________
void TChain::Loop(Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Loop on nentries of this chain starting at firstentry
//*-*              ===================================================


   Error("Loop","Function not yet implemented");

   if (option || nentries || firstentry) { }  // keep warnings away

#ifdef NEVER
   if (LoadTree(firstentry) < 0) return;

   if (firstentry < 0) firstentry = 0;
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > fEntries-1) {
      lastentry = (Int_t)fEntries -1;
   }

   GetPlayer();
   GetSelector();
   fSelector->Start(option);

   Int_t entry = firstentry;
   Int_t tree,e0,en;
   for (tree=0;tree<fNtrees;tree++) {
      e0 = fTreeOffset[tree];
      en = fTreeOffset[tree+1] - 1;
      if (en > lastentry) en = lastentry;
      if (entry > en) continue;

      LoadTree(entry);
      fSelector->BeginFile();

      while (entry <= en) {
         fSelector->Execute(fTree, entry - e0);
         entry++;
      }
      fSelector->EndFile();
   }

   fSelector->Finish(option);
#endif
}


//______________________________________________________________________________
void TChain::ls(Option_t *option)
{
   fFiles->ls(option);
}

//______________________________________________________________________________
void TChain::Merge(const char *name)
{
//     Merge all files in this chain into a new file
// see important note in the following function Merge

   TFile *file = TFile::Open(name,"recreate","chain files",1);
   Merge(file,0,"");
   file->Close();
   delete file;
}


//______________________________________________________________________________
void TChain::Merge(TFile *file, Int_t basketsize, Option_t *option)
{
//     Merge all files in this chain into a new file
//     if option ="C" is given, the compression level for all branches
//        in the new Tree is set to the file compression level.
//     By default, the compression level of all branches is the
//     original compression level in the old Trees.
//
//     if (basketsize > 1000, the basket size for all branches of the
//     new Tree will be set to basketsize.
//
// IMPORTANT: Before invoking this function, the branch addresses
//            of the TTree must have been set.
//  example using the file generated in $ROOTSYS/test/Event
//  merge two copies of Event.root
//
//        gSystem.Load("libEvent");
//        Event *event = new Event();
//        TChain ch("T");
//        ch.SetBranchAddress("event",&event);
//        ch.Add("Event1.root");
//        ch.Add("Event2.root");
//        ch.Merge("all.root");
//
//  The SetBranchAddress statement is not necessary if the Tree
//  contains only basic types (case of files converted from hbook)

   if (!file) return;
   TObjArray *lbranches = GetListOfBranches();
   if (!lbranches) return;
   if (!fTree) return;

// Clone Chain tree
   file->cd();
   TTree *hnew = (TTree*)fTree->CloneTree(0);
   hnew->SetAutoSave(2000000000);

// May be reset branches compression level?
   TBranch *branch;
   TIter nextb(hnew->GetListOfBranches());
   if (strstr(option,"c") || strstr(option,"C")) {
      while ((branch = (TBranch*)nextb())) {
         branch->SetCompressionLevel(file->GetCompressionLevel());
      }
      nextb.Reset();
   }

// May be reset branches basket size?
   if (basketsize > 1000) {
      while ((branch = (TBranch*)nextb())) {
         branch->SetBasketSize(basketsize);
      }
      nextb.Reset();
   }

   Int_t treeNumber = -1;
   Int_t nentries = Int_t(GetEntries());
   for (Int_t i=0;i<nentries;i++) {
      GetEntry(i);
      if (treeNumber != fTreeNumber) {
         treeNumber = fTreeNumber;
         TIter next(fTree->GetListOfBranches());
	 Bool_t failed = kFALSE;
         while ((branch = (TBranch*)next())) {
            void *add = branch->GetAddress();
            // in case branch addresses have not been set, give a last chance
            // for simple Trees (h2root converted for example)
            if (!add) {
	       TLeaf *leaf, *new_leaf;
	       TBranch *new_branch = hnew->GetBranch( branch->GetName() );
	       if (!new_branch) continue;
               TIter next_l(branch->GetListOfLeaves());
	       while ((leaf = (TLeaf*) next_l())) {
		 add = leaf->GetValuePointer();
		 if (add) {
		   new_leaf = new_branch->GetLeaf(leaf->GetName());
		   if(new_leaf) new_leaf->SetAddress(add);
		 } else {
		   failed = kTRUE;
		 }
	       }
            } else {
               hnew->SetBranchAddress(branch->GetName(),add);
	    }
            if (failed) Warning("Merge","Tree branch addresses not defined");
         }
      }
      hnew->Fill();
   }

// Write new tree header
   hnew->Write();
}


//______________________________________________________________________________
void TChain::Print(Option_t *option)
{
   fFiles->ls(option);
}

//______________________________________________________________________________
Int_t TChain::Process(const char *filename,Option_t *option,  Int_t nentries, Int_t firstentry)
{
   // Process all entries in this chain, calling functions in filename
   // see TTree::Process

   if (LoadTree(firstentry) < 0) return 0;
   return TTree::Process(filename,option,nentries,firstentry);
}

//______________________________________________________________________________
Int_t TChain::Process(TSelector *selector,Option_t *option,  Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Process this chain executing the code in selector*-*-*-*-*
//*-*              ================================================

   return TTree::Process(selector,option,nentries,firstentry);
}

//_______________________________________________________________________
void TChain::SetBranchAddress(const char *bname, void *add)
{
//*-*-*-*-*-*-*-*-*Set branch address*-*-*-*-*-*-*-*
//*-*              ==================
//
//      bname is the name of a branch.
//      add is the address of the branch.

   //Check if bname is already in the Status list
   //Otherwise create a TChainElement object and set its address
   TChainElement *element = (TChainElement*)fStatus->FindObject(bname);
   if (!element) {
      element = new TChainElement(bname,"");
      fStatus->Add(element);
   }

   element->SetBaddress(add);

   // invalidate current Tree
   fTreeNumber = -1;
}

//_______________________________________________________________________
void TChain::SetBranchStatus(const char *bname, Bool_t status)
{
//*-*-*-*-*-*-*-*-*Set branch status Process or DoNotProcess*-*-*-*-*-*-*-*
//*-*              =========================================
//
//      bname is the name of a branch. if bname="*", apply to all branches.
//      status = 1  branch will be processed
//             = 0  branch will not be processed

   //Check if bname is already in the Status list
   //Otherwise create a TChainElement object and set its status
   TChainElement *element = (TChainElement*)fStatus->FindObject(bname);
   if (!element) {
      element = new TChainElement(bname,"");
      fStatus->Add(element);
   }

   element->SetStatus(status);

   // invalidate current Tree
   fTreeNumber = -1;
}

//_______________________________________________________________________
void TChain::SetPacketSize(Int_t size)
{
//*-*-*-*-*-*-*-*-*Set number of entries per packet for parallel root*-*-*-*-*
//*-*              =================================================

   fPacketSize = size;
   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      element->SetPacketSize(size);
   }
}

//_______________________________________________________________________
void TChain::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TChain::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TTree::Streamer(b);
      b >> fTreeOffsetLen;
      b >> fNtrees;
      fFiles->Streamer(b);
      if (R__v > 1) {
         fStatus->Streamer(b);
         fTreeOffset = new Int_t[fTreeOffsetLen];
         b.ReadFastArray(fTreeOffset,fTreeOffsetLen);
      }
      b.CheckByteCount(R__s, R__c, TChain::IsA());
      //====end of old versions
      
   } else {
      TChain::Class()->WriteBuffer(b,this);
   }
}
