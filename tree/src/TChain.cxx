// @(#)root/tree:$Name:  $:$Id: TChain.cxx,v 1.83 2003/09/12 15:54:16 rdm Exp $
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
#include "TCut.h"
#include "TFile.h"
#include "TSelector.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBrowser.h"
#include "TChainElement.h"
#include "TFriendElement.h"
#include "TSystem.h"
#include "TRegexp.h"
#include "TObjString.h"

ClassImp(TChain)

//______________________________________________________________________________
TChain::TChain(): TTree()
{
//*-*-*-*-*-*Default constructor for Chain*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==============================

   fTreeOffsetLen  = 100;
   fNtrees         = 0;
   fTreeNumber     = -1;
   fTreeOffset     = new Int_t[fTreeOffsetLen];
   fTree           = 0;
   fFile           = 0;
   fFiles          = new TObjArray(fTreeOffsetLen );
   fStatus         = new TList();
   fMaxCacheSize   = 0;
   fPageSize       = 0;
   fCanDeleteRefs  = kFALSE;
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
//   In case the Tree is in a subdirectory, do, eg:
//     TChain ch("subdir/treename");
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
   fMaxCacheSize   = 0;
   fPageSize       = 0;
   fTreeOffset[0]  = 0;
   fCanDeleteRefs  = kFALSE;
   gDirectory->GetList()->Remove(this);
   gROOT->GetListOfSpecials()->Add(this);
   fDirectory = 0;
}

//______________________________________________________________________________
TChain::~TChain()
{
// destructor for a Chain

   fDirectory = 0;
   delete fFile; fFile = 0; fTree = 0;
   gROOT->GetListOfSpecials()->Remove(this);
   delete [] fTreeOffset;
   fFiles->Delete();
   delete fFiles;
   fStatus->Delete();
   delete fStatus;
}


//______________________________________________________________________________
Int_t TChain::Add(TChain *chain)
{
// Add all files referenced by the TChain chain to this chain.

   //Check enough space in fTreeOffset
   if (fNtrees+chain->GetNtrees() >= fTreeOffsetLen) {
      fTreeOffsetLen += 2*chain->GetNtrees();
      Int_t *trees = new Int_t[fTreeOffsetLen];
      for (Int_t i=0;i<=fNtrees;i++) trees[i] = fTreeOffset[i];
      delete [] fTreeOffset;
      fTreeOffset = trees;
   }

   TIter next(chain->GetListOfFiles());
   TChainElement *element, *newelement;
   Int_t nf = 0;
   while ((element = (TChainElement*)next())) {
      Int_t nentries = element->GetEntries();
      fTreeOffset[fNtrees+1] = fTreeOffset[fNtrees] + nentries;
      fNtrees++;
      fEntries += nentries;
      newelement = new TChainElement(element->GetName(),element->GetTitle());
      newelement->SetPacketSize(element->GetPacketSize());
      newelement->SetNumberEntries(nentries);
      fFiles->Add(newelement);
      nf++;
   }
   return nf;
}

//______________________________________________________________________________
Int_t TChain::Add(const char *name, Int_t nentries)
{
// Add a new file to this chain.
// Argument name may have the following format:
//   //machine/file_name.root/subdir/tree_name
// machine, subdir and tree_name are optional. If tree_name is missing,
// the chain name will be assumed.
// Name may use the wildcarding notation, eg "xxx*.root" means all files
// starting with xxx in the current file system directory.
// NB. To add all the files of a TChain to a chain, use Add(TChain *chain).
//
//    A- if nentries <= 0, the file is connected and the tree header read
//       in memory to get the number of entries.
//
//    B- if (nentries > 0, the file is not connected, nentries is assumed to be
//       the number of entries in the file. In this case, no check is made that
//       the file exists and the Tree existing in the file. This second mode
//       is interesting in case the number of entries in the file is already stored
//       in a run data base for example.
//
//    C- if (nentries == kBigNumber) (default), the file is not connected.
//       the number of entries in each file will be read only when the file
//       will need to be connected to read an entry.
//       This option is the default and very efficient if one process
//       the chain sequentially. Note that in case TChain::GetEntry(entry)
//       is called and entry refers to an entry in the 3rd file, for example,
//       this forces the Tree headers in the first and second file
//       to be read to find the number of entries in these files.
//       Note that if one calls TChain::GetEntriesFast() after having created
//       a chain with this default, GetEntriesFast will return kBigNumber!
//       TChain::GetEntries will force of the Tree headers in the chain to be
//       read to read the number of entries in each Tree.

   // case with one single file
   if (!TString(name).MaybeWildcard()) {
      return AddFile(name,nentries);
   }

   // wildcarding used in name
   Int_t nf = 0;
   TString basename(name);

   Int_t dotslashpos = basename.Index(".root/");
   TString behind_dot_root;
   if (dotslashpos>=0) {
      // Copy the tree name specification
      behind_dot_root = basename(dotslashpos+6,basename.Length()-dotslashpos+6);
      // and remove it from basename
      basename.Remove(dotslashpos+5);
   }

   Int_t slashpos = basename.Last('/');
   TString directory;
   if (slashpos>=0) {
      directory = basename(0,slashpos); // Copy the directory name
      basename.Remove(0,slashpos+1);      // and remove it from basename
   } else {
      directory = gSystem->WorkingDirectory();
   }

   const char *file;
   void *dir = gSystem->OpenDirectory(gSystem->ExpandPathName(directory.Data()));

   if (dir) {
      //create a TList to store the file names (not yet sorted)
      TList l;
      TRegexp re(basename,kTRUE);
      while ((file = gSystem->GetDirEntry(dir))) {
         if (!strcmp(file,".") || !strcmp(file,"..")) continue;
         TString s = file;
         if ( (basename!=file) && s.Index(re) == kNPOS) continue;
         l.Add(new TObjString(file));
      }
      gSystem->FreeDirectory(dir);
      //sort the files in alphanumeric order
      l.Sort();
      TIter next(&l);
      TObjString *obj;
      while ((obj = (TObjString*)next())) {
         file = obj->GetName();
         if (behind_dot_root.Length() != 0)
            nf += AddFile(Form("%s/%s/%s",directory.Data(),file,behind_dot_root.Data()),kBigNumber);
         else
            nf += AddFile(Form("%s/%s",directory.Data(),file),kBigNumber);
      }
      l.Delete();
   }
   return nf;
}

//______________________________________________________________________________
Int_t TChain::AddFile(const char *name, Int_t nentries)
{
//       Add a new file to this chain.
//
//    A- if nentries <= 0, the file is connected and the tree header read
//       in memory to get the number of entries.
//
//    B- if (nentries > 0, the file is not connected, nentries is assumed to be
//       the number of entries in the file. In this case, no check is made that
//       the file exists and the Tree existing in the file. This second mode
//       is interesting in case the number of entries in the file is already stored
//       in a run data base for example.
//
//    C- if (nentries == kBigNumber) (default), the file is not connected.
//       the number of entries in each file will be read only when the file
//       will need to be connected to read an entry.
//       This option is the default and very efficient if one process
//       the chain sequentially. Note that in case TChain::GetEntry(entry)
//       is called and entry refers to an entry in the 3rd file, for example,
//       this forces the Tree headers in the first and second file
//       to be read to find the number of entries in these files.
//       Note that if one calls TChain::GetEntriesFast() after having created
//       a chain with this default, GetEntriesFast will return kBigNumber!
//       TChain::GetEntries will force of the Tree headers in the chain to be
//       read to read the number of entries in each Tree.

   TDirectory *cursav = gDirectory;
   char *treename = (char*)GetName();
   char *dot = (char*)strstr(name,".root");
   //the ".root" is mandatory only if one wants to specify a treename
   //if (!dot) {
   //   Error("AddFile","a chain element name must contain the string .root");
   //   return 0;
   //}

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
   if (dot) {
      char *pos = (char*)strstr(filename,".root") + 5;
      while (*pos) {
         if (*pos == '/') {
            treename = pos+1;
            *pos = 0;
            break;
         }
         pos++;
      }
   }

   //Connect the file to get the number of entries
   Int_t pksize = 0;
   if (nentries <= 0) {
      TFile *file = TFile::Open(filename);
      if (file->IsZombie()) {
         delete file;
         delete [] filename;
         return 0;
      }

      //Check that tree with the right name exists in the file
      TObject *obj = file->Get(treename);
      if (!obj || !obj->InheritsFrom("TTree") ) {
         Error("AddFile","cannot find tree with name %s in file %s", treename,filename);
         delete file;
         delete [] filename;
         return 0;
      }
      TTree *tree = (TTree*)obj;
      nentries = (Int_t)tree->GetEntries();
      pksize   = tree->GetPacketSize();
      delete file;
   }

   if (nentries > 0) {
      if (nentries < kBigNumber) {
         fTreeOffset[fNtrees+1] = fTreeOffset[fNtrees] + nentries;
         fEntries += nentries;
      } else {
         fTreeOffset[fNtrees+1] = kBigNumber;
         fEntries = nentries;
      }
      fNtrees++;

      TChainElement *element = new TChainElement(treename,filename);
      element->SetPacketSize(pksize);
      element->SetNumberEntries(nentries);
      fFiles->Add(element);
   } else {
      Warning("Add","Adding Tree with no entries from file: %s",filename);
   }

   delete [] filename;
   if (cursav) cursav->cd();
   return 1;
}

//______________________________________________________________________________
TFriendElement *TChain::AddFriend(const char *chain, const char *dummy)
{
// Add a TFriendElement to the list of friends of this chain.
//
//   A TChain has a list of friends similar to a tree (see TTree::AddFriend).
// You can add a friend to a chain with the TChain::AddFriend method, and you
// can retrieve the list of friends with TChain::GetListOfFriends.
// This example has four chains each has 20 ROOT trees from 20 ROOT files.
//
// TChain ch("t"); // a chain with 20 trees from 20 files
// TChain ch1("t1");
// TChain ch2("t2");
// TChain ch3("t3");
// Now we can add the friends to the first chain.
//
// ch.AddFriend("t1")
// ch.AddFriend("t2")
// ch.AddFriend("t3")
//
//Begin_Html
/*
<img src="gif/chain_friend.gif">
*/
//End_Html
//
// The parameter is the name of friend chain (the name of a chain is always
// the name of the tree from which it was created).
// The original chain has access to all variable in its friends.
// We can use the TChain::Draw method as if the values in the friends were
// in the original chain.
// To specify the chain to use in the Draw method, use the syntax:
//
// <chainname>.<branchname>.<varname>
// If the variable name is enough to uniquely identify the variable, you can
// leave out the chain and/or branch name.
// For example, this generates a 3-d scatter plot of variable "var" in the
// TChain ch versus variable v1 in TChain t1 versus variable v2 in TChain t2.
//
// ch.Draw("var:t1.v1:t2.v2");
// When a TChain::Draw is executed, an automatic call to TTree::AddFriend
// connects the trees in the chain. When a chain is deleted, its friend
// elements are also deleted.
//
// The number of entries in the friend must be equal or greater to the number
// of entries of the original chain. If the friend has fewer entries a warning
// is given and the resulting histogram will have missing entries.
// For additional information see TTree::AddFriend.

   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,chain,dummy);
   if (fe) {
      fFriends->Add(fe);

      // We need to invalidate the loading of the current because if list
      // of real friend is now obsolete.  It is repairable only from LoadTree
      fTreeNumber = -1;

      TTree *t = fe->GetTree();
      if (t) {
         if (t->GetEntries() < fEntries) {
            //Warning("AddFriend","FriendElement %s in file %s has less entries %g than its parent Tree: %g",
            //         chain,filename,t->GetEntries(),fEntries);
         }
      } else {
         Warning("AddFriend","Unknown TChain %s",chain);
      }
   } else {
      Warning("AddFriend","Cannot add FriendElement %s",chain);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement *TChain::AddFriend(const char *chain, TFile *dummy)
{
   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,chain,dummy);
   if (fe) {
      fFriends->Add(fe);

      // We need to invalidate the loading of the current because if list
      // of real friend is now obsolete.  It is repairable only from LoadTree
      fTreeNumber = -1;

      TTree *t = fe->GetTree();
      if (t) {
         if (t->GetEntries() < fEntries) {
            //Warning("AddFriend","FriendElement %s in file %s has less entries %g than its parent Tree: %g",
            //         chain,filename,t->GetEntries(),fEntries);
         }
      } else {
         Warning("AddFriend","Unknown TChain %s",chain);
      }
   } else {
      Warning("AddFriend","Cannot add FriendElement %s",chain);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement *TChain::AddFriend(TTree *chain, const char* alias,
                                  Bool_t /*warn*/)
{
   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,chain,alias);
   if (fe) {
      fFriends->Add(fe);

      // We need to invalidate the loading of the current because if list
      // of real friend is now obsolete.  It is repairable only from LoadTree
      fTreeNumber = -1;

      TTree *t = fe->GetTree();
      if (t) {
         if (t->GetEntries() < fEntries) {
            //Warning("AddFriend","FriendElement %s in file %s has less entries %g than its parent Tree: %g",
            //         chain,filename,t->GetEntries(),fEntries);
         }
      } else {
         Warning("AddFriend","Unknown TChain %s",chain->GetName());
      }
   } else {
      Warning("AddFriend","Cannot add FriendElement %s",chain->GetName());
   }
   return fe;
}

//______________________________________________________________________________
void TChain::Browse(TBrowser *)
{

}

//_______________________________________________________________________
void TChain::CanDeleteRefs(Bool_t flag)
{
// when closing a file during the chain processing, the file
// may be closed with option "R" if flag is set to kTRUE.
// by default flag is kTRUE.
// When closing a file with option "R", all TProcessIDs referenced by this
// file are deleted.
// Calling TFile::Close("R") might be necessary in case one reads a long list
// of files having TRef, writing some of the referenced objects or TRef
// to a new file. If the TRef or referenced objects of the file being closed
// will not be referenced again, it is possible to minimize the size
// of the TProcessID data structures in memory by forcing a delete of
// the unused TProcessID.

   fCanDeleteRefs = flag;
}

//_______________________________________________________________________
void TChain::CreatePackets()
{
// Initialize the packet descriptor string

   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      element->CreatePackets();
   }
}

//______________________________________________________________________________
Int_t TChain::Draw(const char *varexp, const TCut &selection, Option_t *option, Int_t nentries, Int_t firstentry)
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
// Return pointer to the branch name in the current tree

   if (fTree) return fTree->GetBranch(name);
   LoadTree(0);
   if (fTree) return fTree->GetBranch(name);
   return 0;
}

//______________________________________________________________________________
Int_t TChain::GetChainEntryNumber(Int_t entry) const
{
// return absolute entry number in the chain
// the input parameter entry is the entry number in the current Tree of this chain

  return entry + fTreeOffset[fTreeNumber];
}

//______________________________________________________________________________
Double_t TChain::GetEntries() const
{
// return the total number of entries in the chain.
// In case the number of entries in each tree is not yet known,
// the offset table is computed

   if (fEntries >= (Stat_t)kBigNumber) {
      ((TChain*)this)->LoadTree(Int_t(fEntries)-1);
   }
   return fEntries;
}

//______________________________________________________________________________
Int_t TChain::GetEntry(Int_t entry, Int_t getall)
{
// Get entry from the file to memory
//
//     getall = 0 : get only active branches
//     getall = 1 : get all branches
//
// return the total number of bytes read

   if (LoadTree(entry) < 0) return 0;
   return fTree->GetEntry(fReadEntry,getall);
}


//______________________________________________________________________________
TFile *TChain::GetFile() const
{
//  Return a pointer to the current file.
//  if no file is connected, the first file is automatically loaded.

   if (fFile) return fFile;
   ((TChain*)this)->LoadTree(0); //force reading first entry
   return fFile;
}


//______________________________________________________________________________
TLeaf *TChain::GetLeaf(const char *name)
{
//  Return pointer to the leaf name in the current tree

   if (fTree) return fTree->GetLeaf(name);
   LoadTree(0);
   if (fTree) return fTree->GetLeaf(name);
   return 0;
}


//______________________________________________________________________________
TObjArray *TChain::GetListOfBranches()
{
// Return pointer to list of branches of current tree

   if (fTree) return fTree->GetListOfBranches();
   LoadTree(0);
   if (fTree) return fTree->GetListOfBranches();
   return 0;
}


//______________________________________________________________________________
TObjArray *TChain::GetListOfLeaves()
{
// Return pointer to list of leaves of current tree

   if (fTree) return fTree->GetListOfLeaves();
   LoadTree(0);
   if (fTree) return fTree->GetListOfLeaves();
   return 0;
}

//______________________________________________________________________________
Double_t TChain::GetMaximum(const char *columname)
{
// Return maximum of column with name columname

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
// Return minimum of column with name columname

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
// Return number of branches of current tree

   if (fTree) return fTree->GetNbranches();
   LoadTree(0);
   if (fTree) return fTree->GetNbranches();
   return 0;
}

//______________________________________________________________________________
const char *TChain::GetAlias(const char *aliasName) const
{
   // Returns the expanded value of the alias.  Search in the friend if any

   const char *alias = TTree::GetAlias(aliasName);
   if (alias) return alias;

   if (fTree) return fTree->GetAlias(aliasName);
   const_cast<TChain*>(this)->LoadTree(0);
   if (fTree) return fTree->GetAlias(aliasName);
   return 0;
}

//______________________________________________________________________________
Double_t TChain::GetWeight() const
{
//  return the chain weight.
//  by default, the weight is the weight of the current Tree in the TChain.
//  However, if the weight has been set in TChain::SetWeight with
//  the option "global", each Tree will use the same weight stored
//  in TChain::fWeight.

   if (TestBit(kGlobalWeight)) return fWeight;
   else {
      if (fTree) return fTree->GetWeight();
      ((TChain*)this)->LoadTree(0);
      if (fTree) return fTree->GetWeight();
      return 0;
   }
}

//______________________________________________________________________________
Int_t TChain::LoadTree(Int_t entry)
{
//  The input argument entry is the entry serial number in the whole chain.
//  The function finds the corresponding Tree and returns the entry number
//  in this tree.

   if (!fNtrees) return 1;
   if (entry < 0 || entry >= fEntries) return -2;

   // Find in which tree this entry belongs to
   Int_t t;
   if (fTreeNumber!=-1 &&
       (entry >= fTreeOffset[fTreeNumber] && entry < fTreeOffset[fTreeNumber+1])){
      t = fTreeNumber;
   }
   else {
      for (t=0;t<fNtrees;t++) {
         if (entry < fTreeOffset[t+1]) break;
      }
   }

   fReadEntry = entry - fTreeOffset[t];
   // If entry belongs to the current tree return entry
   if (t == fTreeNumber) {
      // This need to be done first because it will set the friend tree's
      // fReadEntry to the current one (which is possibly wrong).  The
      // call to t->LoadTree inside the chain would fix that.
      fTree->LoadTree(fReadEntry);
      if (fFriends) {
         // The current tree has not changed but some of its friend might.

         //An Alternative would move this code to each of the function calling LoadTree
         //(and to overload a few more).
         TIter next(fFriends);
         TFriendElement *fe;
         Bool_t needUpdate = kFALSE;
         while ((fe = (TFriendElement*)next())) {
            TTree *t = fe->GetTree();
            if (t->InheritsFrom(TChain::Class())) {
               Int_t oldNumber = ((TChain*)t)->GetTreeNumber();
               TTree* old = t->GetTree();

               t->LoadTree(entry);

               Int_t newNumber = ((TChain*)t)->GetTreeNumber();
               if (oldNumber!=newNumber) {
                  // We can not compare the tree pointers because they could be reused.
                  // so we compare the tree number instead.
                  needUpdate = kTRUE;
                  fTree->RemoveFriend(old);
                  fTree->AddFriend(t->GetTree(),fe->GetName());
               }
            } else {
               // else we assume it is a simple tree
               // If the tree is a direct friend of the chain, it should be scanned
               // used the chain entry number and NOT the tree entry number (fReadEntry)
               // hence we redo:
               t->LoadTree(entry);
            }
         }

         if (needUpdate) {
            //update list of leaves in all TTreeFormula of the TTreePlayer (if any)
            if (fPlayer) fPlayer->UpdateFormulaLeaves();
            //Notify user if requested
            if (fNotify) fNotify->Notify();
         }

      }
      return fReadEntry;
   }

   // If the tree has some clone, let migrate them into the chain so we can
   // continue to keep track of it.  This is to support the syntax:
   //    clone = (TTree*)chain->GetTree()->CloneTree(0)
   if (fTree && fTree->GetListOfClones()) {
      TObjLink *lnk = fTree->GetListOfClones()->FirstLink();
      while (lnk) {
         TTree *clone = (TTree*)lnk->GetObject();
         AddClone(clone);
         lnk = lnk->Next();
      }
      fTree->GetListOfClones()->Clear();
   }

   //Delete current tree and connect new tree
   TDirectory *cursav = gDirectory;
   //delete file unless the file owns this chain !!
   if (fFile) {
      if (!fDirectory->GetList()->FindObject(this)) {
         if (cursav && cursav->GetFile()==fFile) {
            cursav = gROOT;
         }
         if (fCanDeleteRefs) fFile->Close("R");
         delete fFile; fFile = 0; fTree = 0;
      }
   }
   TChainElement *element = (TChainElement*)fFiles->At(t);
   if (!element) return -4;
   fFile = TFile::Open(element->GetTitle());
   if (fFile==0) return -3;
   if (fFile->IsZombie()) {
      delete fFile; fFile = 0;
      return -3;
   }

   if (fMaxCacheSize > 0)
      fFile->UseCache(fMaxCacheSize, fPageSize);

   fTree = (TTree*)fFile->Get(element->GetName());
   if (fTree==0) {
      // Now that we do not check during the addition, we need to check here!
      Error("LoadTree","cannot find tree with name %s in file %s",
            element->GetName(),element->GetTitle());
      delete fFile; fFile = 0;
      return -4;
   }
   fTreeNumber = t;
   fDirectory = fFile;

   //check if fTreeOffset has really been set
   Int_t nentries = (Int_t)fTree->GetEntries();
   if (fTreeOffset[fTreeNumber+1] != fTreeOffset[fTreeNumber] + nentries) {
      fTreeOffset[fTreeNumber+1] = fTreeOffset[fTreeNumber] + nentries;
      fEntries = fTreeOffset[fNtrees];
      element->SetNumberEntries(nentries);
      //below, we must test >= in case the Tree has no entries
      if (entry >= fTreeOffset[fTreeNumber+1]) {
         cursav->cd();
         if (fTreeNumber < fNtrees-1 && entry < fTreeOffset[fTreeNumber+2]) return LoadTree(entry);
         else  fReadEntry = -2;
      }
   }

   // Since some of the friend of this chain might a simple tree (i.e. not a chain),
   // we need to execute this before the calling LoadTree(entry) on the friend (so
   // that those tree use the correct read entry number!
   fTree->LoadTree(fReadEntry);

   if (fFriends) {
      //An Alternative would move this code to each of the function calling LoadTree
      //(and to overload a few more).
      TIter next(fFriends);
      TFriendElement *fe;
      while ((fe = (TFriendElement*)next())) {
         TTree *t = fe->GetTree();
         t->LoadTree(entry);
         TTree *friend_t = t->GetTree();
         if (friend_t) fTree->AddFriend(friend_t,fe->GetName());
      }
   }

   //Set the branches status and address for the newly connected file
   fTree->SetMakeClass(fMakeClass);
   fTree->SetMaxVirtualSize(fMaxVirtualSize);
   SetChainOffset(fTreeOffset[t]);
   TIter next(fStatus);
   Int_t status;
   while ((element = (TChainElement*)next())) {
      status = element->GetStatus();
      fTree->SetBranchStatus(element->GetName(),status);
   }
   next.Reset();
   while ((element = (TChainElement*)next())) {
      void *add = element->GetBaddress();
      if (add) {
         TBranch *br = fTree->GetBranch(element->GetName());
         if (br) {
            br->SetAddress(add);
            if (TestBit(kAutoDelete)) br->SetAutoDelete(kTRUE);
         }
      }
   }

   if (cursav) cursav->cd();

   //update the address of cloned trees, if any
   if (fClones) {
      TObjLink *lnk = fClones->FirstLink();
      while (lnk) {
         TTree *clone = (TTree*)lnk->GetObject();
         CopyAddresses(clone);
         lnk = lnk->Next();
      }
   }

   //update list of leaves in all TTreeFormula of the TTreePlayer (if any)
   if (fPlayer) fPlayer->UpdateFormulaLeaves();

   //Notify user if requested
   if (fNotify) fNotify->Notify();

   return fReadEntry;
}

//______________________________________________________________________________
void TChain::Loop(Option_t *option, Int_t nentries, Int_t firstentry)
{
// Loop on nentries of this chain starting at firstentry

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
void TChain::ls(Option_t *option) const
{
   TIter next(fFiles);
   TChainElement *file;
   while ((file = (TChainElement*)next())) {
      file->ls(option);
   }
}

//______________________________________________________________________________
Int_t TChain::Merge(const char *name)
{
   // Merge all files in this chain into a new file.
   // See important note in the following function Merge().

   TFile *file = TFile::Open(name,"recreate","chain files",1);
   return Merge(file,0,"");
}


//______________________________________________________________________________
Int_t TChain::Merge(TFile *file, Int_t basketsize, Option_t *option)
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
//  example using the file generated in $ROOTSYS/test/Event
//  merge two copies of Event.root
//
//        gSystem.Load("libEvent");
//        TChain ch("T");
//        ch.Add("Event1.root");
//        ch.Add("Event2.root");
//        ch.Merge("all.root");
//
// IMPORTANT Note 1: AUTOMATIC FILE OVERFLOW
// -----------------------------------------
// When merging many files, it may happen that the resulting file
// reaches a size > TTree::fgMaxTreeSize (default = 1.9 GBytes). In this case
// the current file is automatically closed and a new file started.
// If the name of the merged file was "merged.root", the subsequent files
// will be named "merged_1.root", "merged_2.root", etc.
// fgMaxTreeSize may be modified via the static function TTree::SetMaxTreeSize.
//
// IMPORTANT Note 2: The input file is automatically closed and deleted.
// This is required because in general the automatic file overflow described
// above may happen during the merge.
//
// The function returns the total number of files produced.

   if (!file) return 0;
   TObjArray *lbranches = GetListOfBranches();
   if (!lbranches) return 0;
   if (!fTree) return 0;

// Clone Chain tree
   //file->cd();  //in case a user wants to write in a file/subdir
   TTree *hnew = CloneTree(0);
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

   char *firstname = new char[1000];
   firstname[0] = 0;
   strcpy(firstname,gFile->GetName());

   Int_t nentries = Int_t(GetEntriesFast());
   for (Int_t i=0;i<nentries;i++) {
      if (GetEntry(i) <= 0) break;
      hnew->Fill();
   }

// Write new tree header
   hnew->Write();
   Int_t nfiles = hnew->GetFileNumber()+1;
   delete [] firstname;
   delete hnew->GetCurrentFile();
   return nfiles;
}


//______________________________________________________________________________
void TChain::Print(Option_t *option) const
{
   // Print the header information of each Tree in the chain.
   // see TTree::Print for a list of options

   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      TFile *file = TFile::Open(element->GetTitle());
      if (!file->IsZombie()) {
         TTree *tree = (TTree*)file->Get(element->GetName());
         if (tree) tree->Print(option);
      }
      delete file;
   }
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
// Process this chain executing the code in selector

   return TTree::Process(selector,option,nentries,firstentry);
}

//______________________________________________________________________________
void TChain::Reset(Option_t *)
{
// Resets the definition of this chain

   delete fFile;
   fNtrees         = 0;
   fTreeNumber     = -1;
   fTree           = 0;
   fFile           = 0;
   fFiles->Delete();
   fStatus->Delete();
   fTreeOffset[0]  = 0;
   TChainElement *element = new TChainElement("*","");
   fStatus->Add(element);
   fDirectory = 0;

   TTree::Reset();
}

//_______________________________________________________________________
void TChain::SetAutoDelete(Bool_t autodelete)
{
//  Set the global branch kAutoDelete bit
//  When LoadTree loads a new Tree, the branches for which
//  the address is set will have the option AutoDelete set
//  For more details on AutoDelete, see TBranch::SetAutoDelete.

   if (autodelete) SetBit(kAutoDelete,1);
   else            SetBit(kAutoDelete,0);
}

//_______________________________________________________________________
void TChain::SetBranchAddress(const char *bname, void *add)
{
// Set branch address
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

   // Set also address in current Tree
   if (fTreeNumber >= 0) {
       TBranch *branch = fTree->GetBranch(bname);
       if (branch) {
          if (fClones) {
             void *oldAdd = branch->GetAddress();
             TObjLink *lnk = fClones->FirstLink();
             while (lnk) {
                TTree *clone = (TTree*)lnk->GetObject();
                TBranch *cloneBr = clone->GetBranch(bname);
                if (cloneBr && cloneBr->GetAddress() == oldAdd ) {
                   // the clone's branch is still pointing to us
                   cloneBr->SetAddress(add);
                }
                lnk = lnk->Next();
             } // while(lnk)
          } // if (fClones)
          branch->SetAddress(add);
       }
   }
}

//_______________________________________________________________________
void TChain::SetBranchStatus(const char *bname, Bool_t status, UInt_t *found)
{
// Set branch status Process or DoNotProcess
//
//      bname is the name of a branch. if bname="*", apply to all branches.
//      status = 1  branch will be processed
//             = 0  branch will not be processed
//  See IMPORTANT REMARKS in TTree::SetBranchStatus
//
//  If found is not 0, the number of branch(es) found matching the regular
//  expression is returned in *found AND the error message 'unknown branch'
//  is suppressed.

   //Check if bname is already in the Status list
   //Otherwise create a TChainElement object and set its status
   TChainElement *element = (TChainElement*)fStatus->FindObject(bname);
   if (element)
      fStatus->Remove (element);
   else
      element = new TChainElement(bname,"");
   fStatus->Add(element);

   element->SetStatus(status);

   // Set also status in current Tree
   if (fTreeNumber >= 0) {
       fTree->SetBranchStatus(bname,status,found);
   } else if (found) {
      *found = 1;
   }
}

//______________________________________________________________________________
void TChain::SetDirectory(TDirectory *dir)
{
   // Remove reference to this chain from current directory and add
   // reference to new directory dir. dir can be 0 in which case the chain
   // does not belong to any directory.

   if (fDirectory == dir) return;
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fDirectory = dir;
   if (fDirectory) {
      fDirectory->GetList()->Add(this);
      fFile = fDirectory->GetFile();
   } else {
      fFile = 0;
   }
}

//_______________________________________________________________________
void TChain::SetPacketSize(Int_t size)
{
// Set number of entries per packet for parallel root

   fPacketSize = size;
   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      element->SetPacketSize(size);
   }
}

//______________________________________________________________________________
void TChain::SetWeight(Double_t w, Option_t *option)
{
//  Set chain weight.
//  The weight is used by TTree::Draw to automatically weight each
//  selected entry in the resulting histogram.
//  For example the equivalent of
//     chain.Draw("x","w")
//  is
//     chain.SetWeight(w,"global");
//     chain.Draw("x");
//
//  By default the weight used will be the weight
//  of each Tree in the TChain. However, one can force the individual
//  weights to be ignored by specifying the option "global".
//  In this case, the TChain global weight will be used for all Trees.

   fWeight = w;
   TString opt = option;
   opt.ToLower();
   ResetBit(kGlobalWeight);
   if (opt.Contains("global")) {
      SetBit(kGlobalWeight);
   }
}

//______________________________________________________________________________
void TChain::Streamer(TBuffer &b)
{
// Stream a class object

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

//______________________________________________________________________________
void TChain::UseCache(Int_t maxCacheSize, Int_t pageSize)
{
   // Activate file caching. Use maxCacheSize to specify the maximum cache size
   // in MB's (default is 10 MB) and pageSize to specify the page size
   // (default is 512 KB). To turn off the cache use maxCacheSize=0.
   // Not needed for normal disk files since the operating system will
   // do proper caching (via the "buffer cache"). Use it for TNetFile,
   // TWebFile, TRFIOFile, TDCacheFile, etc.

   fMaxCacheSize = maxCacheSize;
   fPageSize     = pageSize;
}
