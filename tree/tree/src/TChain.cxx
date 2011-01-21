// @(#)root/tree:$Id$
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

#include "TChain.h"

#include "TBranch.h"
#include "TBrowser.h"
#include "TChainElement.h"
#include "TClass.h"
#include "TCut.h"
#include "TError.h"
#include "TMath.h"
#include "TFile.h"
#include "TFileInfo.h"
#include "TFriendElement.h"
#include "TLeaf.h"
#include "TList.h"
#include "TObjString.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TRegexp.h"
#include "TSelector.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeCloner.h"
#include "TTreeCache.h"
#include "TUrl.h"
#include "TVirtualIndex.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TEntryListFromFile.h"
#include "TFileStager.h"

const Long64_t theBigNumber = Long64_t(1234567890)<<28;

ClassImp(TChain)

//______________________________________________________________________________
TChain::TChain()
: TTree()
, fTreeOffsetLen(100)
, fNtrees(0)
, fTreeNumber(-1)
, fTreeOffset(0)
, fCanDeleteRefs(kFALSE)
, fTree(0)
, fFile(0)
, fFiles(0)
, fStatus(0)
, fProofChain(0)
{
   // -- Default constructor.

   fTreeOffset = new Long64_t[fTreeOffsetLen];
   fFiles = new TObjArray(fTreeOffsetLen);
   fStatus = new TList();
   fTreeOffset[0]  = 0;
   gDirectory->Remove(this);
   gROOT->GetListOfSpecials()->Add(this);
   fFile = 0;
   fDirectory = 0;

   // Reset PROOF-related bits
   ResetBit(kProofUptodate);
   ResetBit(kProofLite);

   // Add to the global list
   gROOT->GetListOfDataSets()->Add(this);

   // Make sure we are informed if the TFile is deleted.
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TChain::TChain(const char* name, const char* title)
:TTree(name, title)
, fTreeOffsetLen(100)
, fNtrees(0)
, fTreeNumber(-1)
, fTreeOffset(0)
, fCanDeleteRefs(kFALSE)
, fTree(0)
, fFile(0)
, fFiles(0)
, fStatus(0)
, fProofChain(0)
{
   // -- Create a chain.
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
   //   The TChain data structure
   //       Each TChainElement has a name equal to the tree name of this TChain
   //       and a title equal to the file name. So, to loop over the
   //       TFiles that have been added to this chain:
   //
   //         TObjArray *fileElements=chain->GetListOfFiles();
   //         TIter next(fileElements);
   //         TChainElement *chEl=0;
   //         while (( chEl=(TChainElement*)next() )) {
   //            TFile f(chEl->GetTitle());
   //            ... do something with f ...
   //         }

   //
   //*-*

   fTreeOffset = new Long64_t[fTreeOffsetLen];
   fFiles = new TObjArray(fTreeOffsetLen);
   fStatus = new TList();
   fTreeOffset[0]  = 0;
   gDirectory->Remove(this);
   gROOT->GetListOfSpecials()->Add(this);
   fFile = 0;
   fDirectory = 0;

   // Reset PROOF-related bits
   ResetBit(kProofUptodate);
   ResetBit(kProofLite);

   // Add to the global list
   gROOT->GetListOfDataSets()->Add(this);

   // Make sure we are informed if the TFile is deleted.
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TChain::~TChain()
{
   // -- Destructor.
   gROOT->GetListOfCleanups()->Remove(this);

   SafeDelete(fProofChain);
   fStatus->Delete();
   delete fStatus;
   fStatus = 0;
   fFiles->Delete();
   delete fFiles;
   fFiles = 0;
   delete fFile;
   fFile = 0;
   // Note: We do *not* own the tree.
   fTree = 0;
   delete[] fTreeOffset;
   fTreeOffset = 0;

   gROOT->GetListOfSpecials()->Remove(this);

   // Remove from the global list
   gROOT->GetListOfDataSets()->Remove(this);

   // This is the same as fFile, don't delete it a second time.
   fDirectory = 0;
}

//______________________________________________________________________________
Int_t TChain::Add(TChain* chain)
{
   // -- Add all files referenced by the passed chain to this chain.
   // The function returns the total number of files connected.

   if (!chain) return 0;

   // Check for enough space in fTreeOffset.
   if ((fNtrees + chain->GetNtrees()) >= fTreeOffsetLen) {
      fTreeOffsetLen += 2 * chain->GetNtrees();
      Long64_t* trees = new Long64_t[fTreeOffsetLen];
      for (Int_t i = 0; i <= fNtrees; i++) {
         trees[i] = fTreeOffset[i];
      }
      delete[] fTreeOffset;
      fTreeOffset = trees;
   }
   chain->GetEntries(); //to force the computation of nentries
   TIter next(chain->GetListOfFiles());
   Int_t nf = 0;
   TChainElement* element = 0;
   while ((element = (TChainElement*) next())) {
      Long64_t nentries = element->GetEntries();
      if (fTreeOffset[fNtrees] == theBigNumber) {
         fTreeOffset[fNtrees+1] = theBigNumber;
      } else {
         fTreeOffset[fNtrees+1] = fTreeOffset[fNtrees] + nentries;
      }
      fNtrees++;
      fEntries += nentries;
      TChainElement* newelement = new TChainElement(element->GetName(), element->GetTitle());
      newelement->SetPacketSize(element->GetPacketSize());
      newelement->SetNumberEntries(nentries);
      fFiles->Add(newelement);
      nf++;
   }
   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   return nf;
}

//______________________________________________________________________________
Int_t TChain::Add(const char* name, Long64_t nentries /* = kBigNumber */)
{
   // -- Add a new file to this chain.
   //
   // Argument name may have the following format:
   //   //machine/file_name.root/subdir/tree_name
   // machine, subdir and tree_name are optional. If tree_name is missing,
   // the chain name will be assumed.
   // In the file name part (but not in preceding directories) wildcarding
   // notation may be used, eg. specifying "xxx*.root" adds all files starting
   // with xxx in the current file system directory.
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
   //
   //
   //    D- The TChain data structure
   //       Each TChainElement has a name equal to the tree name of this TChain
   //       and a title equal to the file name. So, to loop over the
   //       TFiles that have been added to this chain:
   //
   //         TObjArray *fileElements=chain->GetListOfFiles();
   //         TIter next(fileElements);
   //         TChainElement *chEl=0;
   //         while (( chEl=(TChainElement*)next() )) {
   //            TFile f(chEl->GetTitle());
   //            ... do something with f ...
   //         }
   //
   // The function returns the total number of files connected.

   // case with one single file
   if (!TString(name).MaybeWildcard()) {
      return AddFile(name, nentries);
   }

   // wildcarding used in name
   Int_t nf = 0;
   TString basename(name);

   Int_t dotslashpos = -1;
   {
      Int_t next_dot = basename.Index(".root");
      while(next_dot>=0) {
         dotslashpos = next_dot;
         next_dot = basename.Index(".root",dotslashpos+1);
      }
      if (basename[dotslashpos+5]!='/') {
         // We found the 'last' .root in the name and it is not followed by
         // a '/', so the tree name is _not_ specified in the name.
         dotslashpos = -1;
      }
   }
   //Int_t dotslashpos = basename.Index(".root/");
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
      directory = gSystem->UnixPathName(gSystem->WorkingDirectory());
   }

   const char *file;
   const char *epath = gSystem->ExpandPathName(directory.Data());
   void *dir = gSystem->OpenDirectory(epath);
   delete [] epath;
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
            nf += AddFile(Form("%s/%s/%s",directory.Data(),file,behind_dot_root.Data()),nentries);
         else
            nf += AddFile(Form("%s/%s",directory.Data(),file),nentries);
      }
      l.Delete();
   }
   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   return nf;
}

//______________________________________________________________________________
Int_t TChain::AddFile(const char* name, Long64_t nentries /* = kBigNumber */, const char* tname /* = "" */)
{
   // -- Add a new file to this chain.
   //
   //    If tname is specified, the chain will load the tree named tname
   //    from the file, otherwise the original treename specified in the
   //    TChain constructor will be used.
   //
   // A. If nentries <= 0, the file is opened and the tree header read
   //    into memory to get the number of entries.
   //
   // B. If nentries > 0, the file is not opened, and nentries is assumed
   //    to be the number of entries in the file. In this case, no check
   //    is made that the file exists nor that the tree exists in the file.
   //    This second mode is interesting in case the number of entries in
   //    the file is already stored in a run database for example.
   //
   // C. If nentries == kBigNumber (default), the file is not opened.
   //    The number of entries in each file will be read only when the file
   //    is opened to read an entry.  This option is the default and very
   //    efficient if one processes the chain sequentially.  Note that in
   //    case GetEntry(entry) is called and entry refers to an entry in the
   //    third file, for example, this forces the tree headers in the first
   //    and second file to be read to find the number of entries in those
   //    files.  Note that if one calls GetEntriesFast() after having created
   //    a chain with this default, GetEntriesFast() will return kBigNumber!
   //    Using the GetEntries() function instead will force all of the tree
   //    headers in the chain to be read to read the number of entries in
   //    each tree.
   //
   // D. The TChain data structure
   //    Each TChainElement has a name equal to the tree name of this TChain
   //    and a title equal to the file name. So, to loop over the
   //    TFiles that have been added to this chain:
   //
   //      TObjArray *fileElements=chain->GetListOfFiles();
   //      TIter next(fileElements);
   //      TChainElement *chEl=0;
   //      while (( chEl=(TChainElement*)next() )) {
   //         TFile f(chEl->GetTitle());
   //         ... do something with f ...
   //      }
   //
   // The function returns 1 if the file is successfully connected, 0 otherwise.


   const char *treename = GetName();
   if (tname && strlen(tname) > 0) treename = tname;
   char *dot = 0;
   {
      char *nextdot =  (char*)strstr(name,".root");
      while (nextdot) {
         dot = nextdot;
         nextdot = (char*)strstr(dot+1,".root");
      }
   }
   //the ".root" is mandatory only if one wants to specify a treename
   //if (!dot) {
   //   Error("AddFile","a chain element name must contain the string .root");
   //   return 0;
   //}

   //Check enough space in fTreeOffset
   if (fNtrees+1 >= fTreeOffsetLen) {
      fTreeOffsetLen *= 2;
      Long64_t *trees = new Long64_t[fTreeOffsetLen];
      for (Int_t i=0;i<=fNtrees;i++) trees[i] = fTreeOffset[i];
      delete [] fTreeOffset;
      fTreeOffset = trees;
   }

   //Search for a a slash between the .root and the end
   Int_t nch = strlen(name) + strlen(treename);
   char *filename = new char[nch+1];
   strlcpy(filename,name,nch+1); 
   if (dot) {
      char *pos = filename + (dot-name) + 5;
      while (*pos) {
         if (*pos == '/') {
            treename = pos+1;
            *pos = 0;
            break;
         }
         pos++;
      }
   }

   // Open the file to get the number of entries.
   Int_t pksize = 0;
   if (nentries <= 0) {
      TFile* file;
      {
         TDirectory::TContext ctxt(0);
         file = TFile::Open(filename);
      }
      if (!file || file->IsZombie()) {
         delete file;
         file = 0;
         delete[] filename;
         filename = 0;
         return 0;
      }

      // Check that tree with the right name exists in the file.
      // Note: We are not the owner of obj, the file is!
      TObject* obj = file->Get(treename);
      if (!obj || !obj->InheritsFrom(TTree::Class())) {
         Error("AddFile", "cannot find tree with name %s in file %s", treename, filename);
         delete file;
         file = 0;
         delete[] filename;
         filename = 0;
         return 0;
      }
      TTree* tree = (TTree*) obj;
      nentries = tree->GetEntries();
      pksize = tree->GetPacketSize();
      // Note: This deletes the tree we fetched.
      delete file;
      file = 0;
   }

   if (nentries > 0) {
      if (nentries != kBigNumber) {
         fTreeOffset[fNtrees+1] = fTreeOffset[fNtrees] + nentries;
         fEntries += nentries;
      } else {
         fTreeOffset[fNtrees+1] = theBigNumber;
         fEntries = nentries;
      }
      fNtrees++;

      TChainElement* element = new TChainElement(treename, filename);
      element->SetPacketSize(pksize);
      element->SetNumberEntries(nentries);
      fFiles->Add(element);
   } else {
      Warning("AddFile", "Adding tree with no entries from file: %s", filename);
   }

   delete [] filename;
   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   return 1;
}

//______________________________________________________________________________
Int_t TChain::AddFileInfoList(TCollection* filelist, Long64_t nfiles /* = kBigNumber */)
{
   // Add all files referenced in the list to the chain. The object type in the
   // list must be either TFileInfo or TObjString or TUrl .
   // The function return 1 if successful, 0 otherwise.
   if (!filelist)
      return 0;
   TIter next(filelist);

   TObject *o = 0;
   Long64_t cnt=0;
   while ((o = next())) {
      // Get the url
      TString cn = o->ClassName();
      const char *url = 0;
      if (cn == "TFileInfo") {
         TFileInfo *fi = (TFileInfo *)o;
         url = (fi->GetCurrentUrl()) ? fi->GetCurrentUrl()->GetUrl() : 0;
         if (!url) {
            Warning("AddFileInfoList", "found TFileInfo with empty Url - ignoring");
            continue;
         }
      } else if (cn == "TUrl") {
         url = ((TUrl*)o)->GetUrl();
      } else if (cn == "TObjString") {
         url = ((TObjString*)o)->GetName();
      }
      if (!url) {
         Warning("AddFileInfoList", "object is of type %s : expecting TFileInfo, TUrl"
                                 " or TObjString - ignoring", o->ClassName());
         continue;
      }
      // Good entry
      cnt++;
      AddFile(url);
      if (cnt >= nfiles)
         break;
   }
   if (fProofChain) {
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);
   }

   return 1;
}

//______________________________________________________________________________
TFriendElement* TChain::AddFriend(const char* chain, const char* dummy /* = "" */)
{
   // -- Add a TFriendElement to the list of friends of this chain.
   //
   // A TChain has a list of friends similar to a tree (see TTree::AddFriend).
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

   if (!fFriends) {
      fFriends = new TList();
   }
   TFriendElement* fe = new TFriendElement(this, chain, dummy);

   R__ASSERT(fe); // There used to be a "if (fe)" test ... Keep this assert until we are sure that fe is never null

   fFriends->Add(fe);

   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   // We need to invalidate the loading of the current tree because its list
   // of real friends is now obsolete.  It is repairable only from LoadTree.
   fTreeNumber = -1;

   TTree* tree = fe->GetTree();
   if (!tree) {
      Warning("AddFriend", "Unknown TChain %s", chain);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement* TChain::AddFriend(const char* chain, TFile* dummy)
{
   // -- Add the whole chain or tree as a friend of this chain.

   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,chain,dummy);

   R__ASSERT(fe); // There used to be a "if (fe)" test ... Keep this assert until we are sure that fe is never null

   fFriends->Add(fe);

   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   // We need to invalidate the loading of the current tree because its list
   // of real friend is now obsolete.  It is repairable only from LoadTree
   fTreeNumber = -1;

   TTree *t = fe->GetTree();
   if (!t) {
      Warning("AddFriend","Unknown TChain %s",chain);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement* TChain::AddFriend(TTree* chain, const char* alias, Bool_t /* warn = kFALSE */)
{
   // -- Add the whole chain or tree as a friend of this chain.

   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,chain,alias);
   R__ASSERT(fe);

   fFriends->Add(fe);

   if (fProofChain)
      // This updates the proxy chain when we will really use PROOF
      ResetBit(kProofUptodate);

   // We need to invalidate the loading of the current tree because its list
   // of real friend is now obsolete.  It is repairable only from LoadTree
   fTreeNumber = -1;

   TTree *t = fe->GetTree();
   if (!t) {
      Warning("AddFriend","Unknown TChain %s",chain->GetName());
   }
   return fe;
}

//______________________________________________________________________________
void TChain::Browse(TBrowser* b)
{
   // -- Browse the contents of the chain.

   TTree::Browse(b);
}

//_______________________________________________________________________
void TChain::CanDeleteRefs(Bool_t flag /* = kTRUE */)
{
   // When closing a file during the chain processing, the file
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
   // -- Initialize the packet descriptor string.

   TIter next(fFiles);
   TChainElement* element = 0;
   while ((element = (TChainElement*) next())) {
      element->CreatePackets();
   }
}

//______________________________________________________________________________
void TChain::DirectoryAutoAdd(TDirectory * /* dir */)
{
   // Override the TTree::DirectoryAutoAdd behavior:
   // we never auto add.

}

//______________________________________________________________________________
Long64_t TChain::Draw(const char* varexp, const TCut& selection,
                      Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Draw expression varexp for selected entries.
   // Returns -1 in case of error or number of selected events in case of success.
   //
   // This function accepts TCut objects as arguments.
   // Useful to use the string operator +, example:
   //    ntuple.Draw("x",cut1+cut2+cut3);
   //
   if (fProofChain) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      fProofChain->SetEventList(fEventList);
      fProofChain->SetEntryList(fEntryList);
      return fProofChain->Draw(varexp, selection, option, nentries, firstentry);
   }

   return TChain::Draw(varexp, selection.GetTitle(), option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TChain::Draw(const char* varexp, const char* selection,
                      Option_t* option,Long64_t nentries, Long64_t firstentry)
{
   // Process all entries in this chain and draw histogram corresponding to
   // expression varexp.
   // Returns -1 in case of error or number of selected events in case of success.

   if (fProofChain) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      fProofChain->SetEventList(fEventList);
      fProofChain->SetEntryList(fEntryList);
      return fProofChain->Draw(varexp, selection, option, nentries, firstentry);
   }
   GetPlayer();
   if (LoadTree(firstentry) < 0) return 0;
   return TTree::Draw(varexp,selection,option,nentries,firstentry);
}

//______________________________________________________________________________
TBranch* TChain::FindBranch(const char* branchname)
{
   // -- See TTree::GetReadEntry().

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->FindBranch(branchname);
   }
   if (fTree) {
      return fTree->FindBranch(branchname);
   }
   LoadTree(0);
   if (fTree) {
      return fTree->FindBranch(branchname);
   }
   return 0;
}

//______________________________________________________________________________
TLeaf* TChain::FindLeaf(const char* searchname)
{
   // -- See TTree::GetReadEntry().

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->FindLeaf(searchname);
   }
   if (fTree) {
      return fTree->FindLeaf(searchname);
   }
   LoadTree(0);
   if (fTree) {
      return fTree->FindLeaf(searchname);
   }
   return 0;
}

//______________________________________________________________________________
const char* TChain::GetAlias(const char* aliasName) const
{
   // -- Returns the expanded value of the alias.  Search in the friends if any.

   const char* alias = TTree::GetAlias(aliasName);
   if (alias) {
      return alias;
   }
   if (fTree) {
      return fTree->GetAlias(aliasName);
   }
   const_cast<TChain*>(this)->LoadTree(0);
   if (fTree) {
      return fTree->GetAlias(aliasName);
   }
   return 0;
}

//______________________________________________________________________________
TBranch* TChain::GetBranch(const char* name)
{
   // -- Return pointer to the branch name in the current tree.

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->GetBranch(name);
   }
   if (fTree) {
      return fTree->GetBranch(name);
   }
   LoadTree(0);
   if (fTree) {
      return fTree->GetBranch(name);
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TChain::GetBranchStatus(const char* branchname) const
{
   // -- See TTree::GetReadEntry().
   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         Warning("GetBranchStatus", "PROOF proxy not up-to-date:"
                                    " run TChain::SetProof(kTRUE, kTRUE) first");
      return fProofChain->GetBranchStatus(branchname);
   }
   return TTree::GetBranchStatus(branchname);
}

//______________________________________________________________________________
Long64_t TChain::GetChainEntryNumber(Long64_t entry) const
{
   // -- Return absolute entry number in the chain.
   // The input parameter entry is the entry number in
   // the current tree of this chain.

   return entry + fTreeOffset[fTreeNumber];
}

//______________________________________________________________________________
Long64_t TChain::GetEntries() const
{
   // -- Return the total number of entries in the chain.
   // In case the number of entries in each tree is not yet known,
   // the offset table is computed.

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         Warning("GetEntries", "PROOF proxy not up-to-date:"
                               " run TChain::SetProof(kTRUE, kTRUE) first");
      return fProofChain->GetEntries();
   }
   if (fEntries >= theBigNumber || fEntries==kBigNumber) {
      const_cast<TChain*>(this)->LoadTree(theBigNumber-1);
   }
   return fEntries;
}

//______________________________________________________________________________
Int_t TChain::GetEntry(Long64_t entry, Int_t getall)
{
   // -- Get entry from the file to memory.
   //
   //     getall = 0 : get only active branches
   //     getall = 1 : get all branches
   //
   // Return the total number of bytes read,
   // 0 bytes read indicates a failure.

   Long64_t treeReadEntry = LoadTree(entry);
   if (treeReadEntry < 0) {
      return 0;
   }
   if (!fTree) {
      return 0;
   }
   return fTree->GetEntry(treeReadEntry, getall);
}

//______________________________________________________________________________
Long64_t TChain::GetEntryNumber(Long64_t entry) const
{
   // -- Return entry number corresponding to entry.
   //
   // if no TEntryList set returns entry
   // else returns entry #entry from this entry list and
   // also computes the global entry number (loads all tree headers)


   if (fEntryList){
      Int_t treenum = 0;
      Long64_t localentry = fEntryList->GetEntryAndTree(entry, treenum);
      //find the global entry number
      //same const_cast as in the GetEntries() function
      if (localentry<0) return -1;
      if (treenum != fTreeNumber){
         if (fTreeOffset[treenum]==theBigNumber){
            for (Int_t i=0; i<=treenum; i++){
               if (fTreeOffset[i]==theBigNumber)
                  (const_cast<TChain*>(this))->LoadTree(fTreeOffset[i-1]);
            }
         }
         //(const_cast<TChain*>(this))->LoadTree(fTreeOffset[treenum]);
      }
      Long64_t globalentry = fTreeOffset[treenum] + localentry;
      return globalentry;
   }
   return entry;
}

//______________________________________________________________________________
Int_t TChain::GetEntryWithIndex(Int_t major, Int_t minor)
{
   // -- Return entry corresponding to major and minor number.
   //
   //  The function returns the total number of bytes read.
   //  If the Tree has friend trees, the corresponding entry with
   //  the index values (major,minor) is read. Note that the master Tree
   //  and its friend may have different entry serial numbers corresponding
   //  to (major,minor).

   Long64_t serial = GetEntryNumberWithIndex(major, minor);
   if (serial < 0) return -1;
   return GetEntry(serial);
}

//______________________________________________________________________________
TFile* TChain::GetFile() const
{
   // -- Return a pointer to the current file.
   // If no file is connected, the first file is automatically loaded.

   if (fFile) {
      return fFile;
   }
   // Force opening the first file in the chain.
   const_cast<TChain*>(this)->LoadTree(0);
   return fFile;
}

//______________________________________________________________________________
TLeaf* TChain::GetLeaf(const char* name)
{
   // -- Return a pointer to the leaf name in the current tree.

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->GetLeaf(name);
   }
   if (fTree) {
      return fTree->GetLeaf(name);
   }
   LoadTree(0);
   if (fTree) {
      return fTree->GetLeaf(name);
   }
   return 0;
}

//______________________________________________________________________________
TObjArray* TChain::GetListOfBranches()
{
   // -- Return a pointer to the list of branches of the current tree.
   //
   // Warning: If there is no current TTree yet, this routine will open the
   //     first in the chain.
   //
   // Returns 0 on failure.

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->GetListOfBranches();
   }
   if (fTree) {
      return fTree->GetListOfBranches();
   }
   LoadTree(0);
   if (fTree) {
      return fTree->GetListOfBranches();
   }
   return 0;
}

//______________________________________________________________________________
TObjArray* TChain::GetListOfLeaves()
{
   // -- Return a pointer to the list of leaves of the current tree.
   //
   // Warning: May set the current tree!
   //

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      return fProofChain->GetListOfLeaves();
   }
   if (fTree) {
      return fTree->GetListOfLeaves();
   }
   LoadTree(0);
   if (fTree) {
      return fTree->GetListOfLeaves();
   }
   return 0;
}

//______________________________________________________________________________
Double_t TChain::GetMaximum(const char* columname)
{
   // -- Return maximum of column with name columname.

   Double_t theMax = -FLT_MAX;
   for (Int_t file = 0; file < fNtrees; file++) {
      Long64_t first = fTreeOffset[file];
      LoadTree(first);
      Double_t curmax = fTree->GetMaximum(columname);
      if (curmax > theMax) {
         theMax = curmax;
      }
   }
   return theMax;
}

//______________________________________________________________________________
Double_t TChain::GetMinimum(const char* columname)
{
   // -- Return minimum of column with name columname.

   Double_t theMin = FLT_MAX;
   for (Int_t file = 0; file < fNtrees; file++) {
      Long64_t first = fTreeOffset[file];
      LoadTree(first);
      Double_t curmin = fTree->GetMinimum(columname);
      if (curmin < theMin) {
         theMin = curmin;
      }
   }
   return theMin;
}

//______________________________________________________________________________
Int_t TChain::GetNbranches()
{
   // -- Return the number of branches of the current tree.
   //
   // Warning: May set the current tree!
   //

   if (fTree) {
      return fTree->GetNbranches();
   }
   LoadTree(0);
   if (fTree) {
      return fTree->GetNbranches();
   }
   return 0;
}

//______________________________________________________________________________
Long64_t TChain::GetReadEntry() const
{
   // -- See TTree::GetReadEntry().

   if (fProofChain && !(fProofChain->TestBit(kProofLite))) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         Warning("GetBranchStatus", "PROOF proxy not up-to-date:"
                                    " run TChain::SetProof(kTRUE, kTRUE) first");
      return fProofChain->GetReadEntry();
   }
   return TTree::GetReadEntry();
}

//______________________________________________________________________________
Double_t TChain::GetWeight() const
{
   // -- Return the chain weight.
   //
   // By default the weight is the weight of the current tree.
   // However, if the weight has been set in TChain::SetWeight()
   // with the option "global", then that weight will be returned.
   //
   // Warning: May set the current tree!
   //

   if (TestBit(kGlobalWeight)) {
      return fWeight;
   } else {
      if (fTree) {
         return fTree->GetWeight();
      }
      const_cast<TChain*>(this)->LoadTree(0);
      if (fTree) {
         return fTree->GetWeight();
      }
      return 0;
   }
}

//______________________________________________________________________________
Int_t TChain::LoadBaskets(Long64_t /*maxmemory*/)
{
   // -- Dummy function.
   // It could be implemented and load all baskets of all trees in the chain.
   // For the time being use TChain::Merge and TTree::LoadBasket
   // on the resulting tree.

   Error("LoadBaskets", "Function not yet implemented for TChain.");
   return 0;
}

//______________________________________________________________________________
Long64_t TChain::LoadTree(Long64_t entry)
{
   // -- Find the tree which contains entry, and set it as the current tree.
   //
   // Returns the entry number in that tree.
   //
   // The input argument entry is the entry serial number in the whole chain.
   //
   // Note: This is the only routine which sets the value of fTree to
   //       a non-zero pointer.
   //

   // We already have been visited while recursively looking
   // through the friends tree, let's return.
   if (kLoadTree & fFriendLockStatus) {
      return 0;
   }

   if (!fNtrees) {
      // -- The chain is empty.
      return 1;
   }

   if ((entry < 0) || ((entry > 0) && (entry >= fEntries && entry!=(theBigNumber-1) ))) {
      // -- Invalid entry number.
      if (fTree) fTree->LoadTree(-1);
      fReadEntry = -1;
      return -2;
   }

   // Find out which tree in the chain contains the passed entry.
   Int_t treenum = fTreeNumber;
   if ((fTreeNumber == -1) || (entry < fTreeOffset[fTreeNumber]) || (entry >= fTreeOffset[fTreeNumber+1]) || (entry==theBigNumber-1)) {
      // -- Entry is *not* in the chain's current tree.
      // Do a linear search of the tree offset array.
      // FIXME: We could be smarter by starting at the
      //        current tree number and going forwards,
      //        then wrapping around at the end.
      for (treenum = 0; treenum < fNtrees; treenum++) {
         if (entry < fTreeOffset[treenum+1]) {
            break;
         }
      }
   }

   // Calculate the entry number relative to the found tree.
   Long64_t treeReadEntry = entry - fTreeOffset[treenum];
   fReadEntry = entry;

   // If entry belongs to the current tree return entry.
   if (fTree && treenum == fTreeNumber) {
      // First set the entry the tree on its owns friends
      // (the friends of the chain will be updated in the
      // next loop).
      fTree->LoadTree(treeReadEntry);
      if (fFriends) {
         // The current tree has not changed but some of its friends might.
         //
         // An alternative would move this code to each of
         // the functions calling LoadTree (and to overload a few more).
         TIter next(fFriends);
         TFriendLock lock(this, kLoadTree);
         TFriendElement* fe = 0;
         TFriendElement* fetree = 0;
         Bool_t needUpdate = kFALSE;
         while ((fe = (TFriendElement*) next())) {
            TObjLink* lnk = 0;
            if (fTree->GetListOfFriends()) {
               lnk = fTree->GetListOfFriends()->FirstLink();
            }
            fetree = 0;
            while (lnk) {
               TObject* obj = lnk->GetObject();
               if (obj->TestBit(TFriendElement::kFromChain) && obj->GetName() && !strcmp(fe->GetName(), obj->GetName())) {
                  fetree = (TFriendElement*) obj;
                  break;
               }
               lnk = lnk->Next();
            }
            TTree* at = fe->GetTree();
            if (at->InheritsFrom(TChain::Class())) {
               Int_t oldNumber = ((TChain*) at)->GetTreeNumber();
               TTree* old = at->GetTree();
               TTree* oldintree = fetree ? fetree->GetTree() : 0;
               at->LoadTreeFriend(entry, this);
               Int_t newNumber = ((TChain*) at)->GetTreeNumber();
               if ((oldNumber != newNumber) || (old != at->GetTree()) || (oldintree && (oldintree != at->GetTree()))) {
                  // We can not compare just the tree pointers because
                  // they could be reused. So we compare the tree
                  // number instead.
                  needUpdate = kTRUE;
                  fTree->RemoveFriend(oldintree);
                  fTree->AddFriend(at->GetTree(), fe->GetName())->SetBit(TFriendElement::kFromChain);
               }
            } else {
               // else we assume it is a simple tree If the tree is a
               // direct friend of the chain, it should be scanned
               // used the chain entry number and NOT the tree entry
               // number (treeReadEntry) hence we redo:
               at->LoadTreeFriend(entry, this);
            }
         }
         if (needUpdate) {
            // Update the branch/leaf addresses and
            // thelist of leaves in all TTreeFormula of the TTreePlayer (if any).

            // Set the branch statuses for the newly opened file.
            TChainElement *frelement;
            TIter fnext(fStatus);
            while ((frelement = (TChainElement*) fnext())) {
               Int_t status = frelement->GetStatus();
               fTree->SetBranchStatus(frelement->GetName(), status);
            }

            // Set the branch addresses for the newly opened file.
            fnext.Reset();
            while ((frelement = (TChainElement*) fnext())) {
               void* addr = frelement->GetBaddress();
               if (addr) {
                  TBranch* br = fTree->GetBranch(frelement->GetName());
                  TBranch** pp = frelement->GetBranchPtr();
                  if (pp) {
                     // FIXME: What if br is zero here?
                     *pp = br;
                  }
                  if (br) {
                     // FIXME: We may have to tell the branch it should
                     //        not be an owner of the object pointed at.
                     br->SetAddress(addr);
                     if (TestBit(kAutoDelete)) {
                        br->SetAutoDelete(kTRUE);
                     }
                  }
               }
            }
            if (fPlayer) {
               fPlayer->UpdateFormulaLeaves();
            }
            // Notify user if requested.
            if (fNotify) {
               fNotify->Notify();
            }
         }
      }
      return treeReadEntry;
   }

   // If the tree has clones, copy them into the chain
   // clone list so we can change their branch addresses
   // when necessary.
   //
   // This is to support the syntax:
   //
   //      TTree* clone = chain->GetTree()->CloneTree(0);
   //
   if (fTree && fTree->GetListOfClones()) {
      for (TObjLink* lnk = fTree->GetListOfClones()->FirstLink(); lnk; lnk = lnk->Next()) {
         TTree* clone = (TTree*) lnk->GetObject();
         AddClone(clone);
      }
   }

   // Delete the current tree and open the new tree.
   TTreeCache* tpf = 0;

   // Delete file unless the file owns this chain!
   // FIXME: The "unless" case here causes us to leak memory.
   if (fFile) {
      if (!fDirectory->GetList()->FindObject(this)) {
         tpf = (TTreeCache*) fFile->GetCacheRead();
         if (tpf) tpf->ResetCache();
         fFile->SetCacheRead(0);
         if (fCanDeleteRefs) {
            fFile->Close("R");
         }
         delete fFile;
         fFile = 0;
         // Note: We do *not* own fTree.
         fTree = 0;
      }
   }

   TChainElement* element = (TChainElement*) fFiles->At(treenum);
   if (!element) {
      if (treeReadEntry) {
         return -4;
      }
      // Last attempt, just in case all trees in the chain have 0 entries.
      element = (TChainElement*) fFiles->At(0);
      if (!element) {
         return -4;
      }
   }

   // FIXME: We leak memory here, we've just lost the open file
   //        if we did not delete it above.
   {
      TDirectory::TContext ctxt(0);
      fFile = TFile::Open(element->GetTitle());
      if (fFile) fFile->SetBit(kMustCleanup);
   }

   // ----- Begin of modifications by MvL
   Int_t returnCode = 0;
   if (!fFile || fFile->IsZombie()) {
      if (fFile) {
         delete fFile;
         fFile = 0;
      }
      // Note: We do *not* own fTree.
      fTree = 0;
      returnCode = -3;
   } else {
      // Note: We do *not* own fTree after this, the file does!
      fTree = (TTree*) fFile->Get(element->GetName());
      if (!fTree) {
         // Now that we do not check during the addition, we need to check here!
         Error("LoadTree", "Cannot find tree with name %s in file %s", element->GetName(), element->GetTitle());
         delete fFile;
         fFile = 0;
         // We do not return yet so that 'fEntries' can be updated with the
         // sum of the entries of all the other trees.
         returnCode = -4;
      }
   }

   fTreeNumber = treenum;
   // FIXME: We own fFile, we must be careful giving away a pointer to it!
   // FIXME: We may set fDirectory to zero here!
   fDirectory = fFile;

   // Reuse cache from previous file (if any).
   if (tpf) {
      if (fFile) {
         tpf->ResetCache();
         fFile->SetCacheRead(tpf);
         tpf->SetFile(fFile);
         // FIXME: fTree may be zero here.
         tpf->UpdateBranches(fTree);
      } else {
         // FIXME: One of the file in the chain is missing
         // we have no place to hold the pointer to the
         // TTreeCache.
         delete tpf;
         tpf = 0;
         this->SetCacheSize(fCacheSize);
      }
   } else {
      this->SetCacheSize(fCacheSize);
   }

   // Check if fTreeOffset has really been set.
   Long64_t nentries = 0;
   if (fTree) {
      nentries = fTree->GetEntries();
   }

   if (fTreeOffset[fTreeNumber+1] != (fTreeOffset[fTreeNumber] + nentries)) {
      fTreeOffset[fTreeNumber+1] = fTreeOffset[fTreeNumber] + nentries;
      fEntries = fTreeOffset[fNtrees];
      element->SetNumberEntries(nentries);
      // Below we must test >= in case the tree has no entries.
      if (entry >= fTreeOffset[fTreeNumber+1]) {
         if ((fTreeNumber < (fNtrees - 1)) && (entry < fTreeOffset[fTreeNumber+2])) {
            return LoadTree(entry);
         } else {
            treeReadEntry = fReadEntry = -2;
         }
      }
   }

   if (!fTree) {
      // The Error message already issued.  However if we reach here
      // we need to make sure that we do not use fTree.
      //
      // Force a reload of the tree next time.
      fTreeNumber = -1;
      return returnCode;
   }
   // ----- End of modifications by MvL

   // Copy the chain's clone list into the new tree's
   // clone list so that branch addresses stay synchronized.
   if (fClones) {
      for (TObjLink* lnk = fClones->FirstLink(); lnk; lnk = lnk->Next()) {
         TTree* clone = (TTree*) lnk->GetObject();
         ((TChain*) fTree)->TTree::AddClone(clone);
      }
   }

   // Since some of the friends of this chain might simple trees
   // (i.e., not really chains at all), we need to execute this
   // before calling LoadTree(entry) on the friends (so that
   // they use the correct read entry number).

   // Change the new current tree to the new entry.
   fTree->LoadTree(treeReadEntry);

   // Change the chain friends to the new entry.
   if (fFriends) {
      // An alternative would move this code to each of the function
      // calling LoadTree (and to overload a few more).
      TIter next(fFriends);
      TFriendLock lock(this, kLoadTree);
      TFriendElement* fe = 0;
      while ((fe = (TFriendElement*) next())) {
         TTree* t = fe->GetTree();
         if (!t) continue;
         if (t->GetTreeIndex()) {
            t->GetTreeIndex()->UpdateFormulaLeaves(0);
         }
         if (t->GetTree() && t->GetTree()->GetTreeIndex()) {
            t->GetTree()->GetTreeIndex()->UpdateFormulaLeaves(GetTree());
         }
         t->LoadTreeFriend(entry, this);
         TTree* friend_t = t->GetTree();
         if (friend_t) {
            fTree->AddFriend(friend_t, fe->GetName())->SetBit(TFriendElement::kFromChain);
         }
      }
   }

   fTree->SetMakeClass(fMakeClass);
   fTree->SetMaxVirtualSize(fMaxVirtualSize);

   SetChainOffset(fTreeOffset[fTreeNumber]);

   // Set the branch statuses for the newly opened file.
   TIter next(fStatus);
   while ((element = (TChainElement*) next())) {
      Int_t status = element->GetStatus();
      fTree->SetBranchStatus(element->GetName(), status);
   }

   // Set the branch addresses for the newly opened file.
   next.Reset();
   while ((element = (TChainElement*) next())) {
      void* addr = element->GetBaddress();
      if (addr) {
         TBranch* br = fTree->GetBranch(element->GetName());
         TBranch** pp = element->GetBranchPtr();
         if (pp) {
            // FIXME: What if br is zero here?
            *pp = br;
         }
         if (br) {
            // FIXME: We may have to tell the branch it should
            //        not be an owner of the object pointed at.
            br->SetAddress(addr);
            if (TestBit(kAutoDelete)) {
               br->SetAutoDelete(kTRUE);
            }
         }
      }
   }

   // Update the addresses of the chain's cloned trees, if any.
   if (fClones) {
      for (TObjLink* lnk = fClones->FirstLink(); lnk; lnk = lnk->Next()) {
         TTree* clone = (TTree*) lnk->GetObject();
         CopyAddresses(clone);
      }
   }

   // Update list of leaves in all TTreeFormula's of the TTreePlayer (if any).
   if (fPlayer) {
      fPlayer->UpdateFormulaLeaves();
   }

   // Notify user we have switched trees if requested.
   if (fNotify) {
      fNotify->Notify();
   }

   // Return the new local entry number.
   return treeReadEntry;
}

//______________________________________________________________________________
void TChain::Lookup(Bool_t force)
{
   // Check / locate the files in the chain.
   // By default only the files not yet looked up are checked.
   // Use force = kTRUE to check / re-check every file.

   TIter next(fFiles);
   TChainElement* element = 0;
   Int_t nelements = fFiles->GetEntries();
   printf("\n");
   printf("TChain::Lookup - Looking up %d files .... \n", nelements);
   Int_t nlook = 0;
   TFileStager *stg = 0;
   while ((element = (TChainElement*) next())) {
      // Do not do it more than needed
      if (element->HasBeenLookedUp() && !force) continue;
      // Count
      nlook++;
      // Get the Url
      TUrl elemurl(element->GetTitle(), kTRUE);
      // Save current options and anchor
      TString anchor = elemurl.GetAnchor();
      TString options = elemurl.GetOptions();
      // Reset options and anchor
      elemurl.SetOptions("");
      elemurl.SetAnchor("");
      // Locate the file
      TString eurl(elemurl.GetUrl());
      if (!stg || !stg->Matches(eurl)) {
         SafeDelete(stg);
         {
            TDirectory::TContext ctxt(0);
            stg = TFileStager::Open(eurl);
         }
         if (!stg) {
            Error("Lookup", "TFileStager instance cannot be instantiated");
            break;
         }
      }
      Int_t n1 = (nelements > 100) ? (Int_t) nelements / 100 : 1;
      if (stg->Locate(eurl.Data(), eurl) == 0) {
         if (nlook > 0 && !(nlook % n1)) {
            printf("Lookup | %3d %% finished\r", 100 * nlook / nelements);
            fflush(stdout);
         }
         // Get the effective end-point Url
         elemurl.SetUrl(eurl);
         // Restore original options and anchor, if any
         elemurl.SetOptions(options);
         elemurl.SetAnchor(anchor);
         // Save it into the element
         element->SetTitle(elemurl.GetUrl());
         // Remember
         element->SetLookedUp();
      } else {
         // Failure: remove
         fFiles->Remove(element);
         if (gSystem->AccessPathName(eurl))
            Error("Lookup", "file %s does not exist\n", eurl.Data());
         else
            Error("Lookup", "file %s cannot be read\n", eurl.Data());
      }
   }
   if (nelements > 0)
      printf("Lookup | %3d %% finished\n", 100 * nlook / nelements);
   else
      printf("\n");
   fflush(stdout);
   SafeDelete(stg);
}

//______________________________________________________________________________
void TChain::Loop(Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Loop on nentries of this chain starting at firstentry.  (NOT IMPLEMENTED)

   Error("Loop", "Function not yet implemented");

   if (option || nentries || firstentry) { }  // keep warnings away

#if 0
   if (LoadTree(firstentry) < 0) return;

   if (firstentry < 0) firstentry = 0;
   Long64_t lastentry = firstentry + nentries -1;
   if (lastentry > fEntries-1) {
      lastentry = fEntries -1;
   }

   GetPlayer();
   GetSelector();
   fSelector->Start(option);

   Long64_t entry = firstentry;
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
void TChain::ls(Option_t* option) const
{
   // -- List the chain.
   TIter next(fFiles);
   TChainElement* file = 0;
   while ((file = (TChainElement*)next())) {
      file->ls(option);
   }
}

//______________________________________________________________________________
Long64_t TChain::Merge(const char* name, Option_t* option)
{
   // Merge all the entries in the chain into a new tree in a new file.
   //
   // See important note in the following function Merge().
   //
   // If the chain is expecting the input tree inside a directory,
   // this directory is NOT created by this routine.
   //
   // So in a case where we have:
   //
   //      TChain ch("mydir/mytree");
   //      ch.Merge("newfile.root");
   //
   // The resulting file will have not subdirectory. To recreate
   // the directory structure do:
   //
   //      TFile* file = TFile::Open("newfile.root", "RECREATE");
   //      file->mkdir("mydir")->cd();
   //      ch.Merge(file);
   //

   TFile *file = TFile::Open(name, "recreate", "chain files", 1);
   return Merge(file, 0, option);
}

//______________________________________________________________________________
Long64_t TChain::Merge(TCollection* /* list */, Option_t* /* option */ )
{
   // Merge all chains in the collection.  (NOT IMPLEMENTED)

   Error("Merge", "not implemented");
   return -1;
}

//______________________________________________________________________________
Long64_t TChain::Merge(TFile* file, Int_t basketsize, Option_t* option)
{
   // Merge all the entries in the chain into a new tree in the current file.
   //
   // Note: The "file" parameter is *not* the file where the new
   //       tree will be inserted.  The new tree is inserted into
   //       gDirectory, which is usually the most recently opened
   //       file, or the directory most recently cd()'d to.
   //
   // If option = "C" is given, the compression level for all branches
   // in the new Tree is set to the file compression level.  By default,
   // the compression level of all branches is the original compression
   // level in the old trees.
   //
   // If basketsize > 1000, the basket size for all branches of the
   // new tree will be set to basketsize.
   //
   // Example using the file generated in $ROOTSYS/test/Event
   // merge two copies of Event.root
   //
   //        gSystem.Load("libEvent");
   //        TChain ch("T");
   //        ch.Add("Event1.root");
   //        ch.Add("Event2.root");
   //        ch.Merge("all.root");
   //
   // If the chain is expecting the input tree inside a directory,
   // this directory is NOT created by this routine.
   //
   // So if you do:
   //
   //      TChain ch("mydir/mytree");
   //      ch.Merge("newfile.root");
   //
   // The resulting file will not have subdirectories.  In order to
   // preserve the directory structure do the following instead:
   //
   //      TFile* file = TFile::Open("newfile.root", "RECREATE");
   //      file->mkdir("mydir")->cd();
   //      ch.Merge(file);
   //
   // If 'option' contains the word 'fast' the merge will be done without
   // unzipping or unstreaming the baskets (i.e., a direct copy of the raw
   // bytes on disk).
   //
   // When 'fast' is specified, 'option' can also contains a
   // sorting order for the baskets in the output file.
   //
   // There is currently 3 supported sorting order:
   //    SortBasketsByOffset (the default)
   //    SortBasketsByBranch
   //    SortBasketsByEntry
   //
   // When using SortBasketsByOffset the baskets are written in
   // the output file in the same order as in the original file
   // (i.e. the basket are sorted on their offset in the original
   // file; Usually this also means that the baskets are sorted
   // on the index/number of the _last_ entry they contain)
   //
   // When using SortBasketsByBranch all the baskets of each
   // individual branches are stored contiguously.  This tends to
   // optimize reading speed when reading a small number (1->5) of
   // branches, since all their baskets will be clustered together
   // instead of being spread across the file.  However it might
   // decrease the performance when reading more branches (or the full
   // entry).
   //
   // When using SortBasketsByEntry the baskets with the lowest
   // starting entry are written first.  (i.e. the baskets are
   // sorted on the index/number of the first entry they contain).
   // This means that on the file the baskets will be in the order
   // in which they will be needed when reading the whole tree
   // sequentially.
   //
   // IMPORTANT Note 1: AUTOMATIC FILE OVERFLOW
   // -----------------------------------------
   // When merging many files, it may happen that the resulting file
   // reaches a size > TTree::fgMaxTreeSize (default = 1.9 GBytes).
   // In this case the current file is automatically closed and a new
   // file started.  If the name of the merged file was "merged.root",
   // the subsequent files will be named "merged_1.root", "merged_2.root",
   // etc.  fgMaxTreeSize may be modified via the static function
   // TTree::SetMaxTreeSize.
   // When in fast mode, the check and switch is only done in between each
   // input file.
   //
   // IMPORTANT Note 2: The output file is automatically closed and deleted.
   // ----------------------------------------------------------------------
   // This is required because in general the automatic file overflow described
   // above may happen during the merge.
   // If only the current file is produced (the file passed as first argument),
   // one can instruct Merge to not close and delete the file by specifying
   // the option "keep".
   //
   // The function returns the total number of files produced.
   // To check that all files have been merged use something like:
   //    if (newchain->GetEntries()!=oldchain->GetEntries()) {
   //      ... not all the file have been copied ...
   //    }

   // We must have been passed a file, we will use it
   // later to reset the compression level of the branches.
   if (!file) {
      // FIXME: We need an error message here.
      return 0;
   }

   // Options
   Bool_t fastClone = kFALSE;
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("fast")) {
      fastClone = kTRUE;
   }

   // The chain tree must have a list of branches
   // because we may try to change their basket
   // size later.
   TObjArray* lbranches = GetListOfBranches();
   if (!lbranches) {
      // FIXME: We need an error message here.
      return 0;
   }

   // The chain must have a current tree because
   // that is the one we will clone.
   if (!fTree) {
      // -- LoadTree() has not yet been called, no current tree.
      // FIXME: We need an error message here.
      return 0;
   }

   // Copy the chain's current tree without
   // copying any entries, we will do that later.
   TTree* newTree = CloneTree(0);
   if (!newTree) {
      // FIXME: We need an error message here.
      return 0;
   }

   // Strip out the (potential) directory name.
   // FIXME: The merged chain may or may not have the
   //        same name as the original chain.  This is
   //        bad because the chain name determines the
   //        names of the trees in the chain by default.
   newTree->SetName(gSystem->BaseName(GetName()));

   // FIXME: Why do we do this?
   newTree->SetAutoSave(2000000000);

   // Circularity is incompatible with merging, it may
   // force us to throw away entries, which is not what
   // we are supposed to do.
   newTree->SetCircular(0);

   // Reset the compression level of the branches.
   if (opt.Contains("c")) {
      TBranch* branch = 0;
      TIter nextb(newTree->GetListOfBranches());
      while ((branch = (TBranch*) nextb())) {
         branch->SetCompressionLevel(file->GetCompressionLevel());
      }
   }

   // Reset the basket size of the branches.
   if (basketsize > 1000) {
      TBranch* branch = 0;
      TIter nextb(newTree->GetListOfBranches());
      while ((branch = (TBranch*) nextb())) {
         branch->SetBasketSize(basketsize);
      }
   }

   // Copy the entries.
   if (fastClone) {
      if ( newTree->CopyEntries( this, -1, option ) < 0 ) {
         // There was a problem!
         Error("Merge", "TTree has not been cloned\n");
      }
   } else {
      newTree->CopyEntries( this, -1, option );
   }

   // Write the new tree header.
   newTree->Write();

   // Get our return value.
   Int_t nfiles = newTree->GetFileNumber() + 1;

   // Close and delete the current file of the new tree.
   if (!opt.Contains("keep")) {
      // Delete the currentFile and the TTree object.
      delete newTree->GetCurrentFile();
   }
   return nfiles;
}

//______________________________________________________________________________
void TChain::Print(Option_t *option) const
{
   // -- Print the header information of each tree in the chain.
   // See TTree::Print for a list of options.

   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      TFile *file = TFile::Open(element->GetTitle());
      if (file && !file->IsZombie()) {
         TTree *tree = (TTree*)file->Get(element->GetName());
         if (tree) tree->Print(option);
      }
      delete file;
   }
}

//______________________________________________________________________________
Long64_t TChain::Process(const char *filename, Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Process all entries in this chain, calling functions in filename.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   // See TTree::Process.

   if (fProofChain) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      fProofChain->SetEventList(fEventList);
      fProofChain->SetEntryList(fEntryList);
      return fProofChain->Process(filename, option, nentries, firstentry);
   }

   if (LoadTree(firstentry) < 0) {
      return 0;
   }
   return TTree::Process(filename, option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TChain::Process(TSelector* selector, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Process this chain executing the code in selector.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (fProofChain) {
      // Make sure the element list is uptodate
      if (!TestBit(kProofUptodate))
         SetProof(kTRUE, kTRUE);
      fProofChain->SetEventList(fEventList);
      fProofChain->SetEntryList(fEntryList);
      return fProofChain->Process(selector, option, nentries, firstentry);
   }

   return TTree::Process(selector, option, nentries, firstentry);
}

//______________________________________________________________________________
void TChain::RecursiveRemove(TObject *obj)
{
   // Make sure that obj (which is being deleted or will soon be) is no
   // longer referenced by this TTree.

   if (fFile == obj) {
      fFile = 0;
      fDirectory = 0;
      fTree = 0;
   }
   if (fDirectory == obj) {
      fDirectory = 0;
      fTree = 0;
   }
   if (fTree == obj) {
      fTree = 0;
   }
}

//______________________________________________________________________________
void TChain::Reset(Option_t*)
{
   // -- Resets the state of this chain.

   delete fFile;
   fFile = 0;
   fNtrees         = 0;
   fTreeNumber     = -1;
   fTree           = 0;
   fFile           = 0;
   fFiles->Delete();
   fStatus->Delete();
   fTreeOffset[0]  = 0;
   TChainElement* element = new TChainElement("*", "");
   fStatus->Add(element);
   fDirectory = 0;

   TTree::Reset();
}

//_______________________________________________________________________
Long64_t TChain::Scan(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Loop on tree and print entries passing selection.
   // If varexp is 0 (or "") then print only first 8 columns.
   // If varexp = "*" print all columns.
   // Otherwise a columns selection can be made using "var1:var2:var3".
   // See TTreePlayer::Scan for more information.

   if (LoadTree(firstentry) < 0) {
      return 0;
   }
   return TTree::Scan(varexp, selection, option, nentries, firstentry);
}

//_______________________________________________________________________
void TChain::SetAutoDelete(Bool_t autodelete)
{
   // -- Set the global branch kAutoDelete bit.
   //
   //  When LoadTree loads a new Tree, the branches for which
   //  the address is set will have the option AutoDelete set
   //  For more details on AutoDelete, see TBranch::SetAutoDelete.

   if (autodelete) {
      SetBit(kAutoDelete, 1);
   } else {
      SetBit(kAutoDelete, 0);
   }
}

//______________________________________________________________________________
void TChain::ResetBranchAddress(TBranch *branch)
{
   // -- Reset the addresses of the branch.

   TChainElement* element = (TChainElement*) fStatus->FindObject(branch->GetName());
   if (element) {
      element->SetBaddress(0);
   }   
   if (fTree) {
      fTree->ResetBranchAddress(branch);
   }
}

//______________________________________________________________________________
void TChain::ResetBranchAddresses()
{
   // Reset the addresses of the branches.

   TIter next(fStatus);
   TChainElement* element = 0;
   while ((element = (TChainElement*) next())) {
      element->SetBaddress(0);
   }
   if (fTree) {
      fTree->ResetBranchAddresses();
   }
}

//_______________________________________________________________________
Int_t TChain::SetBranchAddress(const char *bname, void* add, TBranch** ptr)
{
   // Set branch address.
   //
   //      bname is the name of a branch.
   //      add is the address of the branch.
   //
   //    Note: See the comments in TBranchElement::SetAddress() for a more
   //          detailed discussion of the meaning of the add parameter.
   //
   // IMPORTANT REMARK:
   // In case TChain::SetBranchStatus is called, it must be called
   // BEFORE calling this function.
   //
   // See TTree::CheckBranchAddressType for the semantic of the return value.

   Int_t res = kNoCheck;

   // Check if bname is already in the status list.
   // If not, create a TChainElement object and set its address.
   TChainElement* element = (TChainElement*) fStatus->FindObject(bname);
   if (!element) {
      element = new TChainElement(bname, "");
      fStatus->Add(element);
   }
   element->SetBaddress(add);
   element->SetBranchPtr(ptr);
   // Also set address in current tree.
   // FIXME: What about the chain clones?
   if (fTreeNumber >= 0) {
      TBranch* branch = fTree->GetBranch(bname);
      if (ptr) {
         *ptr = branch;
      }
      if (branch) {
         res = CheckBranchAddressType(branch, TClass::GetClass(element->GetBaddressClassName()), (EDataType) element->GetBaddressType(), element->GetBaddressIsPtr());
         if (fClones) {
            void* oldAdd = branch->GetAddress();
            for (TObjLink* lnk = fClones->FirstLink(); lnk; lnk = lnk->Next()) {
               TTree* clone = (TTree*) lnk->GetObject();
               TBranch* cloneBr = clone->GetBranch(bname);
               if (cloneBr && (cloneBr->GetAddress() == oldAdd)) {
                  // the clone's branch is still pointing to us
                  cloneBr->SetAddress(add);
               }
            }
         }
         branch->SetAddress(add);
      }
   } else {
      if (ptr) {
         *ptr = 0;
      }
   }
   return res;
}

//_______________________________________________________________________
Int_t TChain::SetBranchAddress(const char* bname, void* add, TClass* realClass, EDataType datatype, Bool_t isptr)
{
   // Check if bname is already in the status list, and if not, create a TChainElement object and set its address.
   // See TTree::CheckBranchAddressType for the semantic of the return value.
   //
   //    Note: See the comments in TBranchElement::SetAddress() for a more
   //          detailed discussion of the meaning of the add parameter.
   //
   return SetBranchAddress(bname, add, 0, realClass, datatype, isptr);
}

//_______________________________________________________________________
Int_t TChain::SetBranchAddress(const char* bname, void* add, TBranch** ptr, TClass* realClass, EDataType datatype, Bool_t isptr)
{
   // Check if bname is already in the status list, and if not, create a TChainElement object and set its address.
   // See TTree::CheckBranchAddressType for the semantic of the return value.
   //
   //    Note: See the comments in TBranchElement::SetAddress() for a more
   //          detailed discussion of the meaning of the add parameter.
   //

   TChainElement* element = (TChainElement*) fStatus->FindObject(bname);
   if (!element) {
      element = new TChainElement(bname, "");
      fStatus->Add(element);
   }
   if (realClass) {
      element->SetBaddressClassName(realClass->GetName());
   }
   element->SetBaddressType((UInt_t) datatype);
   element->SetBaddressIsPtr(isptr);
   element->SetBranchPtr(ptr);
   return SetBranchAddress(bname, add, ptr);
}

//_______________________________________________________________________
void TChain::SetBranchStatus(const char* bname, Bool_t status, UInt_t* found)
{
   // -- Set branch status to Process or DoNotProcess
   //
   //      bname is the name of a branch. if bname="*", apply to all branches.
   //      status = 1  branch will be processed
   //             = 0  branch will not be processed
   //  See IMPORTANT REMARKS in TTree::SetBranchStatus and TChain::SetBranchAddress
   //
   //  If found is not 0, the number of branch(es) found matching the regular
   //  expression is returned in *found AND the error message 'unknown branch'
   //  is suppressed.

   // FIXME: We never explicitly set found to zero!

   // Check if bname is already in the status list,
   // if not create a TChainElement object and set its status.
   TChainElement* element = (TChainElement*) fStatus->FindObject(bname);
   if (element) {
      fStatus->Remove(element);
   } else {
      element = new TChainElement(bname, "");
   }
   fStatus->Add(element);
   element->SetStatus(status);
   // Also set status in current tree.
   if (fTreeNumber >= 0) {
      fTree->SetBranchStatus(bname, status, found);
   } else if (found) {
      *found = 1;
   }
}

//______________________________________________________________________________
void TChain::SetDirectory(TDirectory* dir)
{
   // Remove reference to this chain from current directory and add
   // reference to new directory dir. dir can be 0 in which case the chain
   // does not belong to any directory.

   if (fDirectory == dir) return;
   if (fDirectory) fDirectory->Remove(this);
   fDirectory = dir;
   if (fDirectory) {
      fDirectory->Append(this);
      fFile = fDirectory->GetFile();
   } else {
      fFile = 0;
   }
}

//_______________________________________________________________________
void TChain::SetEntryList(TEntryList *elist, Option_t *opt)
{
   //Set the input entry list (processing the entries of the chain will then be
   //limited to the entries in the list)
   //This function finds correspondance between the sub-lists of the TEntryList
   //and the trees of the TChain
   //By default (opt=""), both the file names of the chain elements and
   //the file names of the TEntryList sublists are expanded to full path name.
   //If opt = "ne", the file names are taken as they are and not expanded

   if (fEntryList){
      //check, if the chain is the owner of the previous entry list
      //(it happens, if the previous entry list was created from a user-defined
      //TEventList in SetEventList() function)
      if (fEntryList->TestBit(kCanDelete)) {
         TEntryList *tmp = fEntryList;
         fEntryList = 0; // Avoid problem with RecursiveRemove.
         delete tmp;
      } else {
         fEntryList = 0;
      }
   }
   if (!elist){
      fEntryList = 0;
      fEventList = 0;
      return;
   }
   if (!elist->TestBit(kCanDelete)){
      //this is a direct call to SetEntryList, not via SetEventList
      fEventList = 0;
   }
   if (elist->GetN() == 0){
      fEntryList = elist;
      return;
   }
   if (fProofChain){
      //for processing on proof, event list and entry list can't be
      //set at the same time.
      fEventList = 0;
      fEntryList = elist;
      return;
   }

   Int_t ne = fFiles->GetEntries();
   Int_t listfound=0;
   TString treename, filename;

   TEntryList *templist = 0;
   for (Int_t ie = 0; ie<ne; ie++){
      treename = gSystem->BaseName( ((TChainElement*)fFiles->UncheckedAt(ie))->GetName() );
      filename = ((TChainElement*)fFiles->UncheckedAt(ie))->GetTitle();
      templist = elist->GetEntryList(treename.Data(), filename.Data(), opt);
      if (templist) {
         listfound++;
         templist->SetTreeNumber(ie);
      }
   }

   if (listfound == 0){
      Error("SetEntryList", "No list found for the trees in this chain");
      fEntryList = 0;
      return;
   }
   fEntryList = elist;
   TList *elists = elist->GetLists();
   Bool_t shift = kFALSE;
   TIter next(elists);

   //check, if there are sub-lists in the entry list, that don't
   //correspond to any trees in the chain
   while((templist = (TEntryList*)next())){
      if (templist->GetTreeNumber() < 0){
         shift = kTRUE;
         break;
      }
   }
   fEntryList->SetShift(shift);

}

//_______________________________________________________________________
void TChain::SetEntryListFile(const char *filename, Option_t * /*opt*/)
{
// Set the input entry list (processing the entries of the chain will then be
// limited to the entries in the list). This function creates a special kind
// of entry list (TEntryListFromFile object) that loads lists, corresponding
// to the chain elements, one by one, so that only one list is in memory at a time.
//
// If there is an error opening one of the files, this file is skipped and the
// next file is loaded
//
// File naming convention:
// - by default, filename_elist.root is used, where filename is the
//   name of the chain element
// - xxx$xxx.root - $ sign is replaced by the name of the chain element
// If the list name is not specified (by passing filename_elist.root/listname to
// the TChain::SetEntryList() function, the first object of class TEntryList
// in the file is taken.
//
// It is assumed, that there are as many list files, as there are elements in
// the chain and they are in the same order


   if (fEntryList){
      //check, if the chain is the owner of the previous entry list
      //(it happens, if the previous entry list was created from a user-defined
      //TEventList in SetEventList() function)
      if (fEntryList->TestBit(kCanDelete)) {
         TEntryList *tmp = fEntryList;
         fEntryList = 0; // Avoid problem with RecursiveRemove.
         delete tmp;
      } else {
         fEntryList = 0;
      }
   }

   fEventList = 0;

   TString basename(filename);

   Int_t dotslashpos = basename.Index(".root/");
   TString behind_dot_root = "";
   if (dotslashpos>=0) {
      // Copy the list name specification
      behind_dot_root = basename(dotslashpos+6,basename.Length()-dotslashpos+6);
      // and remove it from basename
      basename.Remove(dotslashpos+5);
   }
   fEntryList = new TEntryListFromFile(basename.Data(), behind_dot_root.Data(), fNtrees);
   fEntryList->SetBit(kCanDelete, kTRUE);
   fEntryList->SetDirectory(0);
   ((TEntryListFromFile*)fEntryList)->SetFileNames(fFiles);
}


//_______________________________________________________________________
void TChain::SetEventList(TEventList *evlist)
{
//This function transfroms the given TEventList into a TEntryList
//
//NOTE, that this function loads all tree headers, because the entry numbers
//in the TEventList are global and have to be recomputed, taking into account
//the number of entries in each tree.
//
//The new TEntryList is owned by the TChain and gets deleted when the chain
//is deleted. This TEntryList is returned by GetEntryList() function, and after
//GetEntryList() function is called, the TEntryList is not owned by the chain
//any more and will not be deleted with it.

   fEventList = evlist;
   if (fEntryList) {
      if (fEntryList->TestBit(kCanDelete)) {
         TEntryList *tmp = fEntryList;
         fEntryList = 0; // Avoid problem with RecursiveRemove.
         delete tmp;
      } else {
         fEntryList = 0;
      }
   }

   if (!evlist) {
      fEntryList = 0;
      fEventList = 0;
      return;
   }

   if(fProofChain) {
      //on proof, fEventList and fEntryList shouldn't be set at the same time
      if (fEntryList){
         //check, if the chain is the owner of the previous entry list
         //(it happens, if the previous entry list was created from a user-defined
         //TEventList in SetEventList() function)
         if (fEntryList->TestBit(kCanDelete)){
            TEntryList *tmp = fEntryList;
            fEntryList = 0; // Avoid problem with RecursiveRemove.
            delete tmp;
         } else {
            fEntryList = 0;
         }
      }
      return;
   }

   char enlistname[100];
   snprintf(enlistname,100, "%s_%s", evlist->GetName(), "entrylist");
   TEntryList *enlist = new TEntryList(enlistname, evlist->GetTitle());
   enlist->SetDirectory(0);

   Int_t nsel = evlist->GetN();
   Long64_t globalentry, localentry;
   const char *treename;
   const char *filename;
   if (fTreeOffset[fNtrees-1]==theBigNumber){
      //Load all the tree headers if the tree offsets are not known
      //It is assumed here, that loading the last tree will load all
      //previous ones
      printf("loading trees\n");
      (const_cast<TChain*>(this))->LoadTree(evlist->GetEntry(evlist->GetN()-1));
   }
   for (Int_t i=0; i<nsel; i++){
      globalentry = evlist->GetEntry(i);
      //add some protection from globalentry<0 here
      Int_t treenum = 0;
      while (globalentry>=fTreeOffset[treenum])
         treenum++;
      treenum--;
      localentry = globalentry - fTreeOffset[treenum];
      // printf("globalentry=%lld, treeoffset=%lld, localentry=%lld\n", globalentry, fTreeOffset[treenum], localentry);
      treename = ((TNamed*)fFiles->At(treenum))->GetName();
      filename = ((TNamed*)fFiles->At(treenum))->GetTitle();
      //printf("entering for tree %s %s\n", treename, filename);
      enlist->SetTree(treename, filename);
      enlist->Enter(localentry);
   }
   enlist->SetBit(kCanDelete, kTRUE);
   enlist->SetReapplyCut(evlist->GetReapplyCut());
   SetEntryList(enlist);
}

//_______________________________________________________________________
void TChain::SetPacketSize(Int_t size)
{
   // -- Set number of entries per packet for parallel root.

   fPacketSize = size;
   TIter next(fFiles);
   TChainElement *element;
   while ((element = (TChainElement*)next())) {
      element->SetPacketSize(size);
   }
}

//______________________________________________________________________________
void TChain::SetProof(Bool_t on, Bool_t refresh, Bool_t gettreeheader)
{
   // Enable/Disable PROOF processing on the current default Proof (gProof).
   //
   // "Draw" and "Processed" commands will be handled by PROOF.
   // The refresh and gettreeheader are meaningfull only if on == kTRUE.
   // If refresh is kTRUE the underlying fProofChain (chain proxy) is always
   // rebuilt (even if already existing).
   // If gettreeheader is kTRUE the header of the tree will be read from the
   // PROOF cluster: this is only needed for browsing and should be used with
   // care because it may take a long time to execute.

   if (!on) {
      // Disable
      SafeDelete(fProofChain);
      // Reset related bit
      ResetBit(kProofUptodate);
   } else {
      if (fProofChain && !refresh &&
         (!gettreeheader || (gettreeheader && fProofChain->GetTree()))) {
         return;
      }
      SafeDelete(fProofChain);
      ResetBit(kProofUptodate);

      // Make instance of TChainProof via the plugin manager
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TChain", "proof"))) {
         if (h->LoadPlugin() == -1)
            return;
         if (!(fProofChain = reinterpret_cast<TChain *>(h->ExecPlugin(2, this, gettreeheader))))
            Error("SetProof", "creation of TProofChain failed");
         // Set related bits
         SetBit(kProofUptodate);
      }
   }
}

//______________________________________________________________________________
void TChain::SetWeight(Double_t w, Option_t* option)
{
   // -- Set chain weight.
   //
   // The weight is used by TTree::Draw to automatically weight each
   // selected entry in the resulting histogram.
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
void TChain::Streamer(TBuffer& b)
{
   // -- Stream a class object.

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         b.ReadClassBuffer(TChain::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TTree::Streamer(b);
      b >> fTreeOffsetLen;
      b >> fNtrees;
      fFiles->Streamer(b);
      if (R__v > 1) {
         fStatus->Streamer(b);
         fTreeOffset = new Long64_t[fTreeOffsetLen];
         b.ReadFastArray(fTreeOffset,fTreeOffsetLen);
      }
      b.CheckByteCount(R__s, R__c, TChain::IsA());
      //====end of old versions

   } else {
      b.WriteClassBuffer(TChain::Class(),this);
   }
}

//______________________________________________________________________________
void TChain::UseCache(Int_t /* maxCacheSize */, Int_t /* pageSize */)
{
   // -- Dummy function kept for back compatibility.
   // The cache is now activated automatically when processing TTrees/TChain.
}
