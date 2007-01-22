// @(#)root/tree:$Name:  $:$Id: TTree.cxx,v 1.315 2007/01/19 16:48:00 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTree                                                                //
//                                                                      //
//  a TTree object has a header with a name and a title.
//  It consists of a list of independent branches (TBranch). Each branch
//  has its own definition and list of buffers. Branch buffers may be
//  automatically written to disk or kept in memory until the Tree attribute
//  fMaxVirtualSize is reached.
//  Variables of one branch are written to the same buffer.
//  A branch buffer is automatically compressed if the file compression
//  attribute is set (default).
//
//  Branches may be written to different files (see TBranch::SetFile).
//
//  The ROOT user can decide to make one single branch and serialize one
//  object into one single I/O buffer or to make several branches.
//  Making one single branch and one single buffer can be the right choice
//  when one wants to process only a subset of all entries in the tree.
//  (you know for example the list of entry numbers you want to process).
//  Making several branches is particularly interesting in the data analysis
//  phase, when one wants to histogram some attributes of an object (entry)
//  without reading all the attributes.
//Begin_Html
/*
<img src="gif/ttree_classtree.gif">
*/
//End_Html
//
//  ==> TTree *tree = new TTree(name, title)
//     Creates a Tree with name and title.
//
//     Various kinds of branches can be added to a tree:
//       A - simple structures or list of variables. (may be for C or Fortran structures)
//       B - any object (inheriting from TObject). (we expect this option be the most frequent)
//       C - a ClonesArray. (a specialized object for collections of same class objects)
//
//  ==> Case A
//      ======
//     TBranch *branch = tree->Branch(branchname,address, leaflist, bufsize)
//       * address is the address of the first item of a structure
//       * leaflist is the concatenation of all the variable names and types
//         separated by a colon character :
//         The variable name and the variable type are separated by a slash (/).
//         The variable type may be 0,1 or 2 characters. If no type is given,
//         the type of the variable is assumed to be the same as the previous
//         variable. If the first variable does not have a type, it is assumed
//         of type F by default. The list of currently supported types is given below:
//            - C : a character string terminated by the 0 character
//            - B : an 8 bit signed integer (Char_t)
//            - b : an 8 bit unsigned integer (UChar_t)
//            - S : a 16 bit signed integer (Short_t)
//            - s : a 16 bit unsigned integer (UShort_t)
//            - I : a 32 bit signed integer (Int_t)
//            - i : a 32 bit unsigned integer (UInt_t)
//            - F : a 32 bit floating point (Float_t)
//            - D : a 64 bit floating point (Double_t)
//            - L : a 64 bit signed integer (Long64_t)
//            - l : a 64 bit unsigned integer (ULong64_t)
//            - O : a boolean (Bool_t)
//
//  ==> Case B
//      ======
//     TBranch *branch = tree->Branch(branchname,className,object, bufsize, splitlevel)
//          object is the address of a pointer to an existing object (derived from TObject).
//        if splitlevel=0, the object is serialized in the branch buffer.
//        if splitlevel=1 (default), this branch will automatically be split
//          into subbranches, with one subbranch for each data member or object
//          of the object itself. In case the object member is a TClonesArray,
//          the mechanism described in case C is applied to this array.
//        if splitlevel=2 ,this branch will automatically be split
//          into subbranches, with one subbranch for each data member or object
//          of the object itself. In case the object member is a TClonesArray,
//          it is processed as a TObject*, only one branch.
//
//  ==> Case C
//      ======
//     TBranch *branch = tree->Branch(branchname,clonesarray, bufsize, splitlevel)
//         clonesarray is the address of a pointer to a TClonesArray.
//         The TClonesArray is a direct access list of objects of the same class.
//         For example, if the TClonesArray is an array of TTrack objects,
//         this function will create one subbranch for each data member of
//         the object TTrack.
//
//
//  ==> branch->SetAddress(Void *address)
//      In case of dynamic structures changing with each entry for example, one must
//      redefine the branch address before filling the branch again.
//      This is done via the TBranch::SetAddress member function.
//
//  ==> tree->Fill()
//      loops on all defined branches and for each branch invokes the Fill function.
//
//         See also the class TNtuple (a simple Tree with branches of floats)
//
//       Adding a Branch to an Existing Tree
//       ===================================
// You may want to add a branch to an existing tree. For example,
// if one variable in the tree was computed with a certain algorithm,
// you may want to try another algorithm and compare the results.
// One solution is to add a new branch, fill it, and save the tree.
// The code below adds a simple branch to an existing tree.
// Note the kOverwrite option in the Write method, it overwrites the
// existing tree. If it is not specified, two copies of the tree headers
// are saved.
//
// void tree3AddBranch(){
//   TFile f("tree3.root","update");
//
//   Float_t new_v;
//   TTree *t3 = (TTree*)f->Get("t3");
//   TBranch *newBranch = t3->Branch("new_v",&new_v,"new_v/F");
//
//   //read the number of entries in the t3
//   Long64_t nentries = t3->GetEntries();
//
//   for (Long64_t i = 0; i < nentries; i++){
//     new_v= gRandom->Gaus(0,1);
//     newBranch->Fill();
//   }
//   // save only the new version of the tree
//   t3->Write("",TObject::kOverwrite);
// }
// Adding a branch is often not possible because the tree is in a read-only
// file and you do not have permission to save the modified tree with the
// new branch. Even if you do have the permission, you risk loosing the
// original tree with an unsuccessful attempt to save  the modification.
// Since trees are usually large, adding a branch could extend it over the
// 2GB  limit. In this case, the attempt to write the tree fails, and the
// original data is erased.
// In addition, adding a branch to a tree enlarges the tree and increases
// the amount of memory needed to read an entry, and therefore decreases
// the performance.
// For these reasons, ROOT offers the concept of friends for trees (and chains).
// We encourage you to use TTree::AddFriend rather than adding a branch manually.
//
//Begin_Html
/*
<img src="gif/tree_layout.gif">
*/
//End_Html
//  =============================================================================
//______________________________________________________________________________
//*-*-*-*-*-*-*A simple example with histograms and a tree*-*-*-*-*-*-*-*-*-*
//*-*          ===========================================
//
//  This program creates :
//    - a one dimensional histogram
//    - a two dimensional histogram
//    - a profile histogram
//    - a tree
//
//  These objects are filled with some random numbers and saved on a file.
//
//-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//
// #include "TFile.h"
// #include "TH1.h"
// #include "TH2.h"
// #include "TProfile.h"
// #include "TRandom.h"
// #include "TTree.h"
//
//
// //______________________________________________________________________________
// main(int argc, char **argv)
// {
// // Create a new ROOT binary machine independent file.
// // Note that this file may contain any kind of ROOT objects, histograms,trees
// // pictures, graphics objects, detector geometries, tracks, events, etc..
// // This file is now becoming the current directory.
//   TFile hfile("htree.root","RECREATE","Demo ROOT file with histograms & trees");
//
// // Create some histograms and a profile histogram
//   TH1F *hpx   = new TH1F("hpx","This is the px distribution",100,-4,4);
//   TH2F *hpxpy = new TH2F("hpxpy","py ps px",40,-4,4,40,-4,4);
//   TProfile *hprof = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);
//
// // Define some simple structures
//   typedef struct {Float_t x,y,z;} POINT;
//   typedef struct {
//      Int_t ntrack,nseg,nvertex;
//      UInt_t flag;
//      Float_t temperature;
//   } EVENTN;
//   static POINT point;
//   static EVENTN eventn;
//
// // Create a ROOT Tree
//   TTree *tree = new TTree("T","An example of ROOT tree with a few branches");
//   tree->Branch("point",&point,"x:y:z");
//   tree->Branch("eventn",&eventn,"ntrack/I:nseg:nvertex:flag/i:temperature/F");
//   tree->Branch("hpx","TH1F",&hpx,128000,0);
//
//   Float_t px,py,pz;
//   static Float_t p[3];
//
// //--------------------Here we start a loop on 1000 events
//   for ( Int_t i=0; i<1000; i++) {
//      gRandom->Rannor(px,py);
//      pz = px*px + py*py;
//      Float_t random = gRandom->::Rndm(1);
//
// //         Fill histograms
//      hpx->Fill(px);
//      hpxpy->Fill(px,py,1);
//      hprof->Fill(px,pz,1);
//
// //         Fill structures
//      p[0] = px;
//      p[1] = py;
//      p[2] = pz;
//      point.x = 10*(random-1);;
//      point.y = 5*random;
//      point.z = 20*random;
//      eventn.ntrack  = Int_t(100*random);
//      eventn.nseg    = Int_t(2*eventn.ntrack);
//      eventn.nvertex = 1;
//      eventn.flag    = Int_t(random+0.5);
//      eventn.temperature = 20+random;
//
// //        Fill the tree. For each event, save the 2 structures and 3 objects
// //      In this simple example, the objects hpx, hprof and hpxpy are slightly
// //      different from event to event. We expect a big compression factor!
//      tree->Fill();
//   }
//  //--------------End of the loop
//
//   tree->Print();
//
// // Save all objects in this file
//   hfile.Write();
//
// // Close the file. Note that this is automatically done when you leave
// // the application.
//   hfile.Close();
//
//   return 0;
// }
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfig.h"
#include "TTree.h"

#include "Api.h"
#include "TArrayC.h"
#include "TBufferFile.h"
#include "TBaseClass.h"
#include "TBasket.h"
#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TBranchRef.h"
#include "TBrowser.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClonesArray.h"
#include "TCut.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TDirectory.h"
#include "TError.h"
#include "TEntryList.h"
#include "TEventList.h"
#include "TFile.h"
#include "TFolder.h"
#include "TFriendElement.h"
#include "TInterpreter.h"
#include "TLeaf.h"
#include "TLeafB.h"
#include "TLeafC.h"
#include "TLeafD.h"
#include "TLeafElement.h"
#include "TLeafF.h"
#include "TLeafI.h"
#include "TLeafL.h"
#include "TLeafObject.h"
#include "TLeafS.h"
#include "TList.h"
#include "TMath.h"
#include "TROOT.h"
#include "TRealData.h"
#include "TRegexp.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTreeCloner.h"
#include "TTreeCache.h"
#include "TVirtualCollectionProxy.h"
#include "TEmulatedCollectionProxy.h"
#include "TVirtualFitter.h"
#include "TVirtualIndex.h"
#include "TVirtualPad.h"

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

Int_t    TTree::fgBranchStyle = 1;  // Use new TBranch style with TBranchElement.
Long64_t TTree::fgMaxTreeSize = 1900000000;

TTree* gTree;

ClassImp(TTree)

//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//

//______________________________________________________________________________
//  Helper class to prevent infinite recursion in the usage of TTree Friends.

//______________________________________________________________________________
TTree::TFriendLock::TFriendLock(TTree* tree, UInt_t methodbit)
: fTree(tree)
{
   // -- Record in tree that it has been used while recursively looks through the friends.

   // We could also add some code to acquire an actual
   // lock to prevent multi-thread issues
   if (fTree) {
      fMethodBit = methodbit;
      fPrevious = fTree->fFriendLockStatus & fMethodBit;
      fTree->fFriendLockStatus |= fMethodBit;
   }
}

//______________________________________________________________________________
TTree::TFriendLock::TFriendLock(const TFriendLock& tfl) :
  fTree(tfl.fTree),
  fMethodBit(tfl.fMethodBit),
  fPrevious(tfl.fPrevious)
{
   //copy constructor
}

//______________________________________________________________________________
TTree::TFriendLock& TTree::TFriendLock::operator=(const TTree::TFriendLock& tfl)
{
   //assignement operator
   if(this!=&tfl) {
      fTree=tfl.fTree;
      fMethodBit=tfl.fMethodBit;
      fPrevious=tfl.fPrevious;
   }
   return *this;
}

//______________________________________________________________________________
TTree::TFriendLock::~TFriendLock()
{
   // -- Restore the state of tree the same as before we set the lock.

   if (fTree) {
      if (!fPrevious) {
         fTree->fFriendLockStatus &= ~(fMethodBit & kBitMask);
      }
   }
}

//
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//

//______________________________________________________________________________
TTree::TTree()
: TNamed()
, TAttLine()
, TAttFill()
, TAttMarker()
, fEntries(0)
, fTotBytes(0)
, fZipBytes(0)
, fSavedBytes(0)
, fWeight(1)
, fTimerInterval(0)
, fScanField(25)
, fUpdate(0)
, fMaxEntries(0)
, fMaxEntryLoop(0)
, fMaxVirtualSize(0)
, fAutoSave(100000000)
, fEstimate(1000000)
, fCacheSize(10000000)
, fChainOffset(0)
, fReadEntry(-1)
, fTotalBuffers(0)
, fPacketSize(100)
, fNfill(0)
, fDebug(0)
, fDebugMin(0)
, fDebugMax(9999999)
, fMakeClass(0)
, fFileNumber(0)
, fNotify(0)
, fDirectory(0)
, fBranches()
, fLeaves()
, fAliases(0)
, fEventList(0)
, fEntryList(0)
, fIndexValues()
, fIndex()
, fTreeIndex(0)
, fFriends(0)
, fUserInfo(0)
, fPlayer(0)
, fClones(0)
, fBranchRef(0)
, fFriendLockStatus(0)
{
   // -- Default and i/o tree constructor.
   //
   // Note: We do *not* insert ourself into the current directory.
   //

   fMaxEntries = 1000000000;
   fMaxEntries *= 1000;

   fMaxEntryLoop = 1000000000;
   fMaxEntryLoop *= 1000;
}

//______________________________________________________________________________
TTree::TTree(const char* name, const char* title, Int_t splitlevel /* = 99 */)
: TNamed(name, title)
, TAttLine()
, TAttFill()
, TAttMarker()
, fEntries(0)
, fTotBytes(0)
, fZipBytes(0)
, fSavedBytes(0)
, fWeight(1)
, fTimerInterval(0)
, fScanField(25)
, fUpdate(0)
, fMaxEntries(0)
, fMaxEntryLoop(0)
, fMaxVirtualSize(0)
, fAutoSave(100000000)
, fEstimate(1000000)
, fCacheSize(10000000)
, fChainOffset(0)
, fReadEntry(-1)
, fTotalBuffers(0)
, fPacketSize(100)
, fNfill(0)
, fDebug(0)
, fDebugMin(0)
, fDebugMax(9999999)
, fMakeClass(0)
, fFileNumber(0)
, fNotify(0)
, fDirectory(0)
, fBranches()
, fLeaves()
, fAliases(0)
, fEventList(0)
, fEntryList(0)
, fIndexValues()
, fIndex()
, fTreeIndex(0)
, fFriends(0)
, fUserInfo(0)
, fPlayer(0)
, fClones(0)
, fBranchRef(0)
, fFriendLockStatus(0)
{
   // -- Normal tree constructor.
   //
   // The tree is created in the current directory.
   // Use the various functions Branch below to add branches to this tree.
   //
   // If the first character of title is a "/", the function assumes a folder name.
   // In this case, it creates automatically branches following the folder hierarchy.
   // splitlevel may be used in this case to control the split level.

   // TAttLine state.
   SetLineColor(gStyle->GetHistLineColor());
   SetLineStyle(gStyle->GetHistLineStyle());
   SetLineWidth(gStyle->GetHistLineWidth());

   // TAttFill state.
   SetFillColor(gStyle->GetHistFillColor());
   SetFillStyle(gStyle->GetHistFillStyle());

   // TAttMarkerState.
   SetMarkerColor(gStyle->GetMarkerColor());
   SetMarkerStyle(gStyle->GetMarkerStyle());
   SetMarkerSize(gStyle->GetMarkerSize());

   fMaxEntries = 1000000000;
   fMaxEntries *= 1000;

   fMaxEntryLoop = 1000000000;
   fMaxEntryLoop *= 1000;

   // Insert ourself into the current directory.
   // FIXME: This is very annoying behaviour, we should
   //        be able to choose to not do this like we
   //        can with a histogram.
   gDirectory->Append(this);
   fDirectory = gDirectory;

   // We become the current tree.
   gTree = this;

   // if title starts with ":" and is a valid folder name, a superbranch
   // is created.
   // FIXME: Why?
   if (strlen(title) > 2) {
      if (title[0] == '/') {
         Branch(title+1,32000,splitlevel);
      }
   }
}

//______________________________________________________________________________
TTree::~TTree()
{
   // -- Destructor.

   if (fDirectory) {
      // -- We are in a directory, which may possibly be a file.
      if (fDirectory->GetList()) {
         // -- Remove us from the directory listing.
         fDirectory->GetList()->Remove(this);
      }
      //delete the file cache if it points to this Tree
      TFile *file = fDirectory->GetFile();
      if (file) {
         TFileCacheRead *pf = file->GetCacheRead();
         if (pf && pf->InheritsFrom(TTreeCache::Class())) {
            TTreeCache *tpf = (TTreeCache*)pf;
            if (tpf->GetTree() == this) {
               delete tpf;
               tpf = 0;
               file->SetCacheRead(0);
            }
         }
      }
   }
   // We don't own the leaves in fLeaves, the branches do.
   fLeaves.Clear();
   // I'm ready to destroy any objects allocated by
   // SetAddress() by my branches.  If I have clones,
   // tell them to zero their pointers to this shared
   // memory.
   if (fClones && fClones->GetEntries()) {
      // -- I have clones.
      // I am about to delete the objects created by
      // SetAddress() which we are sharing, so tell
      // the clones to release their pointers to them.
      for (TObjLink* lnk = fClones->FirstLink(); lnk; lnk = lnk->Next()) {
         TTree* clone = (TTree*) lnk->GetObject();
         clone->ResetBranchAddresses();
      }
   }
   // Get rid of our branches, note that this will also release
   // any memory allocated by TBranchElement::SetAddress().
   fBranches.Delete();
   // FIXME: We must consider what to do with the reset of these if we are a clone.
   delete fPlayer;
   fPlayer = 0;
   if (fFriends) {
      fFriends->Delete();
      delete fFriends;
      fFriends = 0;
   }
   if (fAliases) {
      fAliases->Delete();
      delete fAliases;
      fAliases = 0;
   }
   if (fUserInfo) {
      fUserInfo->Delete();
      delete fUserInfo;
      fUserInfo = 0;
   }
   if (fClones) {
      // Clone trees should no longer be removed from fClones when they are deleted.
      gROOT->GetListOfCleanups()->Remove(fClones);
      // Note: fClones does not own its content.
      delete fClones;
      fClones = 0;
   if (fEntryList){
      if (fEntryList->TestBit(kCanDelete)){
         delete fEntryList;
         fEntryList=0;
      }
   }
   }
   delete fTreeIndex;
   fTreeIndex = 0;
   delete fBranchRef;
   fBranchRef = 0;
   // Must be done after the destruction of friends.
   // Note: We do *not* own our directory.
   fDirectory = 0;
}

//______________________________________________________________________________
void TTree::AddClone(TTree* clone)
{
   // -- Add a cloned tree to our list of trees to be notified whenever we change our branch addresses or when we are deleted.
   if (!fClones) {
      fClones = new TList();
      fClones->SetOwner(false);
      // So that the clones are automatically removed from the list when
      // they are deleted.
      gROOT->GetListOfCleanups()->Add(fClones);
   }
   if (!fClones->FindObject(clone)) {
      fClones->Add(clone);
   }
}


//______________________________________________________________________________
TFriendElement* TTree::AddFriend(const char* treename, const char* filename)
{
   // -- Add a TFriendElement to the list of friends.
   //
   // This function:
   //   -opens a file if filename is specified
   //   -reads a Tree with name treename from the file (current directory)
   //   -adds the Tree to the list of friends
   // see other AddFriend functions
   //
   // A TFriendElement TF describes a TTree object TF in a file.
   // When a TFriendElement TF is added to the the list of friends of an
   // existing TTree T, any variable from TF can be referenced in a query
   // to T.
   //
   //   A tree keeps a list of friends. In the context of a tree (or a chain),
   // friendship means unrestricted access to the friends data. In this way
   // it is much like adding another branch to the tree without taking the risk
   // of damaging it. To add a friend to the list, you can use the TTree::AddFriend
   // method.  The tree in the diagram below has two friends (friend_tree1 and
   // friend_tree2) and now has access to the variables a,b,c,i,j,k,l and m.
   //
   //Begin_Html
   /*
   <img src="gif/tree_friend1.gif">
   */
   //End_Html
   //
   // The AddFriend method has two parameters, the first is the tree name and the
   // second is the name of the ROOT file where the friend tree is saved.
   // AddFriend automatically opens the friend file. If no file name is given,
   // the tree called ft1 is assumed to be in the same file as the original tree.
   //
   // tree.AddFriend("ft1","friendfile1.root");
   // If the friend tree has the same name as the original tree, you can give it
   // an alias sin the context of the friendship:
   //
   // tree.AddFriend("tree1 = tree","friendfile1.root");
   // Once the tree has friends, we can use TTree::Draw as if the friend's
   // variables were in the original tree. To specify which tree to use in
   // the Draw method, use the syntax:
   //
   // <treeName>.<branchname>.<varname>
   // If the variablename is enough to uniquely identify the variable, you can
   // leave out the tree and/or branch name.
   // For example, these commands generate a 3-d scatter plot of variable "var"
   // in the TTree tree versus variable v1 in TTree ft1 versus variable v2 in
   // TTree ft2.
   //
   // tree.AddFriend("ft1","friendfile1.root");
   // tree.AddFriend("ft2","friendfile2.root");
   // tree.Draw("var:ft1.v1:ft2.v2");
   //
   //Begin_Html
   /*
   <img src="gif/tree_friend2.gif">
   */
   //End_Html
   //
   // The picture illustrates the access of the tree and its friends with a
   // Draw command.
   // When AddFriend is called, the ROOT file is automatically opened and the
   // friend tree (ft1) is read into memory. The new friend (ft1) is added to
   // the list of friends of tree.
   // The number of entries in the friend must be equal or greater to the number
   // of entries of the original tree. If the friend tree has fewer entries a
   // warning is given and the missing entries are not included in the histogram.
   // To retrieve the list of friends from a tree use TTree::GetListOfFriends.
   // When the tree is written to file (TTree::Write), the friends list is saved
   // with it. And when the tree is retrieved, the trees on the friends list are
   // also retrieved and the friendship restored.
   // When a tree is deleted, the elements of the friend list are also deleted.
   // It is possible to declare a friend tree that has the same internal
   // structure (same branches and leaves) as the original tree, and compare the
   // same values by specifying the tree.
   //
   //  tree.Draw("var:ft1.var:ft2.var")

   //if (kAddFriend & fFriendLockStatus)

   if (!fFriends) {
      fFriends = new TList();
   }
   TFriendElement* fe = new TFriendElement(this, treename, filename);
   R__ASSERT(fe); // this assert is for historical reasons. Don't remove it unless you understand all the consequences.
   fFriends->Add(fe);
   TTree* t = fe->GetTree();
   if (t) {
      if (!t->GetTreeIndex() && (t->GetEntries() < fEntries)) {
         Warning("AddFriend", "FriendElement %s in file %s has less entries %g than its parent Tree: %g", treename, filename, t->GetEntries(), fEntries);
      }
   } else {
      Warning("AddFriend", "Cannot add FriendElement %s in file %s", treename, filename);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement* TTree::AddFriend(const char* treename, TFile* file)
{
   // -- Add a TFriendElement to the list of friends.
   //
   // The TFile is managed by the user (e.g. the user must delete the file).
   // For complete description see AddFriend(const char *, const char *).
   // This function:
   //   -reads a Tree with name treename from the file
   //   -adds the Tree to the list of friends

   if (!fFriends) {
      fFriends = new TList();
   }
   TFriendElement *fe = new TFriendElement(this, treename, file);
   R__ASSERT(fe);
   fFriends->Add(fe);
   TTree *t = fe->GetTree();
   if (t) {
      if (!t->GetTreeIndex() && (t->GetEntries() < fEntries)) {
         Warning("AddFriend", "FriendElement %s in file %s has less entries %g than its parent tree: %g", treename, file->GetName(), t->GetEntries(), fEntries);
      }
   } else {
      Warning("AddFriend", "unknown tree '%s' in file '%s'", treename, file->GetName());
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement* TTree::AddFriend(TTree* tree, const char* alias, Bool_t warn)
{
   // -- Add a TFriendElement to the list of friends.
   //
   // The TTree is managed by the user (e.g., the user must delete the file).
   // For a complete description see AddFriend(const char *, const char *).

   if (!tree) {
      return 0;
   }
   if (!fFriends) {
      fFriends = new TList();
   }
   TFriendElement* fe = new TFriendElement(this, tree, alias);
   R__ASSERT(fe); // this assert is for historical reasons. Don't remove it unless you understand all the consequences.
   fFriends->Add(fe);
   TTree* t = fe->GetTree();
   if (warn && (t->GetEntries() < fEntries)) {
      Warning("AddFriend", "FriendElement '%s' in file '%s' has less entries %g than its parent tree: %g", tree->GetName(), fe->GetFile() ? fe->GetFile()->GetName() : "(memory resident)", t->GetEntries(), fEntries);
   }
   return fe;
}

//______________________________________________________________________________
Long64_t TTree::AutoSave(Option_t* option)
{
   // -- AutoSave tree header every fAutoSave bytes.
   //
   //   When large Trees are produced, it is safe to activate the AutoSave
   //   procedure. Some branches may have buffers holding many entries.
   //   AutoSave is automatically called by TTree::Fill when the number of bytes
   //   generated since the previous AutoSave is greater than fAutoSave bytes.
   //   This function may also be invoked by the user, for example every
   //   N entries.
   //   Each AutoSave generates a new key on the file.
   //   Once the key with the tree header has been written, the previous cycle
   //   (if any) is deleted.
   //
   //   Note that calling TTree::AutoSave too frequently (or similarly calling
   //   TTree::SetAutoSave with a small value) is an expensive operation.
   //   You should make tests for your own application to find a compromize
   //   between speed and the quantity of information you may loose in case of
   //   a job crash.
   //
   //   In case your program crashes before closing the file holding this tree,
   //   the file will be automatically recovered when you will connect the file
   //   in UPDATE mode.
   //   The Tree will be recovered at the status corresponding to the last AutoSave.
   //
   //   if option contains "SaveSelf", gDirectory->SaveSelf() is called.
   //   This allows another process to analyze the Tree while the Tree is being filled.
   //
   //   By default the previous header is deleted after having written the new header.
   //   if option contains "Overwrite", the previous Tree header is deleted
   //   before written the new header. This option is slightly faster, but
   //   the default option is safer in case of a problem (disk quota exceeded)
   //   when writing the new header.
   //
   //   The function returns the number of bytes written to the file.
   //   if the number of bytes is null, an error has occured while writing
   //   the header to the file.
   //
   //   How to write a Tree in one process and view it from another process
   //   ===================================================================
   //   The following two scripts illustrate how to do this.
   //   The script treew.C is executed by process1, treer.C by process2
   //
   //   ----- script treew.C
   //   void treew() {
   //     TFile f("test.root","recreate");
   //     TNtuple *ntuple = new TNtuple("ntuple","Demo","px:py:pz:random:i");
   //     Float_t px, py, pz;
   //     for ( Int_t i=0; i<10000000; i++) {
   //        gRandom->Rannor(px,py);
   //        pz = px*px + py*py;
   //        Float_t random = gRandom->Rndm(1);
   //        ntuple->Fill(px,py,pz,random,i);
   //        if (i%1000 == 1) ntuple->AutoSave("SaveSelf");
   //     }
   //   }
   //
   //   ----- script treer.C
   //   void treer() {
   //      TFile f("test.root");
   //      TTree *ntuple = (TTree*)f.Get("ntuple");
   //      TCanvas c1;
   //      Int_t first = 0;
   //      while(1) {
   //         if (first == 0) ntuple->Draw("px>>hpx", "","",10000000,first);
   //         else            ntuple->Draw("px>>+hpx","","",10000000,first);
   //         first = (Int_t)ntuple->GetEntries();
   //         c1.Update();
   //         gSystem->Sleep(1000); //sleep 1 second
   //         ntuple->Refresh();
   //      }
   //   }

   if (!fDirectory || fDirectory == gROOT || !fDirectory->IsWritable()) return 0;
   if (gDebug > 0) {
      printf("AutoSave Tree:%s after %lld bytes written\n",GetName(),fTotBytes);
   }
   TString opt = option;
   opt.ToLower();
   fSavedBytes = fTotBytes;
   TDirectory *dirsav = gDirectory;
   fDirectory->cd();
   TKey *key = (TKey*)fDirectory->GetListOfKeys()->FindObject(GetName());
   Long64_t nbytes;
   if (opt.Contains("overwrite")) {
      nbytes = Write("",TObject::kOverwrite);
   } else {
      nbytes = Write(); //nbytes will be 0 if Write failed (disk space exceeded)
      if (nbytes && key) {
         key->Delete();
         delete key;
      }
   }
   // save StreamerInfo
   TFile *file = fDirectory->GetFile();
   if (file) file->WriteStreamerInfo();

   if (opt.Contains("saveself")) fDirectory->SaveSelf();

   dirsav->cd();
   return nbytes;
}

//______________________________________________________________________________
TBranch* TTree::BranchImp(const char* branchname, const char* classname, TClass* ptrClass, void* addobj, Int_t bufsize, Int_t splitlevel)
{
   // -- Same as TTree::Branch() with added check that addobj matches className.
   //
   // See TTree::Branch() for other details.
   //

   if (!ptrClass) {
      TClass* claim = gROOT->GetClass(classname);
      if (claim && claim->GetCollectionProxy() && dynamic_cast<TEmulatedCollectionProxy*>(claim->GetCollectionProxy())) {
         Error("Branch", "The class requested (%s) for the branch \"%s\" refer to an stl collection and do not have a compiled CollectionProxy.  "
               "Please generate the dictionary for this class (%s)", 
               claim->GetName(), branchname, claim->GetName());
         return 0;
      }
      return Branch(branchname, classname, (void*) addobj, bufsize, splitlevel);
   }
   TClass* claim = gROOT->GetClass(classname);
   TClass* actualClass = 0;
   void** addr = (void**) addobj;
   if (addr) {
      actualClass = ptrClass->GetActualClass(*addr);
   }
   if (ptrClass && claim) {
      if (!(claim->InheritsFrom(ptrClass) || ptrClass->InheritsFrom(claim))) {
         // Note we currently do not warn in case of splicing or over-expectation).
         if (claim->IsLoaded() && ptrClass->IsLoaded() && strcmp( claim->GetTypeInfo()->name(), ptrClass->GetTypeInfo()->name() ) == 0) {
            // The type is the same according to the C++ type_info, we must be in the case of 
            // a template of Double32_t.  This is actually a correct case.
         } else {
            Error("Branch", "The class requested (%s) for \"%s\" is different from the type of the pointer passed (%s)", 
                  claim->GetName(), branchname, ptrClass->GetName());
         }
      } else if (actualClass && (claim != actualClass) && !actualClass->InheritsFrom(claim)) {
         if (claim->IsLoaded() && actualClass->IsLoaded() && strcmp( claim->GetTypeInfo()->name(), actualClass->GetTypeInfo()->name() ) == 0) {
            // The type is the same according to the C++ type_info, we must be in the case of 
            // a template of Double32_t.  This is actually a correct case.
         } else {
            Error("Branch", "The actual class (%s) of the object provided for the definition of the branch \"%s\" does not inherit from %s", 
                  actualClass->GetName(), branchname, claim->GetName());
         }
      }
   }
   if (claim->GetCollectionProxy() && dynamic_cast<TEmulatedCollectionProxy*>(claim->GetCollectionProxy())) {
      Error("Branch", "The class requested (%s) for the branch \"%s\" refer to an stl collection and do not have a compiled CollectionProxy.  "
            "Please generate the dictionary for this class (%s)", 
            claim->GetName(), branchname, claim->GetName());
      return 0;
   }
   return Branch(branchname, classname, (void*) addobj, bufsize, splitlevel);
}

//______________________________________________________________________________
TBranch* TTree::BranchImp(const char* branchname, TClass* ptrClass, void* addobj, Int_t bufsize, Int_t splitlevel)
{
   // -- Same as TTree::Branch but automatic detection of the class name.
   // See TTree::Branch for other details.

   if (!ptrClass) {
      Error("Branch", "The pointer specified for %s is not of a class known to ROOT", branchname);
      return 0;
   }
   TClass* actualClass = 0;
   void** addr = (void**) addobj;
   if (addr && *addr) {
      actualClass = ptrClass->GetActualClass(*addr);
      if (!actualClass) {
         Warning("Branch", "The actual TClass corresponding to the object provided for the definition of the branch \"%s\" is missing.\n\tThe object will be truncated down to its %s part", branchname, ptrClass->GetName());
         actualClass = ptrClass;
      } else if ((ptrClass != actualClass) && !actualClass->InheritsFrom(ptrClass)) {
         Error("Branch", "The actual class (%s) of the object provided for the definition of the branch \"%s\" does not inherit from %s", actualClass->GetName(), branchname, ptrClass->GetName());
         return 0;
      }
   } else {
      actualClass = ptrClass;
   }
   if (actualClass && actualClass->GetCollectionProxy() && dynamic_cast<TEmulatedCollectionProxy*>(actualClass->GetCollectionProxy())) {
      Error("Branch", "The class requested (%s) for the branch \"%s\" refer to an stl collection and do not have a compiled CollectionProxy.  "
            "Please generate the dictionary for this class (%s)",
            actualClass->GetName(), branchname, actualClass->GetName());   
      return 0;
   }
   return Branch(branchname, actualClass->GetName(), (void*) addobj, bufsize, splitlevel);
}

//______________________________________________________________________________
Int_t TTree::Branch(TList* li, Int_t bufsize /* = 32000 */ , Int_t splitlevel /* = 99 */)
{
   // -- Deprecated function. Use next function instead.
   return Branch((TCollection*) li, bufsize, splitlevel);
}

//______________________________________________________________________________
Int_t TTree::Branch(TCollection* li, Int_t bufsize /* = 32000 */, Int_t splitlevel /* = 99 */, const char* name /* = "" */)
{
   // -- Create one branch for each element in the collection.
   //
   //   Each entry in the collection becomes a top level branch if the
   //   corresponding class is not a collection. If it is a collection, the entry
   //   in the collection becomes in turn top level branches, etc.
   //   The splitlevel is decreased by 1 everytime a new collection is found.
   //   For example if list is a TObjArray*
   //     - if splitlevel = 1, one top level branch is created for each element
   //        of the TObjArray.
   //     - if splitlevel = 2, one top level branch is created for each array element.
   //       if, in turn, one of the array elements is a TCollection, one top level
   //       branch will be created for each element of this collection.
   //
   //   In case a collection element is a TClonesArray, the special Tree constructor
   //   for TClonesArray is called.
   //   The collection itself cannot be a TClonesArray.
   //
   //   The function returns the total number of branches created.
   //
   //   If name is given, all branch names will be prefixed with name_.
   //
   // IMPORTANT NOTE1: This function should not be called with splitlevel < 1.
   //
   // IMPORTANT NOTE2: The branches created by this function will have names
   // corresponding to the collection or object names. It is important
   // to give names to collections to avoid misleading branch names or
   // identical branch names. By default collections have a name equal to
   // the corresponding class name, eg the default name for a TList is "TList".
   //
   // Example--------------------------------------------------------------:
   /*
   {
         TTree T("T","test list");
         TList *l = new TList();

         TObjArray *a1 = new TObjArray();
         a1->SetName("a1");
         l->Add(a1);
         TH1F *ha1a = new TH1F("ha1a","ha1",100,0,1);
         TH1F *ha1b = new TH1F("ha1b","ha1",100,0,1);
         a1->Add(ha1a);
         a1->Add(ha1b);
         TObjArray *b1 = new TObjArray();
         b1->SetName("b1");
         l->Add(b1);
         TH1F *hb1a = new TH1F("hb1a","hb1",100,0,1);
         TH1F *hb1b = new TH1F("hb1b","hb1",100,0,1);
         b1->Add(hb1a);
         b1->Add(hb1b);

         TObjArray *a2 = new TObjArray();
         a2->SetName("a2");
         l->Add(a2);
         TH1S *ha2a = new TH1S("ha2a","ha2",100,0,1);
         TH1S *ha2b = new TH1S("ha2b","ha2",100,0,1);
         a2->Add(ha2a);
         a2->Add(ha2b);

         T.Branch(l,16000,2);
         T.Print();
   }
   */
   //----------------------------------------------------------------------

   if (!li) {
      return 0;
   }
   TObject* obj = 0;
   Int_t nbranches = GetListOfBranches()->GetEntries();
   if (li->InheritsFrom(TClonesArray::Class())) {
      Error("Branch", "Cannot call this constructor for a TClonesArray");
      return 0;
   }
   Int_t nch = strlen(name);
   TString branchname;
   TIter next(li);
   while ((obj = next())) {
      if ((splitlevel > 1) &&  obj->InheritsFrom(TCollection::Class()) && !obj->InheritsFrom(TClonesArray::Class())) {
         TCollection* col = (TCollection*) obj;
         if (nch) {
            branchname.Form("%s_%s_", name, col->GetName());
         } else {
            branchname.Form("%s_", col->GetName());
         }
         Branch(col, bufsize, splitlevel - 1, branchname);
      } else {
         if (nch && (name[nch-1] == '_')) {
            branchname.Form("%s%s", name, obj->GetName());
         } else {
            if (nch) {
               branchname.Form("%s_%s", name, obj->GetName());
            } else {
               branchname.Form("%s", obj->GetName());
            }
         }
         if (splitlevel > 99) {
            branchname += ".";
         }
         Bronch(branchname, obj->ClassName(), li->GetObjectRef(obj), bufsize, splitlevel - 1);
      }
   }
   return GetListOfBranches()->GetEntries() - nbranches;
}

//______________________________________________________________________________
Int_t TTree::Branch(const char* foldername, Int_t bufsize /* = 32000 */, Int_t splitlevel /* = 99 */)
{
   // -- Create one branch for each element in the folder.
   // Returns the total number of branches created.

   TObject* ob = gROOT->FindObjectAny(foldername);
   if (!ob) {
      return 0;
   }
   if (ob->IsA() != TFolder::Class()) {
      return 0;
   }
   Int_t nbranches = GetListOfBranches()->GetEntries();
   TFolder* folder = (TFolder*) ob;
   TIter next(folder->GetListOfFolders());
   TObject* obj = 0;
   char* curname = new char[1000];
   char occur[20];
   while ((obj = next())) {
      sprintf(curname, "%s/%s", foldername, obj->GetName());
      if (obj->IsA() == TFolder::Class()) {
         Branch(curname, bufsize, splitlevel - 1);
      } else {
         void* add = (void*) folder->GetListOfFolders()->GetObjectRef(obj);
         for (Int_t i = 0; i < 1000; ++i) {
            if (curname[i] == 0) {
               break;
            }
            if (curname[i] == '/') {
               curname[i] = '.';
            }
         }
         Int_t noccur = folder->Occurence(obj);
         if (noccur > 0) {
            sprintf(occur, "_%d", noccur);
            strcat(curname, occur);
         }
         TBranchElement* br = (TBranchElement*) Bronch(curname, obj->ClassName(), add, bufsize, splitlevel - 1);
         br->SetBranchFolder();
      }
   }
   delete[] curname;
   return GetListOfBranches()->GetEntries() - nbranches;
}

//______________________________________________________________________________
TBranch* TTree::Branch(const char* name, void* address, const char* leaflist, Int_t bufsize /* = 32000 */)
{
   // -- Create a new TTree Branch.
   //
   //     This Branch constructor is provided to support non-objects in
   //     a Tree. The variables described in leaflist may be simple variables
   //     or structures.
   //    See the two following constructors for writing objects in a Tree.
   //
   //    By default the branch buffers are stored in the same file as the Tree.
   //    use TBranch::SetFile to specify a different file
   //
   //       * address is the address of the first item of a structure
   //         or the address of a pointer to an object (see example).
   //       * leaflist is the concatenation of all the variable names and types
   //         separated by a colon character :
   //         The variable name and the variable type are separated by a slash (/).
   //         The variable type may be 0,1 or 2 characters. If no type is given,
   //         the type of the variable is assumed to be the same as the previous
   //         variable. If the first variable does not have a type, it is assumed
   //         of type F by default. The list of currently supported types is given below:
   //            - C : a character string terminated by the 0 character
   //            - B : an 8 bit signed integer (Char_t)
   //            - b : an 8 bit unsigned integer (UChar_t)
   //            - S : a 16 bit signed integer (Short_t)
   //            - s : a 16 bit unsigned integer (UShort_t)
   //            - I : a 32 bit signed integer (Int_t)
   //            - i : a 32 bit unsigned integer (UInt_t)
   //            - F : a 32 bit floating point (Float_t)
   //            - D : a 64 bit floating point (Double_t)
   //            - L : a 64 bit signed integer (Long64_t)
   //            - l : a 64 bit unsigned integer (ULong64_t)
   //            - O : a boolean (Bool_t)
   //
   //         By default, a variable will be copied to the buffer with the number of
   //         bytes specified in the type descriptor character. However, if the type
   //         consists of 2 characters, the second character is an integer that
   //         specifies the number of bytes to be used when copying the variable
   //         to the output buffer. Example:
   //             X         ; variable X, type Float_t
   //             Y/I       : variable Y, type Int_t
   //             Y/I2      ; variable Y, type Int_t converted to a 16 bits integer
   //
   //
   //       * bufsize is the buffer size in bytes for this branch
   //         The default value is 32000 bytes and should be ok for most cases.
   //         You can specify a larger value (eg 256000) if your Tree is not split
   //         and each entry is large (Megabytes)
   //         A small value for bufsize is optimum if you intend to access
   //         the entries in the Tree randomly and your Tree is in split mode.

   gTree = this;
   TBranch* branch = new TBranch(name, address, leaflist, bufsize);
   if (branch->IsZombie()) {
      delete branch;
      branch = 0;
      return 0;
   }
   fBranches.Add(branch);
   return branch;
}

//______________________________________________________________________________
TBranch* TTree::Branch(const char* name, const char* classname, void* addobj, Int_t bufsize /* = 32000 */, Int_t splitlevel /* = 99 */)
{
   // create a new branch with the object of class classname at address addobj.
   //
   // WARNING:
   // Starting with Root version 3.01, the Branch function uses the new style
   // branches (TBranchElement). To get the old behaviour, you can:
   //   - call BranchOld or
   //   - call TTree::SetBranchStyle(0)
   //
   // Note that with the new style, classname does not need to derive from TObject.
   // It must derived from TObject if the branch style has been set to 0 (old)
   //
   // Use splitlevel < 0 instead of splitlevel=0 when the class
   // has a custom Streamer
   //
   // Note: if the split level is set to the default (99),  TTree::Branch will
   // not issue a warning if the class can not be split.

   if (fgBranchStyle == 1) {
      return Bronch(name, classname, addobj, bufsize, splitlevel);
   } else {
      if (splitlevel < 0) {
         splitlevel = 0;
      }
      return BranchOld(name, classname, addobj, bufsize, splitlevel);
   }
}

//______________________________________________________________________________
TBranch* TTree::BranchOld(const char* name, const char* classname, void* addobj, Int_t bufsize /* = 32000 */, Int_t splitlevel /* = 1 */)
{
   // -- Create a new TTree BranchObject.
   //
   //    Build a TBranchObject for an object of class classname.
   //    addobj is the address of a pointer to an object of class classname.
   //    IMPORTANT: classname must derive from TObject.
   //    The class dictionary must be available (ClassDef in class header).
   //
   //    This option requires access to the library where the corresponding class
   //    is defined. Accessing one single data member in the object implies
   //    reading the full object.
   //    See the next Branch constructor for a more efficient storage
   //    in case the entry consists of arrays of identical objects.
   //
   //    By default the branch buffers are stored in the same file as the Tree.
   //    use TBranch::SetFile to specify a different file
   //
   //      IMPORTANT NOTE about branch names
   //    In case two or more master branches contain subbranches with
   //    identical names, one must add a "." (dot) character at the end
   //    of the master branch name. This will force the name of the subbranch
   //    to be master.subbranch instead of simply subbranch.
   //    This situation happens when the top level object (say event)
   //    has two or more members referencing the same class.
   //    For example, if a Tree has two branches B1 and B2 corresponding
   //    to objects of the same class MyClass, one can do:
   //       tree.Branch("B1.","MyClass",&b1,8000,1);
   //       tree.Branch("B2.","MyClass",&b2,8000,1);
   //    if MyClass has 3 members a,b,c, the two instructions above will generate
   //    subbranches called B1.a, B1.b ,B1.c, B2.a, B2.b, B2.c
   //
   //    bufsize is the buffer size in bytes for this branch
   //    The default value is 32000 bytes and should be ok for most cases.
   //    You can specify a larger value (eg 256000) if your Tree is not split
   //    and each entry is large (Megabytes)
   //    A small value for bufsize is optimum if you intend to access
   //    the entries in the Tree randomly and your Tree is in split mode.

   gTree = this;
   TClass* cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("BranchOld", "Cannot find class: '%s'", classname);
      return 0;
   }
   TBranch* branch = new TBranchObject(name, classname, addobj, bufsize, splitlevel);
   fBranches.Add(branch);
   if (!splitlevel) {
      return branch;
   }
   // We are going to fully split the class now.
   TObjArray* blist = branch->GetListOfBranches();
   const char* rdname = 0;
   const char* dname = 0;
   TString branchname;
   char** apointer = (char**) addobj;
   TObject* obj = (TObject*) *apointer;
   Bool_t delobj = kFALSE;
   if (!obj) {
      obj = (TObject*) cl->New();
      delobj = kTRUE;
   }
   // Build the StreamerInfo if first time for the class.
   BuildStreamerInfo(cl, obj);
   // Loop on all public data members of the class and its base classes.
   Int_t lenName = strlen(name);
   Int_t isDot = 0;
   if (name[lenName-1] == '.') {
      isDot = 1;
   }
   TBranch* branch1 = 0;
   TRealData* rd = 0;
   TRealData* rdi = 0;
   TIter nexti(cl->GetListOfRealData());
   TIter next(cl->GetListOfRealData());
   // Note: This loop results in a full split because the
   //       real data list includes all data members of
   //       data members.
   while ((rd = (TRealData*) next())) {
      // -- Loop over all data members creating branches for each one.
      TDataMember* dm = rd->GetDataMember();
      if (!dm->IsPersistent()) {
         // -- Do not process members with an "!" as the first character in the comment field.
         continue;
      }
      if (rd->IsObject()) {
         // -- We skip data members of class type.
         // But we do build their real data, their
         // streamer info, and write their streamer
         // info to the current directory's file.
         // Oh yes, and we also do this for all of
         // their base classes.
         TClass* clm = gROOT->GetClass(dm->GetFullTypeName());
         if (clm) {
            BuildStreamerInfo(clm, (char*) obj + rd->GetThisOffset());
         }
         continue;
      }
      rdname = rd->GetName();
      dname = dm->GetName();
      if (cl->CanIgnoreTObjectStreamer()) {
         // -- Skip the TObject base class data members.
         // FIXME: This prevents a user from ever
         //        using these names himself!
         if (!strcmp(dname, "fBits")) {
            continue;
         }
         if (!strcmp(dname, "fUniqueID")) {
            continue;
         }
      }
      TDataType* dtype = dm->GetDataType();
      Int_t code = 0;
      if (dtype) {
         code = dm->GetDataType()->GetType();
      }
      // Encode branch name. Use real data member name
      branchname = rdname;
      if (isDot) {
         if (dm->IsaPointer()) {
            // FIXME: This is wrong!  The asterisk is not usually in the front!
            branchname.Form("%s%s", name, &rdname[1]);
         } else {
            branchname.Form("%s%s", name, &rdname[0]);
         }
      }
      // FIXME: Change this to a string stream.
      TString leaflist;
      Int_t offset = rd->GetThisOffset();
      char* pointer = ((char*) obj) + offset;
      if (dm->IsaPointer()) {
         // -- We have a pointer to an object or a pointer to an array of basic types.
         TClass* clobj = 0;
         if (!dm->IsBasic()) {
            clobj = gROOT->GetClass(dm->GetTypeName());
         }
         if (clobj && clobj->InheritsFrom("TClonesArray")) {
            // -- We have a pointer to a clones array.
            char* cpointer = (char*) pointer;
            char** ppointer = (char**) cpointer;
            TClonesArray* li = (TClonesArray*) *ppointer;
            if (splitlevel != 2) {
               if (isDot) {
                  branch1 = new TBranchClones(branchname, pointer, bufsize);
               } else {
                  // FIXME: This is wrong!  The asterisk is not usually in the front!
                  branch1 = new TBranchClones(&branchname.Data()[1], pointer, bufsize);
               }
               blist->Add(branch1);
            } else {
               if (isDot) {
                  branch1 = new TBranchObject(branchname, li->ClassName(), pointer, bufsize);
               } else {
                  // FIXME: This is wrong!  The asterisk is not usually in the front!
                  branch1 = new TBranchObject(&branchname.Data()[1], li->ClassName(), pointer, bufsize);
               }
               blist->Add(branch1);
            }
         } else if (clobj) {
            // -- We have a pointer to an object.
            //
            // It must be a TObject object.
            if (!clobj->InheritsFrom(TObject::Class())) {
               continue;
            }
            branch1 = new TBranchObject(dname, clobj->GetName(), pointer, bufsize, 0);
            if (isDot) {
               branch1->SetName(branchname);
            } else {
               // FIXME: This is wrong!  The asterisk is not usually in the front!
               // -- Do not use the first character (*).
               branch1->SetName(&branchname.Data()[1]);
            }
            blist->Add(branch1);
         } else {
            // -- We have a pointer to an array of basic types.
            //
            // Check the comments in the text of the code for an index specification.
            const char* index = dm->GetArrayIndex();
            if (strlen(index) != 0) {
               // -- We are a pointer to a varying length array of basic types.
               //check that index is a valid data member name
               //if member is part of an object (eg fA and index=fN)
               //index must be changed from fN to fA.fN
               TString aindex (rd->GetName());
               Ssiz_t rdot = aindex.Last('.');
               if (rdot>=0) {
                  aindex.Remove(rdot+1);
                  aindex.Append(index);
               }
               nexti.Reset();
               while ((rdi = (TRealData*) nexti())) {
                  if (!strcmp(rdi->GetName(), index)) {
                     break;
                  }
                  if (!strcmp(rdi->GetName(), aindex)) {
                     index = rdi->GetName();
                     break;
                  }
               }
               if (code == 1) {
                  // Note that we differentiate between strings and
                  // char array by the fact that there is NO specified
                  // size for a string (see next if (code == 1)
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "B");
               } else if (code == 11) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "b");
               } else if (code == 18) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "O");
               } else if (code == 2) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "S");
               } else if (code == 12) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "s");
               } else if (code == 3) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "I");
               } else if (code == 13) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "i");
               } else if (code == 5) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "F");
               } else if ((code == 8) || (code == 9)) {
                  leaflist.Form("%s[%s]/%s", &rdname[0], index, "D");
               } else {
                  Error("BranchOld", "Cannot create branch for rdname: %s code: %d", branchname.Data(), code);
                  leaflist = "";
               }
            } else {
               // -- We are possibly a character string.
               if (code == 1) {
                  // -- We are a character string.
                  leaflist.Form("%s/%s", dname, "C");
               } else {
                  // -- Invalid array specification.
                  // FIXME: We need an error message here.
                  continue;
               }
            }
            // There are '*' in both the branchname and leaflist, remove them.
            TString bname( branchname );
            bname.ReplaceAll("*","");
            leaflist.ReplaceAll("*","");
            // Add the branch to the tree and indicate that the address
            // is that of a pointer to be dereferenced before using.
            branch1 = new TBranch(bname, *((void**) pointer), leaflist, bufsize);
            TLeaf* leaf = (TLeaf*) branch1->GetListOfLeaves()->At(0);
            leaf->SetBit(TLeaf::kIndirectAddress);
            leaf->SetAddress((void**) pointer);
            blist->Add(branch1);
         }
      } else if (dm->IsBasic()) {
         // -- We have a basic type.
         if (code == 1) {
            leaflist.Form("%s/%s", rdname, "B");
         } else if (code == 11) {
            leaflist.Form("%s/%s", rdname, "b");
         } else if (code == 18) {
            leaflist.Form("%s/%s", rdname, "O");
         } else if (code == 2) {
            leaflist.Form("%s/%s", rdname, "S");
         } else if (code == 12) {
            leaflist.Form("%s/%s", rdname, "s");
         } else if (code == 3) {
            leaflist.Form("%s/%s", rdname, "I");
         } else if (code == 13) {
            leaflist.Form("%s/%s", rdname, "i");
         } else if (code == 5) {
            leaflist.Form("%s/%s", rdname, "F");
         } else if (code == 8) {
            leaflist.Form("%s/%s", rdname, "D");
         } else if (code == 9) {
            leaflist.Form("%s/%s", rdname, "D");
         } else {
            Error("BranchOld", "Cannot create branch for rdname: %s code: %d", branchname.Data(), code);
            leaflist = "";
         }
         branch1 = new TBranch(branchname, pointer, leaflist, bufsize);
         branch1->SetTitle(rdname);
         blist->Add(branch1);
      } else {
         // -- We have a class type.
         // Note: This cannot happen due to the rd->IsObject() test above.
         // FIXME: Put an error message here just in case.
      }
      if (branch1) {
         branch1->SetOffset(offset);
      } else {
         Warning("BranchOld", "Cannot process member: '%s'", rdname);
      }
   }
   if (delobj) {
      delete obj;
      obj = 0;
   }
   return branch;
}

//______________________________________________________________________________
TBranch* TTree::BranchRef()
{
   // -- Build the optional branch supporting the TRefTable.
   // This branch will keep all the information to find the branches
   // containing referenced objects.
   //
   // At each Tree::Fill, the branch numbers containing the
   // referenced objects are saved to the TBranchRef basket.
   // When the Tree header is saved (via TTree::Write), the branch
   // is saved keeping the information with the pointers to the branches
   // having referenced objects.

   if (!fBranchRef) {
      fBranchRef = new TBranchRef(this);
   }
   return fBranchRef;
}

//______________________________________________________________________________
TBranch* TTree::Bronch(const char* name, const char* classname, void* add, Int_t bufsize /* = 32000 */, Int_t splitlevel /* = 99 */)
{
   // -- Create a new TTree BranchElement.
   //
   //    WARNING about this new function
   //    ===============================
   //    This function is designed to replace the function TTree::Branch above.
   //    This function is far more powerful than the Branch function.
   //    It supports the full C++, including STL and has the same behaviour
   //    in split or non-split mode. classname does not have to derive from TObject.
   //    The function is based on the new TStreamerInfo.
   //
   //    Build a TBranchElement for an object of class classname.
   //    addobj is the address of a pointer to an object of class classname.
   //    The class dictionary must be available (ClassDef in class header).
   //
   //    This option requires access to the library where the corresponding class
   //    is defined. Accessing one single data member in the object implies
   //    reading the full object.
   //
   //    By default the branch buffers are stored in the same file as the Tree.
   //    use TBranch::SetFile to specify a different file
   //
   //      IMPORTANT NOTE about branch names
   //    In case two or more master branches contain subbranches with
   //    identical names, one must add a "." (dot) character at the end
   //    of the master branch name. This will force the name of the subbranch
   //    to be master.subbranch instead of simply subbranch.
   //    This situation happens when the top level object (say event)
   //    has two or more members referencing the same class.
   //    For example, if a Tree has two branches B1 and B2 corresponding
   //    to objects of the same class MyClass, one can do:
   //       tree.Branch("B1.","MyClass",&b1,8000,1);
   //       tree.Branch("B2.","MyClass",&b2,8000,1);
   //    if MyClass has 3 members a,b,c, the two instructions above will generate
   //    subbranches called B1.a, B1.b ,B1.c, B2.a, B2.b, B2.c
   //
   //    bufsize is the buffer size in bytes for this branch
   //    The default value is 32000 bytes and should be ok for most cases.
   //    You can specify a larger value (eg 256000) if your Tree is not split
   //    and each entry is large (Megabytes)
   //    A small value for bufsize is optimum if you intend to access
   //    the entries in the Tree randomly and your Tree is in split mode.
   //
   //    Use splitlevel < 0 instead of splitlevel=0 when the class
   //    has a custom Streamer
   //
   //    Note: if the split level is set to the default (99),  TTree::Branch will
   //    not issue a warning if the class can not be split.

   gTree = this;
   TClass* cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("Bronch", "Cannot find class:%s", classname);
      return 0;
   }

   //if splitlevel <= 0 and class has a custom Streamer, we must create
   //a TBranchObject. We cannot assume that TClass::ReadBuffer is consistent
   //with the custom Streamer. The penalty is that one cannot process
   //this Tree without the class library containing the class.
   //The following convention is used for the RootFlag
   // #pragma link C++ class TExMap;     rootflag = 0
   // #pragma link C++ class TList-;     rootflag = 1
   // #pragma link C++ class TArray!;    rootflag = 2
   // #pragma link C++ class TArrayC-!;  rootflag = 3
   // #pragma link C++ class TBits+;     rootflag = 4
   // #pragma link C++ class Txxxx+!;    rootflag = 6

   char** ppointer = (char**) add;

   if (cl == TClonesArray::Class()) {
      TClonesArray* clones = (TClonesArray*) *ppointer;
      if (!clones) {
         Error("Bronch", "Pointer to TClonesArray is null");
         return 0;
      }
      if (!clones->GetClass()) {
         Error("Bronch", "TClonesArray with no class defined in branch: %s", name);
         return 0;
      }
      G__ClassInfo* classinfo = clones->GetClass()->GetClassInfo();
      if (!classinfo) {
         Error("Bronch", "TClonesArray with no dictionary defined in branch: %s", name);
         return 0;
      }
      if (splitlevel > 0) {
         if (classinfo->RootFlag() & 1)
            Warning("Bronch", "Using split mode on a class: %s with a custom Streamer", clones->GetClass()->GetName());
      } else {
         if (classinfo->RootFlag() & 1) clones->BypassStreamer(kFALSE);
         TBranchObject *branch = new TBranchObject(name,classname,add,bufsize,0);
         fBranches.Add(branch);
         return branch;
      }
   }

   if (cl->GetCollectionProxy()) {
      TVirtualCollectionProxy* collProxy = cl->GetCollectionProxy();
      //if (!collProxy) {
      //   Error("Bronch", "%s is missing its CollectionProxy (for branch %s)", classname, name);
      //}
      TClass* inklass = collProxy->GetValueClass();
      if (!inklass && (collProxy->GetType() == 0)) {
         Error("Bronch", "%s with no class defined in branch: %s", classname, name);
         return 0;
      }
      if ((splitlevel > 0) && inklass && (inklass->GetCollectionProxy() == 0)) {
         Int_t stl = -TClassEdit::IsSTLCont(cl->GetName(), 0);
         if ((stl != TClassEdit::kMap) && (stl != TClassEdit::kMultiMap)) {
            G__ClassInfo* classinfo = inklass->GetClassInfo();
            if (!classinfo) {
               Error("Bronch", "Container with no dictionary defined in branch: %s", name);
               return 0;
            }
            if (classinfo->RootFlag() & 1) {
               Warning("Bronch", "Using split mode on a class: %s with a custom Streamer", inklass->GetName());
            }
         }
      }
      TBranchElement* branch = new TBranchElement(name, collProxy, bufsize, splitlevel);
      fBranches.Add(branch);
      branch->SetAddress(add);
      return branch;
   }

   Bool_t hasCustomStreamer = kFALSE;
   if (!cl->GetClassInfo() && !cl->GetCollectionProxy()) {
      Error("Bronch", "Cannot find dictionary for class: %s", classname);
      return 0;
   }

   if (!cl->GetCollectionProxy() && (cl->GetClassInfo()->RootFlag() & 1)) {
      // Not an STL container and the linkdef file had a "-" after the class name.
      hasCustomStreamer = kTRUE;
   }

   if (splitlevel < 0 || ((splitlevel == 0) && hasCustomStreamer && cl->InheritsFrom(TObject::Class()))) {
      TBranchObject* branch = new TBranchObject(name, classname, add, bufsize, 0);
      fBranches.Add(branch);
      return branch;
   }

   if (cl == TClonesArray::Class()) {
      // -- Special case of TClonesArray.
      // No dummy object is created.
      // The streamer info is not rebuilt unoptimized.
      // No dummy top-level branch is created.
      // No splitting is attempted.
      TBranchElement* branch = new TBranchElement(name, (TClonesArray*) *ppointer, bufsize, splitlevel);
      fBranches.Add(branch);
      branch->SetAddress(add);
      return branch;
   }

   //
   // If we are not given an object to use as an i/o buffer
   // then create a temporary one which we will delete just
   // before returning.
   //

   Bool_t delobj = kFALSE;

   if (!(*ppointer)) {
      *ppointer = (char*) cl->New();
      delobj = kTRUE;
   }

   //
   // Avoid splitting unsplittable classes.
   //

   if ((splitlevel > 0) && !cl->CanSplit()) {
      splitlevel = 0;
      if (splitlevel != 99) {
         Warning("Bronch", "%s cannot be split, resetting splitlevel to 0", cl->GetName());
      }
   }

   //
   // Make sure the streamer info is built and fetch it.
   //
   // If we are splitting, then make sure the streamer info
   // is built unoptimized (data members are not combined).
   //

   Bool_t optim = TStreamerInfo::CanOptimize();
   if (splitlevel > 0) {
      TStreamerInfo::Optimize(kFALSE);
   }
   TStreamerInfo* sinfo = BuildStreamerInfo(cl, *ppointer);
   TStreamerInfo::Optimize(optim);

   //
   // Do we have a final dot in our name?
   //

   // Note: The branch constructor which takes a folder as input
   //       creates top-level branch names with dots in them to
   //       indicate the folder heirarchy.
   char* dot = (char*) strchr(name, '.');
   Int_t nch = strlen(name);
   Bool_t dotlast = kFALSE;
   if (nch && (name[nch-1] == '.')) {
      dotlast = kTRUE;
   }

   //
   // Create a dummy top level branch object.
   //

   Int_t id = -1;
   if (splitlevel > 0) {
      id = -2;
   }
   TBranchElement* branch = new TBranchElement(name, sinfo, id, *ppointer, bufsize, splitlevel);
   fBranches.Add(branch);

   //
   // Do splitting, if requested.
   //

   if (splitlevel > 0) {
      // Loop on all public data members of the class and its base classes and create branches for each one.
      TObjArray* blist = branch->GetListOfBranches();
      TIter next(sinfo->GetElements());
      TStreamerElement* element = 0;
      TString bname;
      for (id = 0; (element = (TStreamerElement*) next()); ++id) {
         char* pointer = (char*) (*ppointer + element->GetOffset());
         // FIXME: This is not good enough, an STL container can be
         //        a base, and the test will fail.
         //        See TBranchElement::InitializeOffsets() for the
         //        correct test.
         Bool_t isBase = (element->IsA() == TStreamerBase::Class());
         if (isBase) {
            TClass* clbase = element->GetClassPointer();
            if ((clbase == TObject::Class()) && cl->CanIgnoreTObjectStreamer()) {
               // Note: TStreamerInfo::Compile() leaves this element
               //       out of the compiled info, although it does
               //       exists in the non-compiled info.  We must
               //       account for the fact that this element is
               //       missing in the compiled streamer info by
               //       making sure that we do not consume an id
               //       number for it.
               // FIXME: The test that TStreamerInfo::Compile() uses
               //        is element->GetType() < 0, so that is what
               //        we should do as well.
               --id;
               continue;
            }
         }
         if (dot) {
            if (dotlast) {
               bname.Form("%s%s", name, element->GetFullName());
            } else {
               // FIXME: We are in the case where we have a top-level
               //        branch name that was created by the branch
               //        constructor which takes a folder as input.
               //        The internal dots in the name are in place of
               //        of the original slashes and represent the
               //        folder heirarchy.
               if (isBase) {
                  // FIXME: This is very strange, this is the only case where
                  //        we create a branch for a base class that does
                  //        not have the base class name in the branch name.
                  // FIXME: This is also quite bad since classes with two
                  //        or more base classes end up with sub-branches
                  //        that have the same name.
                  bname = name;
               } else {
                  bname.Form("%s.%s", name, element->GetFullName());
               }
            }
         } else {
            // Note: For a base class element, this results in the branchname
            //       being the name of the base class.
            bname.Form("%s", element->GetFullName());
         }
         TBranchElement* bre = new TBranchElement(bname, sinfo, id, pointer, bufsize, splitlevel - 1);
         bre->SetParentClass(cl);
         blist->Add(bre);
      }
   }

   //
   // Setup our offsets into the user's i/o buffer.
   //

   branch->SetAddress(add);

   if (delobj) {
      cl->Destructor(*ppointer);
      *ppointer = 0;
   }

   return branch;
}

//______________________________________________________________________________
void TTree::Browse(TBrowser* b)
{
   // -- Browse content of the TTree.

   fBranches.Browse(b);
}

//______________________________________________________________________________
Int_t TTree::BuildIndex(const char* majorname, const char* minorname /* = "0" */)
{
   // -- Build a Tree Index (default is TTreeIndex).
   // See a description of the parameters and functionality in
   // TTreeIndex::TTreeIndex().
   //
   // The return value is the number of entries in the Index (< 0 indicates failure).
   //
   // A TTreeIndex object pointed by fTreeIndex is created.
   // This object will be automatically deleted by the TTree destructor.
   // See also comments in TTree::SetTreeIndex().

   fTreeIndex = GetPlayer()->BuildIndex(this, majorname, minorname);
   if (fTreeIndex->IsZombie()) {
      delete fTreeIndex;
      fTreeIndex = 0;
      return 0;
   }
   return fTreeIndex->GetN();
}

//______________________________________________________________________________
TStreamerInfo* TTree::BuildStreamerInfo(TClass* cl, void* pointer /* = 0 */)
{
   // -- Build StreamerInfo for class cl.
   // pointer is an optional argument that may contain a pointer to an object of cl.

   if (!cl) {
      return 0;
   }
   cl->BuildRealData(pointer);
   TStreamerInfo* sinfo = cl->GetStreamerInfo(cl->GetClassVersion());
   if (fDirectory) {
      sinfo->ForceWriteInfo(fDirectory->GetFile());
   }
   // Create StreamerInfo for all base classes.
   TBaseClass* base = 0;
   TIter nextb(cl->GetListOfBases());
   while((base = (TBaseClass*) nextb())) {
      if (base->IsSTLContainer()) {
         continue;
      }
      TClass* clm = gROOT->GetClass(base->GetName());
      BuildStreamerInfo(clm);
   }
   return sinfo;
}

//______________________________________________________________________________
TFile* TTree::ChangeFile(TFile* file)
{
   // -- Called by TTree::Fill() when file has reached its maximum fgMaxTreeSize.
   // Create a new file. If the original file is named "myfile.root",
   // subsequent files are named "myfile_1.root", "myfile_2.root", etc.
   //
   // Returns a pointer to the new file.
   //
   // Currently, the automatic change of file is restricted
   // to the case where the tree is in the top level directory.
   // The file should not contain sub-directories.
   //
   // Before switching to a new file, the tree header is written
   // to the current file, then the current file is closed.
   //
   // To process the multiple files created by ChangeFile, one must use
   // a TChain.
   //
   // The new file name has a suffix "_N" where N is equal to fFileNumber+1.
   // By default a Root session starts with fFileNumber=0. One can set
   // fFileNumber to a different value via TTree::SetFileNumber.
   // In case a file named "_N" already exists, the function will try
   // a file named "__N", then "___N", etc.
   //
   // fgMaxTreeSize can be set via the static function TTree::SetMaxTreeSize.
   // The default value of fgMaxTreeSize is 1.9 Gigabytes.
   //
   // If the current file contains other objects like TH1 and TTree,
   // these objects are automatically moved to the new file.
   //
   // IMPORTANT NOTE:
   // Be careful when writing the final Tree header to the file!
   // Don't do:
   //  TFile *file = new TFile("myfile.root","recreate");
   //  TTree *T = new TTree("T","title");
   //  T->Fill(); //loop
   //  file->Write();
   //  file->Close();
   // but do the following:
   //  TFile *file = new TFile("myfile.root","recreate");
   //  TTree *T = new TTree("T","title");
   //  T->Fill(); //loop
   //  file = T->GetCurrentFile(); //to get the pointer to the current file
   //  file->Write();
   //  file->Close();

   file->cd();
   Write();
   Reset();
   char* fname = new char[2000];
   ++fFileNumber;
   char uscore[10];
   for (Int_t i = 0; i < 10; ++i) {
      uscore[i] = 0;
   }
   Int_t nus = 0;
   // Try to find a suitable file name that does not already exist.
   while (nus < 10) {
      uscore[nus] = '_';
      fname[0] = 0;
      strcpy(fname, file->GetName());
      if (fFileNumber > 1) {
         char* cunder = strrchr(fname, '_');
         if (cunder) {
            sprintf(cunder, "%s%d", uscore, fFileNumber);
            strcat(fname, strrchr(file->GetName(), '.'));
         } else {
            char fcount[10];
            sprintf(fcount, "%s%d", uscore, fFileNumber);
            strcat(fname, fcount);
         }
      } else {
         char* cdot = strrchr(fname, '.');
         if (cdot) {
            sprintf(cdot, "%s%d", uscore, fFileNumber);
            strcat(fname, strrchr(file->GetName(), '.'));
         } else {
            char fcount[10];
            sprintf(fcount, "%s%d", uscore, fFileNumber);
            strcat(fname, fcount);
         }
      }
      if (gSystem->AccessPathName(fname)) {
         break;
      }
      ++nus;
      Warning("ChangeFile", "file %s already exist, trying with %d underscores", fname, nus+1);
   }
   Int_t compress = file->GetCompressionLevel();
   TFile* newfile = TFile::Open(fname, "recreate", "chain files", compress);
   Printf("Fill: Switching to new file: %s", fname);
   // The current directory may contain histograms and trees.
   // These objects must be moved to the new file.
   TBranch* branch = 0;
   TObject* obj = 0;
   while ((obj = file->GetList()->First())) {
      file->GetList()->Remove(obj);
      // Histogram: just change the directory.
      if (obj->InheritsFrom("TH1")) {
         gROOT->ProcessLine(Form("((%s*)0x%lx)->SetDirectory((TDirectory*)0x%lx);", obj->ClassName(), (Long_t) obj, (Long_t) newfile));
         continue;
      }
      // Tree: must save all trees in the old file, reset them.
      if (obj->InheritsFrom("TTree")) {
         TTree* t = (TTree*) obj;
         if (t != this) {
            t->AutoSave();
            t->Reset();
            t->fFileNumber = fFileNumber;
         }
         t->SetDirectory(newfile);
         TIter nextb(t->GetListOfBranches());
         while ((branch = (TBranch*)nextb())) {
            branch->SetFile(newfile);
         }
         if (t->GetBranchRef()) {
            t->GetBranchRef()->SetFile(newfile);
         }
         continue;
      }
      // Not a TH1 or a TTree, move object to new file.
      newfile->GetList()->Add(obj);
      file->GetList()->Remove(obj);
   }
   delete file;
   file = 0;
   gFile = newfile;
   delete[] fname;
   fname = 0;
   return newfile;
}

//______________________________________________________________________________
Bool_t TTree::CheckBranchAddressType(TBranch* branch, TClass* ptrClass, EDataType datatype, Bool_t isptr)
{
   // -- Check whether or not the address described by the last 3 parameters matches the content of the branch.

   // Let's determine what we need!
   TClass* expectedClass = 0;
   EDataType expectedType = kOther_t;
   if (branch->InheritsFrom(TBranchObject::Class())) {
      TLeafObject* lobj = (TLeafObject*) branch->GetListOfLeaves()->At(0);
      expectedClass = lobj->GetClass();
   } else if (branch->InheritsFrom(TBranchElement::Class())) {
      TBranchElement* branchEl = (TBranchElement*) branch;
      Int_t type = branchEl->GetStreamerType();
      if ((type == -1) || (branchEl->GetID() == -1)) {
         expectedClass =  branchEl->GetInfo()->GetClass();
      } else {
         // Case of an object data member.  Here we allow for the
         // variable name to be ommitted.  Eg, for Event.root with split
         // level 1 or above  Draw("GetXaxis") is the same as Draw("fH.GetXaxis()")
         TStreamerElement* element = (TStreamerElement*) branchEl->GetInfo()->GetElems()[branchEl->GetID()];
         if (element) {
            expectedClass = element->GetClassPointer();
         }
         if (!expectedClass) {
            TDataType* data = gROOT->GetType(element->GetTypeNameBasic());
            if (!data) {
               Error("CheckBranchAddress", "Did not find the type number for %s", element->GetTypeNameBasic());
            } else {
               expectedType = (EDataType) data->GetType();
            }
         }
      }
      if (ptrClass && (branch->GetMother() == branch)) {
         // Top Level branch
         if (!isptr) {
            Error("SetBranchAddress", "The address for \"%s\" should be the address of a pointer!", branch->GetName());
         }
      }
   } else {
      TLeaf* l = (TLeaf*) branch->GetListOfLeaves()->At(0);
      if (l) {
         expectedType = (EDataType) gROOT->GetType(l->GetTypeName())->GetType();
      }
   }
   if (expectedType == kDouble32_t) {
      expectedType = kDouble_t;
   }
   if (datatype == kDouble32_t) {
      datatype = kDouble_t;
   }
   if (expectedClass && ptrClass && !expectedClass->InheritsFrom(ptrClass)) {
      Error("SetBranchAddress", "The pointer type give (%s) does not correspond to the class needed (%s) by the branch: %s", ptrClass->GetName(), expectedClass->GetName(), branch->GetName());
      return kFALSE;
   } else if ((expectedType != kOther_t) && (datatype != kOther_t) && (expectedType != kNoType_t) && (datatype != kNoType_t) && (expectedType != datatype)) {
      if (datatype != kChar_t) {
         // For backward compatibility we assume that (char*) was just a cast and/or a generic address
         Error("SetBranchAddress", "The pointer type given \"%s\" (%d) does not correspond to the type needed \"%s\" (%d) by the branch: %s", TDataType::GetTypeName(datatype), datatype, TDataType::GetTypeName(expectedType), expectedType, branch->GetName());
         return kFALSE;
      }
   }
   if (expectedClass && expectedClass->GetCollectionProxy() && dynamic_cast<TEmulatedCollectionProxy*>(expectedClass->GetCollectionProxy())) {
      Error("SetBranchAddress", "The class requested (%s) for the branch \"%s\" refer to an stl collection and do not have a compiled CollectionProxy.  "
            "Please generate the dictionary for this class (%s)",
            expectedClass->GetName(), branch->GetName(), expectedClass->GetName());   
   } 
   return kTRUE;
}

//______________________________________________________________________________
TTree* TTree::CloneTree(Long64_t nentries /* = -1 */, Option_t* option /* = "" */)
{
   // -- Create a clone of this tree and copy nentries.
   //
   // By default copy all entries.
   // Note that only active branches are copied.
   // The compression level of the cloned tree is set to the destination file's
   // compression level.
   //
   // IMPORTANT: The cloned tree stays connected with this tree until this tree
   //            is deleted.  In particular, any changes in branch addresses
   //            in this tree are forwarded to the clone trees, unless a branch
   //            in a clone tree has had its address changed, in which case
   //            that change stays in effect.  When this tree is deleted, all the
   //            addresses of the cloned tree are reset to their default values.
   //
   // If 'option' contains the word 'fast' and nentries is -1 and no branch
   // is disabled, the clone will be done without unzipping or unstreaming
   // the baskets (i.e., a direct copy of the raw bytes on disk).
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
   // For examples of CloneTree, see tutorials:
   //
   //  -- copytree
   //
   //     A macro to copy a subset of a TTree to a new TTree.
   //
   //     The input file has been generated by the program in $ROOTSYS/test/Event
   //     with:  Event 1000 1 1 1
   //
   //  -- copytree2
   //
   //     A macro to copy a subset of a TTree to a new TTree.
   //
   //     One branch of the new Tree is written to a separate file.
   //
   //     The input file has been generated by the program in $ROOTSYS/test/Event
   //     with:  Event 1000 1 1 1
   //

   // Options
   Bool_t fastClone = kFALSE;

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("fast")) {
      fastClone = kTRUE;
   }

   // If we are a chain, switch to the first tree.
   if ((fEntries > 0) && (LoadTree(0) < 0)) {
         // FIXME: We need an error message here.
         return 0;
   }

   // Note: For a tree we get the this pointer, for
   //       a chain we get the chain's current tree.
   TTree* thistree = GetTree();

   // Note: For a chain, the returned clone will be
   //       a clone of the chain's first tree.
   TTree* newtree = (TTree*) thistree->Clone();
   if (!newtree) {
      return 0;
   }

   // The clone should not delete any objects allocated by SetAddress().
   TObjArray* branches = newtree->GetListOfBranches();
   Int_t nb = branches->GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i) {
      TBranch* br = (TBranch*) branches->UncheckedAt(i);
      if (br->InheritsFrom("TBranchElement")) {
         ((TBranchElement*) br)->ResetDeleteObject();
      }
   }

   // Add the new tree to the list of clones so that
   // we can later inform it of changes to branch addresses.
   thistree->AddClone(newtree);

   newtree->Reset();

   TDirectory* ndir = newtree->GetDirectory();
   TFile* nfile = 0;
   if (ndir) {
      nfile = ndir->GetFile();
   }
   Int_t newcomp = -1;
   if (nfile) {
      newcomp = nfile->GetCompressionLevel();
   }

   //
   // Delete non-active branches from the clone.
   //
   // Note: If we are a chain, this does nothing
   //       since chains have no leaves.
   TObjArray* leaves = newtree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (Int_t lndx = 0; lndx < nleaves; ++lndx) {
      TLeaf* leaf = (TLeaf*) leaves->UncheckedAt(lndx);
      if (!leaf) {
         continue;
      }
      TBranch* branch = leaf->GetBranch();
      if (branch && (newcomp > -1)) {
         branch->SetCompressionLevel(newcomp);
      }
      if (!branch || !branch->TestBit(kDoNotProcess)) {
         continue;
      }
      TObjArray* branches = newtree->GetListOfBranches();
      Int_t nb = branches->GetEntriesFast();
      for (Long64_t i = 0; i < nb; ++i) {
         TBranch* br = (TBranch*) branches->UncheckedAt(i);
         if (br == branch) {
            branches->RemoveAt(i);
            delete br;
            br = 0;
            branches->Compress();
            break;
         }
         TObjArray* lb = br->GetListOfBranches();
         Int_t nb1 = lb->GetEntriesFast();
         for (Int_t j = 0; j < nb1; ++j) {
            TBranch* b1 = (TBranch*) lb->UncheckedAt(j);
            if (!b1) {
               continue;
            }
            if (b1 == branch) {
               lb->RemoveAt(j);
               delete b1;
               b1 = 0;
               lb->Compress();
               break;
            }
            TObjArray* lb1 = b1->GetListOfBranches();
            Int_t nb2 = lb1->GetEntriesFast();
            for (Int_t k = 0; k < nb2; ++k) {
               TBranch* b2 = (TBranch*) lb1->UncheckedAt(k);
               if (!b2) {
                  continue;
               }
               if (b2 == branch) {
                  lb1->RemoveAt(k);
                  delete b2;
                  b2 = 0;
                  lb1->Compress();
                  break;
               }
            }
         }
      }
   }
   leaves->Compress();

   // Copy MakeClass status.
   newtree->SetMakeClass(fMakeClass);

   // Copy branch addresses.
   CopyAddresses(newtree);

   //
   // Copy entries if requested.
   //

   if (fastClone && (nentries < 0) && (newtree->GetListOfLeaves()->GetEntries() == GetListOfLeaves()->GetEntries())) {
      // -- Quickly copy the basket without decompression and streaming.
      nentries = GetEntriesFast();
      for (Long64_t i = 0; i < nentries; i += this->GetTree()->GetEntries()) {
         if (LoadTree(i) < 0) {
            break;
         }
         TTreeCloner t(GetTree(), newtree, option);
         if (t.IsValid()) {
            newtree->SetEntries(newtree->GetEntries() + GetTree()->GetEntries());
            t.Exec();
         } else {
            if (!i) {
               Error("CloneTree", "Tree has not been cloned\n");
               delete newtree;
               newtree = 0;
               return 0;
            } else {
               if (GetCurrentFile()) {
                  Warning("CloneTree", "Skipped file %s\n", GetCurrentFile()->GetName());
               } else {
                  Warning("Merge", "Skipped file number %d\n", GetTreeNumber());
               }
            }
         }
      }
   } else {
      // -- Maybe copy some entries.
      if (nentries < 0) {
         nentries = fEntries;
      }
      if (nentries > fEntries) {
         nentries = fEntries;
      }
      for (Long64_t i = 0; i < nentries; ++i) {
         // -- Loop over specified entries and copy.
         // If we are a chain, we must switch input files as necessary.
         Long64_t localEntry = LoadTree(i);
         if (localEntry < 0) {
            // FIXME: We need an error message here.
            break;
         }
         GetEntry(i);
         newtree->Fill();
      }
   }

   return newtree;
}

//______________________________________________________________________________
void TTree::CopyAddresses(TTree* tree)
{
   // -- Set branch addresses of passed tree equal to ours.

   // Copy branch addresses starting from branches.
   TObjArray* branches = GetListOfBranches();
   Int_t nbranches = branches->GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* branch = (TBranch*) branches->UncheckedAt(i);
      if (branch->TestBit(kDoNotProcess)) {
         continue;
      }
      char* addr = branch->GetAddress();
      if (!addr) {
         // Note: This may cause an object to be allocated.
         branch->SetAddress(0);
         addr = branch->GetAddress();
      }
      // FIXME: The GetBranch() function is braindead and may
      //        not find the branch!
      TBranch* br = tree->GetBranch(branch->GetName());
      if (br) {
         br->SetAddress(addr);
         // The copy does not own any object allocated by SetAddress().
         if (br->InheritsFrom("TBranchElement")) {
            ((TBranchElement*) br)->ResetDeleteObject();
         }
      } else {
         Warning("CopyAddresses", "Could not find branch named '%s' in tree named '%s'", branch->GetName(), tree->GetName());
      }
   }

   // Copy branch addresses starting from leaves.
   TObjArray* tleaves = tree->GetListOfLeaves();
   Int_t ntleaves = tleaves->GetEntriesFast();
   for (Int_t i = 0; i < ntleaves; ++i) {
      TLeaf* tleaf = (TLeaf*) tleaves->UncheckedAt(i);
      TBranch* tbranch = tleaf->GetBranch();
      TBranch* branch = GetBranch(tbranch->GetName());
      if (!branch) {
         continue;
      }
      TLeaf* leaf = branch->GetLeaf(tleaf->GetName());
      if (!leaf) {
         continue;
      }
      if (branch->TestBit(kDoNotProcess)) {
         continue;
      }
      if (!branch->GetAddress() && !leaf->GetValuePointer()) {
         // We should attempts to set the address of the branch.
         // something like:
         //(TBranchElement*)branch->GetMother()->SetAddress(0)
         //plus a few more subtilities (see TBranchElement::GetEntry).
         //but for now we go the simpliest route:
         //
         // Note: This may result in the allocation of an object.
         branch->GetEntry(0);
      }
      if (branch->GetAddress()) {
         tree->SetBranchAddress(branch->GetName(), (void*) branch->GetAddress());
         TBranch* br = tree->GetBranch(branch->GetName());
         if (br) {
            // The copy does not own any object allocated by SetAddress().
            // FIXME: We do too much here, br may not be a top-level branch.
            if (br->InheritsFrom("TBranchElement")) {
               ((TBranchElement*) br)->ResetDeleteObject();
            }
         } else {
            Warning("CopyAddresses", "Could not find branch named '%s' in tree named '%s'", branch->GetName(), tree->GetName());
         }
      } else {
         tleaf->SetAddress(leaf->GetValuePointer());
      }
   }
}

//______________________________________________________________________________
Long64_t TTree::CopyEntries(TTree* tree, Long64_t nentries /* = -1 */)
{
   // -- Copy nentries from given tree to this tree.
   //
   // By default copy all entries.
   //
   // Returns number of bytes copied to this tree.
   //

   if (!tree) {
      return 0;
   }
   Long64_t nbytes = 0;
   Long64_t treeEntries = tree->GetEntriesFast();
   if (nentries < 0) {
      nentries = treeEntries;
   }
   if (nentries > treeEntries) {
      nentries = treeEntries;
   }
   for (Long64_t i = 0; i < nentries; ++i) {
      if (tree->LoadTree(i) < 0) {
         break;
      }
      tree->GetEntry(i);
      nbytes += Fill();
   }
   return nbytes;
}

//______________________________________________________________________________
TTree* TTree::CopyTree(const char* selection, Option_t* option /* = 0 */, Long64_t nentries /* = 1000000000 */, Long64_t firstentry /* = 0 */)
{
   // -- Copy a tree with selection.
   //
   // IMPORTANT:
   //
   //   The returned copied tree stays connected with the original tree
   //   until the original tree is deleted.  In particular, any changes
   //   to the branch addresses in the original tree are also made to
   //   the copied tree.  Any changes made to the branch addresses of the
   //   copied tree are overridden anytime the original tree changes its
   //   branch addresses.  When the original tree is deleted, all the
   //   branch addresses of the copied tree are set to zero.
   //
   // For examples of CopyTree, see the tutorials:
   //
   // -- copytree
   //
   // Example macro to copy a subset of a tree to a new tree.
   //
   // The input file was generated by running the program in
   // $ROOTSYS/test/Event in this way:
   //
   //      ./Event 1000 1 1 1
   //
   // -- copytree2
   //
   // Example macro to copy a subset of a tree to a new tree.
   //
   // One branch of the new tree is written to a separate file.
   //
   // The input file was generated by running the program in
   // $ROOTSYS/test/Event in this way:
   //
   //      ./Event 1000 1 1 1
   //
   // -- copytree3
   //
   // Example macro to copy a subset of a tree to a new tree.
   //
   // Only selected entries are copied to the new tree.
   // NOTE that only the active branches are copied.
   //

   GetPlayer();
   if (fPlayer) {
      return fPlayer->CopyTree(selection, option, nentries, firstentry);
   }
   return 0;
}

//______________________________________________________________________________
TBasket* TTree::CreateBasket(TBranch* branch)
{
   // -- Create a basket for this tree and given branch.
   if (!branch) {
      return 0;
   }
   return new TBasket(branch->GetName(), GetName(), branch);
}

//______________________________________________________________________________
void TTree::Delete(Option_t* option /* = "" */)
{
   // -- Delete this tree from memory or/and disk.
   //
   //  if option == "all" delete Tree object from memory AND from disk
   //                     all baskets on disk are deleted. All keys with same name
   //                     are deleted.
   //  if option =="" only Tree object in memory is deleted.

   TFile *file = GetCurrentFile();

   // delete all baskets and header from file
   if (file && !strcmp(option,"all")) {
      if (!file->IsWritable()) {
         Error("Delete","File : %s is not writable, cannot delete Tree:%s", file->GetName(),GetName());
         return;
      }

      //find key and import Tree header in memory
      TKey *key = fDirectory->GetKey(GetName());
      if (!key) return;

      TDirectory *dirsav = gDirectory;
      file->cd();

      //get list of leaves and loop on all the branches baskets
      TIter next(GetListOfLeaves());
      TLeaf *leaf;
      char header[16];
      Int_t ntot  = 0;
      Int_t nbask = 0;
      Int_t nbytes,objlen,keylen;
      while ((leaf = (TLeaf*)next())) {
         TBranch *branch = leaf->GetBranch();
         Int_t nbaskets = branch->GetMaxBaskets();
         for (Int_t i=0;i<nbaskets;i++) {
            Long64_t pos = branch->GetBasketSeek(i);
            if (!pos) continue;
            gFile->GetRecordHeader(header,pos,16,nbytes,objlen,keylen);
            if (nbytes <= 0) continue;
            gFile->MakeFree(pos,pos+nbytes-1);
            ntot += nbytes;
            nbask++;
         }
      }

      // delete Tree header key and all keys with the same name
      // A Tree may have been saved many times. Previous cycles are invalid.
      while (key) {
         ntot += key->GetNbytes();
         key->Delete();
         delete key;
         key = fDirectory->GetKey(GetName());
      }
      if (dirsav) dirsav->cd();
      if (gDebug) printf(" Deleting Tree: %s: %d baskets deleted. Total space freed = %d bytes\n",GetName(),nbask,ntot);
   }

   if (fDirectory) fDirectory->GetList()->Remove(this);
   fDirectory = 0;

    // Delete object from CINT symbol table so it can not be used anymore.
   gInterpreter->DeleteGlobal(this);

   delete this;
}

//______________________________________________________________________________
Long64_t TTree::Draw(const char* varexp, const TCut& selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Draw expression varexp for specified entries.
   // Returns -1 in case of error or number of selected events in case of success.
   //
   //      This function accepts TCut objects as arguments.
   //      Useful to use the string operator +
   //         example:
   //            ntuple.Draw("x",cut1+cut2+cut3);
   //

   return TTree::Draw(varexp, selection.GetTitle(), option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TTree::Draw(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Draw expression varexp for specified entries.
   // Returns -1 in case of error or number of selected events in case of success.
   //
   //  varexp is an expression of the general form
   //   - "e1"           produces a 1-d histogram (TH1F) of expression "e1"
   //   - "e1:e2"        produces an unbinned 2-d scatter-plot (TGraph) of "e1" versus "e2"
   //   - "e1:e2:e3"     produces an unbinned 3-d scatter-plot (TPolyMarker3D) of "e1"
   //                    versus "e2" versus "e3"
   //   - "e1:e2:e3:e4"  produces an unbinned 3-d scatter-plot (TPolyMarker3D) of "e1"
   //                    versus "e2" versus "e3" and "e4" mapped on the color number.
   //  (to create histograms in the 2, 3, and 4 dimesional case, see section "Saving
   //  the result of Draw to an histogram")
   //
   //  Example:
   //     varexp = x     simplest case: draw a 1-Dim distribution of column named x
   //            = sqrt(x)            : draw distribution of sqrt(x)
   //            = x*y/z
   //            = y:sqrt(x) 2-Dim distribution of y versus sqrt(x)
   //            = px:py:pz:2.5*E  produces a 3-d scatter-plot of px vs py ps pz
   //              and the color number of each marker will be 2.5*E.
   //              If the color number is negative it is set to 0.
   //              If the color number is greater than the current number of colors
   //                 it is set to the highest color number.
   //              The default number of colors is 50.
   //              see TStyle::SetPalette for setting a new color palette.
   //
   //  Note that the variables e1, e2 or e3 may contain a selection.
   //  example, if e1= x*(y<0), the value histogrammed will be x if y<0
   //  and will be 0 otherwise.
   //
   //  selection is an expression with a combination of the columns.
   //  In a selection all the C++ operators are authorized.
   //  The value corresponding to the selection expression is used as a weight
   //  to fill the histogram.
   //  If the expression includes only boolean operations, the result
   //  is 0 or 1. If the result is 0, the histogram is not filled.
   //  In general, the expression may be of the form:
   //      value*(boolean expression)
   //  if boolean expression is true, the histogram is filled with
   //  a weight = value.
   //  Examples:
   //      selection1 = "x<y && sqrt(z)>3.2"
   //      selection2 = "(x+y)*(sqrt(z)>3.2)"
   //  selection1 returns a weigth = 0 or 1
   //  selection2 returns a weight = x+y if sqrt(z)>3.2
   //             returns a weight = 0 otherwise.
   //
   //  option is the drawing option.
   //    - See TH1::Draw for the list of all drawing options.
   //    - If option COL is specified when varexp has three fields:
   //            tree.Draw("e1:e2:e3","","col");
   //      a 2D scatter is produced with e1 vs e2, and e3 is mapped on the color
   //      table.
   //    - If option contains the string "goff", no graphics is generated.
   //
   //  nentries is the number of entries to process (default is all)
   //  first is the first entry to process (default is 0)
   //
   //  This function returns the number of selected entries. It returns -1
   //  if an error occurs.
   //
   //     Drawing expressions using arrays and array elements
   //     ===================================================
   // Let assumes, a leaf fMatrix, on the branch fEvent, which is a 3 by 3 array,
   // or a TClonesArray.
   // In a TTree::Draw expression you can now access fMatrix using the following
   // syntaxes:
   //
   //   String passed    What is used for each entry of the tree
   //
   //   "fMatrix"       the 9 elements of fMatrix
   //   "fMatrix[][]"   the 9 elements of fMatrix
   //   "fMatrix[2][2]" only the elements fMatrix[2][2]
   //   "fMatrix[1]"    the 3 elements fMatrix[1][0], fMatrix[1][1] and fMatrix[1][2]
   //   "fMatrix[1][]"  the 3 elements fMatrix[1][0], fMatrix[1][1] and fMatrix[1][2]
   //   "fMatrix[][0]"  the 3 elements fMatrix[0][0], fMatrix[1][0] and fMatrix[2][0]
   //
   //   "fEvent.fMatrix...." same as "fMatrix..." (unless there is more than one leaf named fMatrix!).
   //
   // In summary, if a specific index is not specified for a dimension, TTree::Draw
   // will loop through all the indices along this dimension.  Leaving off the
   // last (right most) dimension of specifying then with the two characters '[]'
   // is equivalent.  For variable size arrays (and TClonesArray) the range
   // of the first dimension is recalculated for each entry of the tree.
   //
   // TTree::Draw also now properly handling operations involving 2 or more arrays.
   //
   // Let assume a second matrix fResults[5][2], here are a sample of some
   // of the possible combinations, the number of elements they produce and
   // the loop used:
   //
   //  expression                       element(s)  Loop
   //
   //  "fMatrix[2][1] - fResults[5][2]"   one     no loop
   //  "fMatrix[2][]  - fResults[5][2]"   three   on 2nd dim fMatrix
   //  "fMatrix[2][]  - fResults[5][]"    two     on both 2nd dimensions
   //  "fMatrix[][2]  - fResults[][1]"    three   on both 1st dimensions
   //  "fMatrix[][2]  - fResults[][]"     six     on both 1st and 2nd dimensions of
   //                                             fResults
   //  "fMatrix[][2]  - fResults[3][]"    two     on 1st dim of fMatrix and 2nd of
   //                                             fResults (at the same time)
   //  "fMatrix[][]   - fResults[][]"     six     on 1st dim then on  2nd dim
   //
   //
   // In summary, TTree::Draw loops through all un-specified dimensions.  To
   // figure out the range of each loop, we match each unspecified dimension
   // from left to right (ignoring ALL dimensions for which an index has been
   // specified), in the equivalent loop matched dimensions use the same index
   // and are restricted to the smallest range (of only the matched dimensions).
   // When involving variable arrays, the range can of course be different
   // for each entry of the tree.
   //
   // So the loop equivalent to "fMatrix[][2] - fResults[3][]" is:
   //
   //    for (Int_t i0; i < min(3,2); i++) {
   //       use the value of (fMatrix[i0][2] - fMatrix[3][i0])
   //    }
   //
   // So the loop equivalent to "fMatrix[][2] - fResults[][]" is:
   //
   //    for (Int_t i0; i < min(3,5); i++) {
   //       for (Int_t i1; i1 < 2; i1++) {
   //          use the value of (fMatrix[i0][2] - fMatrix[i0][i1])
   //       }
   //    }
   //
   // So the loop equivalent to "fMatrix[][] - fResults[][]" is:
   //
   //    for (Int_t i0; i < min(3,5); i++) {
   //       for (Int_t i1; i1 < min(3,2); i1++) {
   //          use the value of (fMatrix[i0][i1] - fMatrix[i0][i1])
   //       }
   //    }
   //
   //     Retrieving the result of Draw
   //     =============================
   //
   //  By default the temporary histogram created is called "htemp", but only in
   //  the one dimensional Draw("e1") it contains the TTree's data points. For
   //  a two dimensional Draw, the data is filled into a TGraph which is named
   //  "Graph". They can be retrieved by calling
   //    TH1F *htemp = (TH1F*)gPad->GetPrimitive("htemp"); // 1D
   //    TGraph *graph = (TGraph*)gPad->GetPrimitive("Graph"); // 2D
   //
   //  For a three and four dimensional Draw the TPloyMarker3D is unnamed, and
   //  cannot be retrieved.
   //
   //  gPad always contains a TH1 derived object called "htemp" which allows to
   //  access the axes:
   //    TGraph *graph = (TGraph*)gPad->GetPrimitive("Graph"); // 2D
   //    TH2F   *htemp = (TH2F*)gPad->GetPrimitive("htemp"); // empty, but has axes
   //    TAxis  *xaxis = htemp->GetXaxis();
   //
   //     Saving the result of Draw to an histogram
   //     =========================================
   //
   //  If varexp0 contains >>hnew (following the variable(s) name(s),
   //  the new histogram created is called hnew and it is kept in the current
   //  directory (and also the current pad). This works for all dimensions.
   //  Example:
   //    tree.Draw("sqrt(x)>>hsqrt","y>0")
   //    will draw sqrt(x) and save the histogram as "hsqrt" in the current
   //    directory.  To retrieve it do:
   //    TH1F *hsqrt = (TH1F*)gDirectory->Get("hsqrt");
   //
   //  The binning information is taken from the environment variables
   //
   //     Hist.Binning.?D.?
   //
   //  In addition, the name of the histogram can be followed by up to 9
   //  numbers between '(' and ')', where the numbers describe the
   //  following:
   //
   //   1 - bins in x-direction
   //   2 - lower limit in x-direction
   //   3 - upper limit in x-direction
   //   4-6 same for y-direction
   //   7-9 same for z-direction
   //
   //   When a new binning is used the new value will become the default.
   //   Values can be skipped.
   //  Example:
   //    tree.Draw("sqrt(x)>>hsqrt(500,10,20)")
   //          // plot sqrt(x) between 10 and 20 using 500 bins
   //    tree.Draw("sqrt(x):sin(y)>>hsqrt(100,10,60,50,.1,.5)")
   //          // plot sqrt(x) against sin(y)
   //          // 100 bins in x-direction; lower limit on x-axis is 10; upper limit is 60
   //          //  50 bins in y-direction; lower limit on y-axis is .1; upper limit is .5
   //
   //  By default, the specified histogram is reset.
   //  To continue to append data to an existing histogram, use "+" in front
   //  of the histogram name.
   //  A '+' in front of the histogram name is ignored, when the name is followed by
   //  binning information as described in the previous paragraph.
   //    tree.Draw("sqrt(x)>>+hsqrt","y>0")
   //      will not reset hsqrt, but will continue filling.
   //  This works for 1-D, 2-D and 3-D histograms.
   //
   //     Accessing collection objects
   //     ============================
   //
   //  TTree::Draw default's handling of collections is to assume that any
   //  request on a collection pertain to it content.  For example, if fTracks
   //  is a collection of Track objects, the following:
   //      tree->Draw("event.fTracks.fPx");
   //  will plot the value of fPx for each Track objects inside the collection.
   //  Also
   //     tree->Draw("event.fTracks.size()");
   //  would plot the result of the member function Track::size() for each
   //  Track object inside the collection.
   //  To access information about the collection itself, TTree::Draw support
   //  the '@' notation.  If a variable which points to a collection is prefixed
   //  or postfixed with '@', the next part of the expression will pertain to
   //  the collection object.  For example:
   //     tree->Draw("event.@fTracks.size()");
   //  will plot the size of the collection refered to by fTracks (i.e the number
   //  of Track objects).
   //
   //     Drawing 'objects'
   //     =================
   //
   //  When a class has a member function named AsDouble or AsString, requesting
   //  to directly draw the object will imply a call to one of the 2 functions.
   //  If both AsDouble and AsString are present, AsDouble will be used.
   //  AsString can return either a char*, a std::string or a TString.s
   //  For example, the following
   //     tree->Draw("event.myTTimeStamp");
   //  will draw the same histogram as 
   //     tree->Draw("event.myTTimeStamp.AsDouble()");
   //  In addition, when the object is a type TString or std::string, TTree::Draw
   //  will call respectively TString::Data and std::string::c_str()
   // 
   //  If the object is a TBits, the histogram will contain the index of the bit
   //  that are turned on.
   //
   //     Special functions and variables
   //     ===============================
   //
   //  Entry$:  A TTree::Draw formula can use the special variable Entry$
   //  to access the entry number being read.  For example to draw every
   //  other entry use:
   //    tree.Draw("myvar","Entry$%2==0");
   //
   //  Entry$    : return the current entry number (== TTree::GetReadEntry())
   //  Entries$  : return the total number of entries (== TTree::GetEntries())
   //  Length$   : return the total number of element of this formula for this
   //                 entry (==TTreeFormula::GetNdata())
   //  Iteration$: return the current iteration over this formula for this
   //                 entry (i.e. varies from 0 to Length$).
   //
   //  Length$(formula): return the total number of element of the formula given as a
   //                    parameter.
   //  Sum$(formula): return the sum of the value of the elements of the formula given
   //                    as a parameter.  For eaxmple the mean for all the elements in
   //                    one entry can be calculated with:
   //                Sum$(formula)/Length$(formula)
   //
   //  Alt$(primary,alternate) : return the value of "primary" if it is available
   //                 for the current iteration otherwise return the value of "alternate".
   //                 For example, with arr1[3] and arr2[2]
   //    tree->Draw("arr1+Alt$(arr2,0)");
   //                 will draw arr1[0]+arr2[0] ; arr1[1]+arr2[1] and arr1[2]+0
   //                 Or with a variable size array arr3
   //    tree->Draw("Alt$(arr3[0],0)+Alt$(arr3[1],0)+Alt$(arr3[2],0)");
   //                 will draw the sum arr3 for the index 0 to min(2,actual_size_of_arr3-1)
   //                 As a comparison
   //    tree->Draw("arr3[0]+arr3[1]+arr3[2]");
   //                 will draw the sum arr3 for the index 0 to 2 only if the
   //                 actual_size_of_arr3 is greater or equal to 3.
   //                 Note that the array in 'primary' is flatened/linearilized thus using
   //                 Alt$ with multi-dimensional arrays of different dimensions in unlikely
   //                 to yield the expected results.  To visualize a bit more what elements
   //                 would be matched by TTree::Draw, TTree::Scan can be used:
   //    tree->Scan("arr1:Alt$(arr2,0)");
   //                 will print on one line the value of arr1 and (arr2,0) that will be
   //                 matched by
   //    tree->Draw("arr1-Alt$(arr2,0)");
   //
   //     Drawing a user function accessing the TTree data directly
   //     =========================================================
   //
   //  If the formula contains  a file name, TTree::MakeProxy will be used
   //  to load and execute this file.   In particular it will draw the
   //  result of a function with the same name as the file.  The function
   //  will be executed in a context where the name of the branches can
   //  be used as a C++ variable.
   //
   //  For example draw px using the file hsimple.root (generated by the
   //  hsimple.C tutorial), we need a file named hsimple.cxx:
   //
   //     double hsimple() {
   //        return px;
   //     }
   //
   //  MakeProxy can then be used indirectly via the TTree::Draw interface
   //  as follow:
   //     new TFile("hsimple.root")
   //     ntuple->Draw("hsimple.cxx");
   //
   //  A more complete example is available in the tutorials directory:
   //    h1analysisProxy.cxx , h1analysProxy.h and h1analysisProxyCut.C
   //  which reimplement the selector found in h1analysis.C
   //
   //  The main features of this facility are:
   //
   //    * on-demand loading of branches
   //    * ability to use the 'branchname' as if it was a data member
   //    * protection against array out-of-bound
   //    * ability to use the branch data as object (when the user code is available)
   //
   //  See TTree::MakeProxy for more details.
   //
   //     Making a Profile histogram
   //     ==========================
   //  In case of a 2-Dim expression, one can generate a TProfile histogram
   //  instead of a TH2F histogram by specyfying option=prof or option=profs.
   //  The option=prof is automatically selected in case of y:x>>pf
   //  where pf is an existing TProfile histogram.
   //
   //     Saving the result of Draw to a TEventList or a TEntryList
   //     =========================================================
   //  TTree::Draw can be used to fill a TEventList object (list of entry numbers)
   //  instead of histogramming one variable.
   //  If varexp0 has the form >>elist , a TEventList object named "elist"
   //  is created in the current directory. elist will contain the list
   //  of entry numbers satisfying the current selection.
   //  If option "entrylist" is used, a TEntryList object is created
   //  Example:
   //    tree.Draw(">>yplus","y>0")
   //    will create a TEventList object named "yplus" in the current directory.
   //    In an interactive session, one can type (after TTree::Draw)
   //       yplus.Print("all")
   //    to print the list of entry numbers in the list.
   //    tree.Draw(">>yplus", "y>0", "entrylist")
   //    will create a TEntryList object names "yplus" in the current directory
   //
   //  By default, the specified entry list is reset.
   //  To continue to append data to an existing list, use "+" in front
   //  of the list name;
   //    tree.Draw(">>+yplus","y>0")
   //      will not reset yplus, but will enter the selected entries at the end
   //      of the existing list.
   //
   //      Using a TEventList or a TEntryList as Input
   //      ===========================
   //  Once a TEventList or a TEntryList object has been generated, it can be used as input
   //  for TTree::Draw. Use TTree::SetEventList or TTree::SetEntryList to set the 
   //  current event list
   //  Example1:
   //     TEventList *elist = (TEventList*)gDirectory->Get("yplus");
   //     tree->SetEventList(elist);
   //     tree->Draw("py");
   //  Example2:
   //     TEntryList *elist = (TEntryList*)gDirectory->Get("yplus");
   //     tree->SetEntryList(elist);
   //     tree->Draw("py");
   //  If a TEventList object is used as input, a new TEntryList object is created
   //  inside the SetEventList function. In case of a TChain, all tree headers are loaded
   //  for this transformation. This new object is owned by the chain and is deleted
   //  with it, unless the user extracts it by calling GetEntryList() function.
   //  See also comments to SetEventList() function of TTree and TChain.
   //
   //  If arrays are used in the selection critera, the entry entered in the
   //  list are all the entries that have at least one element of the array that
   //  satisfy the selection.
   //  Example:
   //      tree.Draw(">>pyplus","fTracks.fPy>0");
   //      tree->SetEventList(pyplus);
   //      tree->Draw("fTracks.fPy");
   //  will draw the fPy of ALL tracks in event with at least one track with
   //  a positive fPy.
   //
   //  To select only the elements that did match the original selection
   //  use TEventList::SetReapplyCut or TEntryList::SetReapplyCut.
   //  Example:
   //      tree.Draw(">>pyplus","fTracks.fPy>0");
   //      pyplus->SetReapplyCut(kTRUE);
   //      tree->SetEventList(pyplus);
   //      tree->Draw("fTracks.fPy");
   //  will draw the fPy of only the tracks that have a positive fPy.
   //
   //  Note: Use tree->SetEventList(0) if you do not want use the list as input.
   //
   //      How to obtain more info from TTree::Draw
   //      ========================================
   //
   //  Once TTree::Draw has been called, it is possible to access useful
   //  information still stored in the TTree object via the following functions:
   //    -GetSelectedRows()    // return the number of entries accepted by the
   //                          //selection expression. In case where no selection
   //                          //was specified, returns the number of entries processed.
   //    -GetV1()              //returns a pointer to the double array of V1
   //    -GetV2()              //returns a pointer to the double array of V2
   //    -GetV3()              //returns a pointer to the double array of V3
   //    -GetW()               //returns a pointer to the double array of Weights
   //                          //where weight equal the result of the selection expression.
   //   where V1,V2,V3 correspond to the expressions in
   //   TTree::Draw("V1:V2:V3",selection);
   //
   //   Example:
   //    Root > ntuple->Draw("py:px","pz>4");
   //    Root > TGraph *gr = new TGraph(ntuple->GetSelectedRows(),
   //                                   ntuple->GetV2(), ntuple->GetV1());
   //    Root > gr->Draw("ap"); //draw graph in current pad
   //    creates a TGraph object with a number of points corresponding to the
   //    number of entries selected by the expression "pz>4", the x points of the graph
   //    being the px values of the Tree and the y points the py values.
   //
   //   Important note: By default TTree::Draw creates the arrays obtained
   //    with GetV1, GetV2, GetV3, GetW with a length corresponding to the
   //    parameter fEstimate. By default fEstimate=10000 and can be modified
   //    via TTree::SetEstimate. A possible recipee is to do
   //       tree->SetEstimate(tree->GetEntries());
   //    You must call SetEstimate if the expected number of selected rows
   //    is greater than 10000.
   //
   //    You can use the option "goff" to turn off the graphics output
   //    of TTree::Draw in the above example.
   //
   //           Automatic interface to TTree::Draw via the TTreeViewer
   //           ======================================================
   //
   //    A complete graphical interface to this function is implemented
   //    in the class TTreeViewer.
   //    To start the TTreeViewer, three possibilities:
   //       - select TTree context menu item "StartViewer"
   //       - type the command  "TTreeViewer TV(treeName)"
   //       - execute statement "tree->StartViewer();"
   //

   GetPlayer();
   if (fPlayer)
      return fPlayer->DrawSelect(varexp,selection,option,nentries,firstentry);
   return -1;
}

//______________________________________________________________________________
void TTree::DropBaskets()
{
   // -- Remove some baskets from memory.

   TBranch* branch = 0;
   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i) {
      branch = (TBranch*) fBranches.UncheckedAt(i);
      branch->DropBaskets("all");
   }
}

//______________________________________________________________________________
void TTree::DropBuffers(Int_t)
{
   // -- Drop branch buffers to accomodate nbytes below MaxVirtualsize.

   // Be careful not to remove current read/write buffers.
   Int_t ndrop = 0;
   Int_t nleaves = fLeaves.GetEntriesFast();
   for (Int_t i = 0; i < nleaves; ++i)  {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      TBranch* branch = (TBranch*) leaf->GetBranch();
      Int_t nbaskets = branch->GetListOfBaskets()->GetEntriesFast();
      for (Int_t j = 0; j < nbaskets - 1; ++j) {
         if ((j == branch->GetReadBasket()) || (j == branch->GetWriteBasket())) {
            continue;
         }
         TBasket* basket = branch->GetBasket(j);
         ndrop += basket->DropBuffers();
         if (fTotalBuffers < fMaxVirtualSize) {
            return;
         }
      }
   }
}

//______________________________________________________________________________
Int_t TTree::Fill()
{
   // -- Fill all branches.
   //
   //   This function loops on all the branches of this tree.  For
   //   each branch, it copies to the branch buffer (basket) the current
   //   values of the leaves data types. If a leaf is a simple data type,
   //   a simple conversion to a machine independent format has to be done.
   //
   //   The function returns the number of bytes committed to the
   //   individual branches.
   //
   //   If a write error occurs, the number of bytes returned is -1.
   //
   //   If no data are written, because, e.g., the branch is disabled,
   //   the number of bytes returned is 0.
   //

   Int_t nbytes = 0;
   Int_t nerror = 0;
   Int_t nb = fBranches.GetEntriesFast();
   if (nb == 1) {
      // -- Case of one single super branch. Automatically update
      // all the branch addresses if a new object was created.
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(0);
      branch->UpdateAddress();
   }
   if (fBranchRef) {
      fBranchRef->Clear();
   }
   for (Int_t i = 0; i < nb; ++i) {
      // -- Loop over all branches, filling and accumulating bytes written and error counts.
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (branch->TestBit(kDoNotProcess)) {
         continue;
      }
      Int_t nwrite = branch->Fill();
      if (nwrite < 0)  {
         Error("Fill", "Failed filling branch:%s.%s, nbytes=%d", GetName(), branch->GetName(), nwrite);
         if (nerror < 2) {
            printf(" This error is symptomatic of a Tree created as a memory-resident Tree\n");
            printf(" Instead of doing:\n");
            printf("    TTree *T = new TTree(...)\n");
            printf("    TFile *f = new TFile(...)\n");
            printf(" you should do:\n");
            printf("    TFile *f = new TFile(...)\n");
            printf("    TTree *T = new TTree(...)\n");
         }
         ++nerror;
      } else {
         nbytes += nwrite;
      }
   }
   if (fBranchRef) {
      fBranchRef->Fill();
   }
   ++fEntries;
   if (fEntries > fMaxEntries) {
      KeepCircular();
   }
   if ((fTotBytes - fSavedBytes) > fAutoSave) {
      AutoSave();
   }
   // Check that output file is still below the maximum size.
   // If above, close the current file and continue on a new file.
   // Currently, the automatic change of file is restricted
   // to the case where the tree is in the top level directory.
   if (!fDirectory) {
      return nbytes;
   }
   TFile* file = fDirectory->GetFile();
   if (file && (file->GetEND() > fgMaxTreeSize)) {
      if (fDirectory == (TDirectory*) file) {
         ChangeFile(file);
      }
   }
   if (nerror) {
      return -1;
   }
   return nbytes;
}

//______________________________________________________________________________
TBranch* TTree::FindBranch(const char* branchname)
{
   // -- FIXME: Describe this function.

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kFindBranch & fFriendLockStatus) {
      return 0;
   }
   TIter next(GetListOfBranches());
   // This will allow the branchname to be preceded by
   // the name of this tree.
   char* subbranch = (char*) strstr(branchname, GetName());
   if (subbranch != branchname) {
      subbranch = 0;
   }
   if (subbranch) {
      subbranch += strlen(GetName());
      if (*subbranch != '.') {
         subbranch = 0;
      } else {
         ++subbranch;
      }
   }
   TBranch* branch = 0;
   while ((branch = (TBranch*) next())) {
      std::string name(branch->GetName());
      std::size_t dim = name.find_first_of("[");
      if (dim != std::string::npos) {
         name.erase(dim);
      }
      if (branchname == name) {
         return branch;
      }
      if (subbranch && (subbranch == name)) {
         return branch;
      }
   }
   next.Reset();
   branch = 0;
   while ((branch = (TBranch*) next())) {
      TBranch* nestedbranch = branch->FindBranch(branchname);
      if (nestedbranch) {
         return nestedbranch;
      }
   }
   // Search in list of friends.
   if (!fFriends) {
      return 0;
   }
   TFriendLock lock(this, kFindBranch);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree* t = fe->GetTree();
      if (!t) {
         continue;
      }
      // If the alias is present replace it with the real name.
      char* subbranch = (char*) strstr(branchname, fe->GetName());
      if (subbranch != branchname) {
         subbranch = 0;
      }
      if (subbranch) {
         subbranch += strlen(fe->GetName());
         if (*subbranch != '.') {
            subbranch = 0;
         } else {
            ++subbranch;
         }
      }
      std::ostringstream name;
      if (subbranch) {
         name << t->GetName() << "." << subbranch;
      } else {
         name << branchname;
      }
      branch = t->FindBranch(name.str().c_str());
      if (branch) {
         return branch;
      }
   }
   return 0;
}

//______________________________________________________________________________
TLeaf* TTree::FindLeaf(const char* searchname)
{
   // -- FIXME: Describe this function.

   // We already have been visited while recursively looking
   // through the friends tree, let's return.
   if (kFindLeaf & fFriendLockStatus) {
      return 0;
   }

   // This will allow the branchname to be preceded by
   // the name of this tree.
   char* subsearchname = (char*) strstr(searchname, GetName());
   if (subsearchname != searchname) {
      subsearchname = 0;
   }
   if (subsearchname) {
      subsearchname += strlen(GetName());
      if (*subsearchname != '.') {
         subsearchname = 0;
      } else {
         ++subsearchname;
      }
   }

   TString leafname;
   TString leaftitle;
   TString longname;
   TString longtitle;

   // For leaves we allow for one level up to be prefixed to the name.
   TIter next(GetListOfLeaves());
   TLeaf* leaf = 0;
   while ((leaf = (TLeaf*) next())) {
      leafname = leaf->GetName();
      Ssiz_t dim = leafname.First('[');
      if (dim >= 0) leafname.Remove(dim);

      if (leafname == searchname) {
         return leaf;
      }
      if (subsearchname && leafname == subsearchname) {
         return leaf;
      }
      // The TLeafElement contains the branch name
      // in its name, let's use the title.
      leaftitle = leaf->GetTitle();
      dim = leaftitle.First('[');
      if (dim >= 0) leaftitle.Remove(dim);

      if (leaftitle == searchname) {
         return leaf;
      }
      if (subsearchname && leaftitle == subsearchname) {
         return leaf;
      }
      TBranch* branch = leaf->GetBranch();
      if (branch) {
         longname.Form("%s.%s",branch->GetName(),leafname.Data());
         Ssiz_t dim = longname.First('[');
         if (dim>=0) longname.Remove(dim);
         if (longname == searchname) {
            return leaf;
         }
         if (subsearchname && longname == subsearchname) {
            return leaf;
         }
         longtitle.Form("%s.%s",branch->GetName(),leaftitle.Data());
         dim = longtitle.First('[');
         if (dim>=0) longtitle.Remove(dim);
         if (longtitle == searchname) {
            return leaf;
         }
         if (subsearchname && longtitle == subsearchname) {
            return leaf;
         }
         // The following is for the case where the branch is only
         // a sub-branch.  Since we do not see it through
         // TTree::GetListOfBranches, we need to see it indirectly.
         // This is the less sturdy part of this search ... it may
         // need refining ...
         if (strstr(searchname, ".") && !strcmp(searchname, branch->GetName())) {
            return leaf;
         }
         if (subsearchname && strstr(subsearchname, ".") && !strcmp(subsearchname, branch->GetName())) {
            return leaf;
         }
      }
   }
   // Search in list of friends.
   if (!fFriends) {
      return 0;
   }
   TFriendLock lock(this, kFindLeaf);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree* t = fe->GetTree();
      if (!t) {
         continue;
      }
      // If the alias is present replace it with the real name.
      char* subsearchname = (char*) strstr(searchname, fe->GetName());
      if (subsearchname != searchname) {
         subsearchname = 0;
      }
      if (subsearchname) {
         subsearchname += strlen(fe->GetName());
         if (*subsearchname != '.') {
            subsearchname = 0;
         } else {
            ++subsearchname;
         }
      }
      if (subsearchname) {
         leafname.Form("%s.%s",t->GetName(),subsearchname);
      } else {
         leafname = searchname;
      }
      leaf = t->FindLeaf(leafname);
      if (leaf) {
         return leaf;
      }
   }
   return 0;
}

//______________________________________________________________________________
Long64_t TTree::Fit(const char* funcname, const char* varexp, const char* selection, Option_t* option, Option_t* goption, Long64_t nentries, Long64_t firstentry)
{
   // -- Fit  a projected item(s) from a tree.
   //
   //  funcname is a TF1 function.
   //
   //  See TTree::Draw() for explanations of the other parameters.
   //
   //  By default the temporary histogram created is called htemp.
   //  If varexp contains >>hnew , the new histogram created is called hnew
   //  and it is kept in the current directory.
   //
   //  The function returns the number of selected entries.
   //
   //  Example:
   //    tree.Fit(pol4,sqrt(x)>>hsqrt,y>0)
   //    will fit sqrt(x) and save the histogram as "hsqrt" in the current
   //    directory.
   //
   //   See also TTree::UnbinnedFit

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Fit(funcname, varexp, selection, option, goption, nentries, firstentry);
   }
   return -1;
}

//______________________________________________________________________________
const char* TTree::GetAlias(const char* aliasName) const
{
   // -- Returns the expanded value of the alias.  Search in the friends if any.

   // We already have been visited while recursively looking
   // through the friends tree, let's return.
   if (kGetAlias & fFriendLockStatus) {
      return 0;
   }
   if (fAliases) {
      TObject* alias = fAliases->FindObject(aliasName);
      if (alias) {
         return alias->GetTitle();
      }
   }
   if (!fFriends) {
      return 0;
   }
   TFriendLock lock(const_cast<TTree*>(this), kGetAlias);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree* t = fe->GetTree();
      if (t) {
         const char* alias = t->GetAlias(aliasName);
         if (alias) {
            return alias;
         }
         const char* subAliasName = strstr(aliasName, fe->GetName());
         if (subAliasName && (subAliasName[strlen(fe->GetName())] == '.')) {
            alias = t->GetAlias(aliasName + strlen(fe->GetName()) + 1);
            if (alias) {
               return alias;
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
TBranch* TTree::GetBranch(const char* name)
{
   // -- Return pointer to the branch with the given name in this tree or its friends.

   if (name == 0 || name[0] == '0') return 0;

   // We already have been visited while recursively
   // looking through the friends tree, let's return.
   if (kGetBranch & fFriendLockStatus) {
      return 0;
   }

   // Search using branches.
   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; i++) {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      if (!strcmp(branch->GetName(), name)) {
         return branch;
      }
      TObjArray* lb = branch->GetListOfBranches();
      Int_t nb1 = lb->GetEntriesFast();
      for (Int_t j = 0; j < nb1; j++) {
         TBranch* b1 = (TBranch*) lb->UncheckedAt(j);
         if (!strcmp(b1->GetName(), name)) {
            return b1;
         }
         TObjArray* lb1 = b1->GetListOfBranches();
         Int_t nb2 = lb1->GetEntriesFast();
         for (Int_t k = 0; k < nb2; k++) {
            TBranch* b2 = (TBranch*) lb1->UncheckedAt(k);
            if (!strcmp(b2->GetName(), name)) {
               return b2;
            }
         }
      }
   }

   // Search using leaves.
   TObjArray* leaves = GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (Int_t i = 0; i < nleaves; i++) {
      TLeaf* leaf = (TLeaf*) leaves->UncheckedAt(i);
      TBranch* branch = leaf->GetBranch();
      if (!strcmp(branch->GetName(), name)) {
         return branch;
      }
   }

   if (!fFriends) {
      return 0;
   }

   // Search in list of friends.
   TFriendLock lock(this, kGetBranch);
   TIter next(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) next())) {
      TTree* t = fe->GetTree();
      if (t) {
         TBranch* branch = t->GetBranch(name);
         if (branch) {
            return branch;
         }
      }
   }

   // Second pass in the list of friends when
   // the branch name is prefixed by the tree name.
   next.Reset();
   while ((fe = (TFriendElement*) next())) {
      TTree* t = fe->GetTree();
      if (!t) {
         continue;
      }
      char* subname = (char*) strstr(name, fe->GetName());
      if (subname != name) {
         continue;
      }
      Int_t l = strlen(fe->GetName());
      subname += l;
      if (*subname != '.') {
         continue;
      }
      subname++;
      TBranch* branch = t->GetBranch(subname);
      if (branch) {
         return branch;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TTree::GetBranchStatus(const char* branchname) const
{
   // -- Return status of branch with name branchname.
   // 0 if branch is not activated
   // 1 if branch is activated

   TBranch* br = const_cast<TTree*>(this)->GetBranch(branchname);
   if (br) {
      return br->TestBit(kDoNotProcess) == 0;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TTree::GetBranchStyle()
{
   // -- Static function returning the current branch style.
   // style = 0 old Branch
   // style = 1 new Bronch

   return fgBranchStyle;
}

//______________________________________________________________________________
TFile* TTree::GetCurrentFile() const
{
   // -- Return pointer to the current file.

   if (!fDirectory) {
      return 0;
   }
   return fDirectory->GetFile();
}

//______________________________________________________________________________
Long64_t TTree::GetEntries(const char *selection)
{
   // Return the number of entries matching the selection.
   // Return -1 in case of errors.
   //
   // If the selection uses any arrays or containers, we return the number
   // of entries where at least one element match the selection.
   // GetEntries is implemented using the selector class TSelectorEntries,
   // which can be used directly (see code in TTreePlayer::GetEntries) for
   // additional option.
   // If SetEventList was used on the TTree or TChain, only that subset
   // of entries will be considered.

   GetPlayer();
   if (fPlayer) {
      return fPlayer->GetEntries(selection);
   }
   return -1;
}

//______________________________________________________________________________
Long64_t TTree::GetEntriesFriend() const
{
   // -- Return number of entries of this tree if not zero, otherwise return the number of entries in the first friend tree.

   if (fEntries) return fEntries;
   if (!fFriends) return 0;
   TFriendElement *fr = (TFriendElement*)fFriends->At(0);
   if (!fr) return 0;
   TTree *t = fr->GetTree();
   if (t==0) return 0;
   return t->GetEntriesFriend();
}

//______________________________________________________________________________
Int_t TTree::GetEntry(Long64_t entry, Int_t getall)
{
   // -- Read all branches of entry and return total number of bytes read.
   //
   //     getall = 0 : get only active branches
   //     getall = 1 : get all branches
   //
   //  The function returns the number of bytes read from the input buffer.
   //  If entry does not exist  the function returns 0.
   //  If an I/O error occurs,  the function returns -1.
   //
   //  If the Tree has friends, also read the friends entry
   //
   //  To activate/deactivate one or more branches, use TBranch::SetBranchStatus
   //  For example, if you have a Tree with several hundred branches, and you
   //  are interested only by branches named "u" and "v", do
   //     mytree.SetBranchStatus("*",0); //disable all branches
   //     mytree.SetBranchStatus("a",1);
   //     mytree.SetBranchStatus("b",1);
   //  when calling mytree.GetEntry(i); only branches "a" and "b" will be read.
   //
   //  WARNING!!
   //  If your Tree has been created in split mode with a parent branch "parent",
   //     mytree.SetBranchStatus("parent",1);
   //  will not activate the sub-branches of "parent". You should do:
   //     mytree.SetBranchStatus("parent*",1);
   //
   //  An alternative is to call directly
   //     brancha.GetEntry(i)
   //     branchb.GetEntry(i);
   //
   //  IMPORTANT NOTE
   //  ==============
   // By default, GetEntry reuses the space allocated by the previous object
   // for each branch. You can force the previous object to be automatically
   // deleted if you call mybranch.SetAutoDelete(kTRUE) (default is kFALSE).
   // Example:
   // Consider the example in $ROOTSYS/test/Event.h
   // The top level branch in the tree T is declared with:
   //    Event *event = 0;  //event must be null or point to a valid object
   //                       //it must be initialized
   //    T.SetBranchAddress("event",&event);
   // When reading the Tree, one can choose one of these 3 options:
   //
   //   OPTION 1
   //   --------
   //
   //    for (Long64_t i=0;i<nentries;i++) {
   //       T.GetEntry(i);
   //       // the object event has been filled at this point
   //    }
   //   The default (recommended). At the first entry an object of the
   //   class Event will be created and pointed by event.
   //   At the following entries, event will be overwritten by the new data.
   //   All internal members that are TObject* are automatically deleted.
   //   It is important that these members be in a valid state when GetEntry
   //   is called. Pointers must be correctly initialized.
   //   However these internal members will not be deleted if the characters "->"
   //   are specified as the first characters in the comment field of the data
   //   member declaration.
   //   If "->" is specified, the pointer member is read via pointer->Streamer(buf).
   //   In this case, it is assumed that the pointer is never null (case
   //   of pointer TClonesArray *fTracks in the Event example).
   //   If "->" is not specified, the pointer member is read via buf >> pointer.
   //   In this case the pointer may be null. Note that the option with "->"
   //   is faster to read or write and it also consumes less space in the file.
   //
   //   OPTION 2
   //   --------
   //  The option AutoDelete is set
   //   TBranch *branch = T.GetBranch("event");
   //   branch->SetAddress(&event);
   //   branch->SetAutoDelete(kTRUE);
   //    for (Long64_t i=0;i<nentries;i++) {
   //       T.GetEntry(i);
   //       // the objrect event has been filled at this point
   //    }
   //   In this case, at each iteration, the object event is deleted by GetEntry
   //   and a new instance of Event is created and filled.
   //
   //   OPTION 3
   //   --------
   //   Same as option 1, but you delete yourself the event.
   //    for (Long64_t i=0;i<nentries;i++) {
   //       delete event;
   //       event = 0;  // EXTREMELY IMPORTANT
   //       T.GetEntry(i);
   //       // the objrect event has been filled at this point
   //    }
   //
   //  It is strongly recommended to use the default option 1. It has the
   //  additional advantage that functions like TTree::Draw (internally
   //  calling TTree::GetEntry) will be functional even when the classes in the
   //  file are not available.

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kGetEntry & fFriendLockStatus) return 0;

   if (entry < 0 || entry >= fEntries) return 0;
   Int_t i;
   Int_t nbytes = 0;
   fReadEntry = entry;
   TBranch *branch;

   Int_t nbranches = fBranches.GetEntriesFast();
   Int_t nb=0;
   for (i=0;i<nbranches;i++)  {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      nb = branch->GetEntry(entry, getall);
      if (nb < 0) return nb;
      nbytes += nb;
   }

   // GetEntry in list of friends
   if (!fFriends) return nbytes;
   TFriendLock lock(this,kGetEntry);
   TIter nextf(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)nextf())) {
      TTree *t = fe->GetTree();
      if (t) {
         if ( t->LoadTreeFriend(entry,this) >= 0 ) {
            nb = t->GetEntry(t->GetReadEntry(),getall);
         } else nb = 0;
         if (nb < 0) return nb;
         nbytes += nb;
      }
   }
   return nbytes;
}

TEntryList* TTree::GetEntryList()
{
//Returns the entry list, set to this tree
//
//The returned object is not owned by the tree, even if the entry list was created
//by the SetEventList() function (see also comments of SetEventList())

   if (!fEntryList) return 0;

   //check, if the entry list is owned by the tree.
   //This lack of "constness" is caused by the SetEventList() function
   //creating an entry list.
   if (fEntryList->TestBit(kCanDelete) == kTRUE){
      fEntryList->SetBit(kCanDelete, kFALSE);
   }
   return fEntryList;
}

//______________________________________________________________________________
//______________________________________________________________________________
Long64_t TTree::GetEntryNumber(Long64_t entry) const
{
   // -- Return entry number corresponding to entry.
   //
   // if no TEntryList set returns entry
   // else returns the entry number corresponding to the list index=entry

   if (!fEntryList) {
      return entry;
   }

   return fEntryList->GetEntry(entry);
}

//______________________________________________________________________________
Long64_t TTree::GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const
{
   // -- Return entry number corresponding to major and minor number.
   // Note that this function returns only the entry number, not the data
   // To read the data corresponding to an entry number, use TTree::GetEntryWithIndex
   // the BuildIndex function has created a table of Long64_t* of sorted values
   // corresponding to val = major<<31 + minor;
   // The function performs binary search in this sorted table.
   // If it finds a pair that maches val, it returns directly the
   // index in the table.
   // If an entry corresponding to major and minor is not found, the function
   // returns the index of the major,minor pair immediatly lower than the
   // requested value, ie it will return -1 if the pair is lower than
   // the first entry in the index.
   //
   // See also GetEntryNumberWithIndex

   if (!fTreeIndex) {
      return -1;
   }
   return fTreeIndex->GetEntryNumberWithBestIndex(major, minor);
}

//______________________________________________________________________________
Long64_t TTree::GetEntryNumberWithIndex(Int_t major, Int_t minor) const
{
   // -- Return entry number corresponding to major and minor number.
   // Note that this function returns only the entry number, not the data
   // To read the data corresponding to an entry number, use TTree::GetEntryWithIndex
   // the BuildIndex function has created a table of Long64_t* of sorted values
   // corresponding to val = major<<31 + minor;
   // The function performs binary search in this sorted table.
   // If it finds a pair that maches val, it returns directly the
   // index in the table, otherwise it returns -1.
   //
   // See also GetEntryNumberWithBestIndex

   if (!fTreeIndex) {
      return -1;
   }
   return fTreeIndex->GetEntryNumberWithIndex(major, minor);
}

//______________________________________________________________________________
Int_t TTree::GetEntryWithIndex(Int_t major, Int_t minor)
{
   // -- Read entry corresponding to major and minor number.
   //
   //  The function returns the total number of bytes read.
   //  If the Tree has friend trees, the corresponding entry with
   //  the index values (major,minor) is read. Note that the master Tree
   //  and its friend may have different entry serial numbers corresponding
   //  to (major,minor).

   // We already have been visited while recursively looking
   // through the friends tree, let's return.
   if (kGetEntryWithIndex & fFriendLockStatus) {
      return 0;
   }
   Long64_t serial = GetEntryNumberWithIndex(major, minor);
   if (serial < 0) {
      return -1;
   }
   Int_t i;
   Int_t nbytes = 0;
   fReadEntry = serial;
   TBranch *branch;
   Int_t nbranches = fBranches.GetEntriesFast();
   Int_t nb;
   for (i = 0; i < nbranches; ++i) {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      nb = branch->GetEntry(serial);
      if (nb < 0) return nb;
      nbytes += nb;
   }
   // GetEntry in list of friends
   if (!fFriends) return nbytes;
   TFriendLock lock(this,kGetEntryWithIndex);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree *t = fe->GetTree();
      if (t) {
         serial = t->GetEntryNumberWithIndex(major,minor);
         if (serial <0) return -nbytes;
         nb = t->GetEntry(serial);
         if (nb < 0) return nb;
         nbytes += nb;
      }
   }
   return nbytes;
}

//______________________________________________________________________________
const char* TTree::GetFriendAlias(TTree* tree) const
{
   // -- If the the 'tree' is a friend, this method returns its alias name.
   //
   // This alias is an alternate name for the tree.
   //
   // It can be used in conjunction with a branch or leaf name in a TTreeFormula,
   // to specify in which particular tree the branch or leaf can be found if
   // the friend trees have branches or leaves with the same name as the master
   // tree.
   //
   // It can also be used in conjunction with an alias created using
   // TTree::SetAlias in a TTreeFormula, e.g.:
   //
   //      maintree->Draw("treealias.fPx - treealias.myAlias");
   //
   // where fPx is a branch of the friend tree aliased as 'treealias' and 'myAlias'
   // was created using TTree::SetAlias on the friend tree.
   //
   // However, note that 'treealias.myAlias' will be expanded literally,
   // without remembering that it comes from the aliased friend and thus
   // the branch name might not be disambiguated properly, which means
   // that you may not be able to take advantage of this feature.
   //

   if ((tree == this) || (tree == GetTree())) {
      return 0;
   }

   // We already have been visited while recursively
   // looking through the friends tree, let's return.
   if (kGetFriendAlias & fFriendLockStatus) {
      return 0;
   }
   if (!fFriends) {
      return 0;
   }
   TFriendLock lock(const_cast<TTree*>(this), kGetFriendAlias);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree* t = fe->GetTree();
      if (t == tree) {
         return fe->GetName();
      }
      if (t->IsA()->InheritsFrom("TChain")) {
         if (t->GetTree() == tree) {
            return fe->GetName();
         }
      }
   }
   // After looking at the first level,
   // let's see if it is a friend of friends.
   nextf.Reset();
   fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      const char* res = fe->GetTree()->GetFriendAlias(tree);
      if (res) {
         return res;
      }
   }
   return 0;
}

//______________________________________________________________________________
TIterator* TTree::GetIteratorOnAllLeaves(Bool_t dir)
{
   // -- Creates a new iterator that will go through all the leaves on the tree itself and its friend.

   return new TTreeFriendLeafIter(this, dir);
}

//______________________________________________________________________________
TLeaf* TTree::GetLeaf(const char* aname)
{
   // -- Return pointer to the 1st Leaf named name in any Branch of this Tree or any branch in the list of friend trees.
   //
   //  aname may be of the form branchname/leafname

   if (aname == 0 || aname[0] == '0') return 0;

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kGetLeaf & fFriendLockStatus) {
      return 0;
   }
   char* slash = (char*) strchr(aname, '/');
   char* name = 0;
   UInt_t nbch = 0;
   if (slash) {
      name = slash + 1;
      nbch = slash - aname;
   } else {
      name = (char*) aname;
   }
   TLeaf *leaf = 0;
   TIter nextl(GetListOfLeaves());
   while ((leaf = (TLeaf*)nextl())) {
      if (strcmp(leaf->GetName(),name)) continue;
      if (slash) {
         const char* brname = leaf->GetBranch()->GetName();
         if (strncmp(brname,aname,nbch)) continue;
         // The start of the branch name is indentical to the content
         // of 'aname' before the first '/'.
         // Let's make sure that it is not longer (we are trying
         // to avoid having jet2/value match the branch jet23
         if ((strlen(brname) > nbch) && (brname[nbch] != '.') && (brname[nbch] != '[')) {
            continue;
         }
      }
      return leaf;
   }
   if (!fFriends) return 0;
   TFriendLock lock(this,kGetLeaf);
   TIter next(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      if (t) {
         leaf = t->GetLeaf(aname);
         if (leaf) return leaf;
      }
   }

   //second pass in the list of friends when the leaf name
   //is prefixed by the tree name
   TString strippedArg;
   next.Reset();
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      if (t==0) continue;
      char *subname = (char*)strstr(name,fe->GetName());
      if (subname != name) continue;
      Int_t l = strlen(fe->GetName());
      subname += l;
      if (*subname != '.') continue;
      subname++;
      if (slash) {
         strippedArg = aname;
         strippedArg.Remove(nbch+1);
      } else {
         strippedArg = "";
      }
      strippedArg += subname;
      leaf = t->GetLeaf(strippedArg);
      if (leaf) return leaf;
   }
   return 0;
}

//______________________________________________________________________________
Double_t TTree::GetMaximum(const char* columname)
{
   // -- Return maximum of column with name columname.

   TLeaf* leaf = this->GetLeaf(columname);
   if (!leaf) {
      return 0;
   }
   TBranch* branch = leaf->GetBranch();
   Double_t cmax = -FLT_MAX;
   for (Long64_t i = 0; i < fEntries; ++i) {
      branch->GetEntry(i);
      for (Int_t j = 0; j < leaf->GetLen(); ++j) {
         Double_t val = leaf->GetValue(j);
         if (val > cmax) {
            cmax = val;
         }
      }
   }
   return cmax;
}

//______________________________________________________________________________
Long64_t TTree::GetMaxTreeSize()
{
   // -- Static function which returns the tree file size limit.

   return fgMaxTreeSize;
}

//______________________________________________________________________________
Double_t TTree::GetMinimum(const char* columname)
{
   // -- Return minimum of column with name columname.

   TLeaf* leaf = this->GetLeaf(columname);
   if (!leaf) {
      return 0;
   }
   TBranch* branch = leaf->GetBranch();
   Double_t cmin = FLT_MAX;
   for (Long64_t i = 0; i < fEntries; ++i) {
      branch->GetEntry(i);
      for (Int_t j = 0;j < leaf->GetLen(); ++j) {
         Double_t val = leaf->GetValue(j);
         if (val < cmin) {
            cmin = val;
         }
      }
   }
   return cmin;
}

//______________________________________________________________________________
const char* TTree::GetNameByIndex(TString& varexp, Int_t* index, Int_t colindex) const
{
   // -- Return name corresponding to colindex in varexp.
   //
   //   varexp is a string of names separated by :
   //   index is an array with pointers to the start of name[i] in varexp
   //

   Int_t i1,n;
   static TString column;
   if (colindex<0 ) return "";
   i1 = index[colindex] + 1;
   n  = index[colindex+1] - i1;
   column = varexp(i1,n);
   //  return (const char*)Form((const char*)column);
   return column.Data();
}

//______________________________________________________________________________
TVirtualTreePlayer* TTree::GetPlayer()
{
   // -- Load the TTreePlayer (if not already done).

   if (fPlayer) {
      return fPlayer;
   }
   fPlayer = TVirtualTreePlayer::TreePlayer(this);
   return fPlayer;
}

//______________________________________________________________________________
TList* TTree::GetUserInfo()
{
   // -- Return a pointer to the list containing user objects associated to this tree.
   //
   // The list is automatically created if it does not exist.
   //
   // WARNING: By default the TTree destructor will delete all objects added
   //          to this list. If you do not want these objects to be deleted,
   //          call:
   //
   //               mytree->GetUserInfo()->Clear();
   //
   //          before deleting the tree.

   if (!fUserInfo) {
      fUserInfo = new TList();
   }
   return fUserInfo;
}

//______________________________________________________________________________
void TTree::KeepCircular()
{
   // -- Keep a maximum of fMaxEntries in memory.

   Int_t nb = fBranches.GetEntriesFast();
   Long64_t maxEntries = fMaxEntries - (fMaxEntries / 10);
   for (Int_t i = 0; i < nb; ++i)  {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      branch->KeepCircular(maxEntries);
   }
   fEntries = maxEntries;
   fReadEntry = -1;
}

//______________________________________________________________________________
Int_t TTree::LoadBaskets(Long64_t maxmemory)
{
   // -- Read in memory all baskets from all branches up to the limit of maxmemory bytes.
   //
   // If maxmemory is non null and positive SetMaxVirtualSize is called
   // with this value. Default for maxmemory is 2000000000 (2 Gigabytes).
   // The function returns the total number of baskets read into memory
   // if negative an error occured while loading the branches.
   // This method may be called to force branch baskets in memory
   // when random access to branch entries is required.
   // If random access to only a few branches is required, you should
   // call directly TBranch::LoadBaskets.

   if (maxmemory > 0) SetMaxVirtualSize(maxmemory);

   TIter next(GetListOfLeaves());
   TLeaf *leaf;
   Int_t nimported = 0;
   while ((leaf=(TLeaf*)next())) {
      nimported += leaf->GetBranch()->LoadBaskets();//break;
   }
   return nimported;
}

//______________________________________________________________________________
Long64_t TTree::LoadTree(Long64_t entry)
{
   // -- Set current entry.
   //
   // Returns -2 if entry does not exist (just as TChain::LoadTree()).
   //
   // Note: This function is overloaded in TChain.
   //

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kLoadTree & fFriendLockStatus) {
      return 0;
   }

   if (fNotify) {
      if (fReadEntry < 0) {
         fNotify->Notify();
      }
   }
   fReadEntry = entry;

   Bool_t friendHasEntry = kFALSE;
   if (fFriends) {
      // -- Set current entry in friends as well.
      //
      // An alternative would move this code to each of the
      // functions calling LoadTree (and to overload a few more).
      Bool_t needUpdate = kFALSE;
      {
         // -- This scope is need to insure the lock is released at the right time
         TIter nextf(fFriends);
         TFriendLock lock(this, kLoadTree);
         TFriendElement* fe = 0;
         while ((fe = (TFriendElement*) nextf())) {
            if (fe->TestBit(TFriendElement::kFromChain)) {
               // This friend element was added by the chain that owns this
               // tree, the chain will deal with loading the correct entry.
               continue;
            }
            TTree* friendTree = fe->GetTree();
            if (friendTree->IsA() == TTree::Class()) {
               // -- Friend is actually a tree.
               if (friendTree->LoadTreeFriend(entry, this) >= 0) {
                  friendHasEntry = kTRUE;
               }
            } else {
               // -- Friend is actually a chain.
               // FIXME: This logic should be in the TChain override.
               Int_t oldNumber = friendTree->GetTreeNumber();
               if (friendTree->LoadTreeFriend(entry, this) >= 0) {
                  friendHasEntry = kTRUE;
               }
               Int_t newNumber = friendTree->GetTreeNumber();
               if (oldNumber != newNumber) {
                  // We can not just compare the tree pointers because they could be reused.
                  // So we compare the tree number instead.
                  needUpdate = kTRUE;
               }
            }
         } // for each friend
      }
      if (needUpdate) {
         //update list of leaves in all TTreeFormula of the TTreePlayer (if any)
         if (fPlayer) {
            fPlayer->UpdateFormulaLeaves();
         }
         //Notify user if requested
         if (fNotify) {
            fNotify->Notify();
         }
      }
   }

   if ((fReadEntry >= fEntries) && !friendHasEntry) {
      return -2;
   }
   return fReadEntry;
}

//______________________________________________________________________________
Long64_t TTree::LoadTreeFriend(Long64_t entry, TTree* masterTree)
{
   // -- Load entry on behalf of our master tree, we may use an index.
   //
   // Called by LoadTree() when the masterTree looks for the entry
   // number in a friend tree (us) corresponding to the passed entry
   // number in the masterTree.
   //
   // If we have no index, our entry number and the masterTree entry
   // number are the same.
   //
   // If we *do* have an index, we must find the (major, minor) value pair
   // in masterTree to locate our corresponding entry.
   //

   if (!fTreeIndex) {
      return LoadTree(entry);
   }
   return LoadTree(fTreeIndex->GetEntryNumberFriend(masterTree));
}

//______________________________________________________________________________
Int_t TTree::MakeClass(const char* classname, Option_t* option)
{
   // -- Generate a skeleton analysis class for this tree.
   //
   // The following files are produced: classname.h and classname.C.
   // If classname is 0, classname will be called "nameoftree".
   //
   // The generated code in classname.h includes the following:
   //    - Identification of the original tree and the input file name.
   //    - Definition of an analysis class (data members and member functions).
   //    - The following member functions:
   //       - constructor (by default opening the tree file),
   //       - GetEntry(Long64_t entry),
   //       - Init(TTree* tree) to initialize a new TTree,
   //       - Show(Long64_t entry) to read and dump entry.
   //
   // The generated code in classname.C includes only the main
   // analysis function Loop.
   //
   // To use this function:
   //    - Open your tree file (eg: TFile f("myfile.root");)
   //    - T->MakeClass("MyClass");
   // where T is the name of the TTree in file myfile.root,
   // and MyClass.h, MyClass.C the name of the files created by this function.
   // In a ROOT session, you can do:
   //    root > .L MyClass.C
   //    root > MyClass* t = new MyClass;
   //    root > t->GetEntry(12); // Fill data members of t with entry number 12.
   //    root > t->Show();       // Show values of entry 12.
   //    root > t->Show(16);     // Read and show values of entry 16.
   //    root > t->Loop();       // Loop on all entries.
   //
   //  NOTE: Do not use the code generated for a single TTree which is part
   //        of a TChain to process that entire TChain.  The maximum dimensions
   //        calculated for arrays on the basis of a single TTree from the TChain
   //        might be (will be!) too small when processing all of the TTrees in
   //        the TChain.  You must use myChain.MakeClass() to generate the code,
   //        not myTree.MakeClass(...).
   //

   GetPlayer();
   if (!fPlayer) {
      return 0;
   }
   return fPlayer->MakeClass(classname, option);
}

//______________________________________________________________________________
Int_t TTree::MakeCode(const char* filename)
{
   // -- Generate a skeleton function for this tree.
   //
   // The function code is written on filename.
   // If filename is 0, filename will be called nameoftree.C
   //
   // The generated code includes the following:
   //    - Identification of the original Tree and Input file name,
   //    - Opening the Tree file,
   //    - Declaration of Tree variables,
   //    - Setting of branches addresses,
   //    - A skeleton for the entry loop.
   //
   // To use this function:
   //    - Open your Tree file (eg: TFile f("myfile.root");)
   //    - T->MakeCode("MyAnalysis.C");
   // where T is the name of the TTree in file myfile.root
   // and MyAnalysis.C the name of the file created by this function.
   //
   // NOTE: Since the implementation of this function, a new and better
   //       function TTree::MakeClass() has been developed.

   Warning("MakeCode", "MakeCode is obsolete. Use MakeClass or MakeSelector instead");

   GetPlayer();
   if (!fPlayer) return 0;
   return fPlayer->MakeCode(filename);
}

//______________________________________________________________________________
void TTree::MakeIndex(TString& varexp, Int_t* index)
{
   // -- Build index array for names in varexp.

   Int_t ivar = 1;
   index[0] = -1;
   for (Int_t i = 0; i < varexp.Length(); ++i) {
      if (varexp[i] == ':') {
         index[ivar] = i;
         ++ivar;
      }
   }
   index[ivar] = varexp.Length();
}

//______________________________________________________________________________
Int_t TTree::MakeProxy(const char* proxyClassname, const char* macrofilename, const char* cutfilename, const char* option, Int_t maxUnrolling)
{
   // -- Generate a skeleton analysis class for this Tree using TBranchProxy.
   //
   // TBranchProxy is the base of a class hierarchy implementing an
   // indirect access to the content of the branches of a TTree.
   //
   // "proxyClassname" is expected to be of the form:
   //    [path/]fileprefix
   // The skeleton will then be generated in the file:
   //    fileprefix.h
   // located in the current directory or in 'path/' if it is specified.
   // The class generated will be named 'fileprefix'
   //
   // "macrofilename" and optionally "cutfilename" are expected to point
   // to source files which will be included by the generated skeleton.
   // Method of the same name as the file(minus the extension and path)
   // will be called by the generated skeleton's Process method as follow:
   //    [if (cutfilename())] htemp->Fill(macrofilename());
   //
   // "option" can be used select some of the optional features during
   // the code generation.  The possible options are:
   //    nohist : indicates that the generated ProcessFill should not
   //             fill the histogram.
   //
   // 'maxUnrolling' controls how deep in the class hierachy does the
   // system 'unroll' classes that are not split.  Unrolling a class
   // allows direct access to its data members (this emulates the behavior
   // of TTreeFormula).
   //
   // The main features of this skeleton are:
   //
   //    * on-demand loading of branches
   //    * ability to use the 'branchname' as if it was a data member
   //    * protection against array out-of-bounds errors
   //    * ability to use the branch data as an object (when the user code is available)
   //
   // For example with Event.root, if
   //    Double_t somePx = fTracks.fPx[2];
   // is executed by one of the method of the skeleton,
   // somePx will updated with the current value of fPx of the 3rd track.
   //
   // Both macrofilename and the optional cutfilename are expected to be
   // the name of source files which contain at least a free standing
   // function with the signature:
   //     x_t macrofilename(); // i.e function with the same name as the file
   // and
   //     y_t cutfilename();   // i.e function with the same name as the file
   //
   // x_t and y_t needs to be types that can convert respectively to a double
   // and a bool (because the skeleton uses:
   //     if (cutfilename()) htemp->Fill(macrofilename());
   //
   // These two functions are run in a context such that the branch names are
   // available as local variables of the correct (read-only) type.
   //
   // Note that if you use the same 'variable' twice, it is more efficient
   // to 'cache' the value. For example
   //   Int_t n = fEventNumber; // Read fEventNumber
   //   if (n<10 || n>10) { ... }
   // is more efficient than
   //   if (fEventNumber<10 || fEventNumber>10)
   //
   // Also, optionally, the generated selector will also call methods named
   // macrofilename_methodname in each of 6 main selector methods if the method
   // macrofilename_methodname exist (Where macrofilename is stripped of its
   // extension).
   //
   // Concretely, with the script named h1analysisProxy.C,
   //
   // The method         calls the method (if it exist)
   // Begin           -> h1analysisProxy_Begin
   // SlaveBegin      -> h1analysisProxy_SlaveBegin
   // Notify          -> h1analysisProxy_Notify
   // Process         -> h1analysisProxy_Process
   // SlaveTerminate  -> h1analysisProxy_SlaveTerminate
   // Terminate       -> h1analysisProxy_Terminate
   //
   // If a file name macrofilename.h (or .hh, .hpp, .hxx, .hPP, .hXX) exist
   // it is included before the declaration of the proxy class.  This can
   // be used in particular to insure that the include files needed by
   // the macro file are properly loaded.
   //
   // The default histogram is accessible via the variable named 'htemp'.
   //
   // If the library of the classes describing the data in the branch is
   // loaded, the skeleton will add the needed #include statements and
   // give the ability to access the object stored in the branches.
   //
   // To draw px using the file hsimple.root (generated by the
   // hsimple.C tutorial), we need a file named hsimple.cxx:
   //
   //     double hsimple() {
   //        return px;
   //     }
   //
   // MakeProxy can then be used indirectly via the TTree::Draw interface
   // as follow:
   //     new TFile("hsimple.root")
   //     ntuple->Draw("hsimple.cxx");
   //
   // A more complete example is available in the tutorials directory:
   //   h1analysisProxy.cxx , h1analysProxy.h and h1analysisProxyCut.C
   // which reimplement the selector found in h1analysis.C

   GetPlayer();
   if (!fPlayer) return 0;
   return fPlayer->MakeProxy(proxyClassname,macrofilename,cutfilename,option,maxUnrolling);
}

//______________________________________________________________________________
Int_t TTree::MakeSelector(const char* selector)
{
   // -- Generate skeleton selector class for this tree.
   //
   // The following files are produced: selector.h and selector.C.
   // If selector is 0, the selector will be called "nameoftree".
   //
   // The generated code in selector.h includes the following:
   //    - Identification of the original Tree and Input file name
   //    - Definition of selector class (data and functions)
   //    - The following class functions:
   //       - constructor and destructor
   //       - void    Begin(TTree *tree)
   //       - void    Init(TTree *tree)
   //       - Bool_t  Notify()
   //       - Bool_t  Process(Long64_t entry)
   //       - void    Terminate
   //
   // The class selector derives from TSelector.
   // The generated code in selector.C includes empty functions defined above:
   //
   // To use this function:
   //    - connect your Tree file (eg: TFile f("myfile.root");)
   //    - T->MakeSelector("myselect");
   // where T is the name of the Tree in file myfile.root
   // and myselect.h, myselect.C the name of the files created by this function.
   // In a ROOT session, you can do:
   //    root > T->Process("select.C")

   return MakeClass(selector, "selector");
}

//______________________________________________________________________________
Bool_t TTree::MemoryFull(Int_t nbytes)
{
   // -- Check if adding nbytes to memory we are still below MaxVirtualsize.

   if ((fTotalBuffers + nbytes) < fMaxVirtualSize) {
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
TTree* TTree::MergeTrees(TList* li, Option_t* /* option */)
{
   // -- Static function merging the trees in the TList into a new tree.
   //
   // Trees in the list can be memory or disk-resident trees.
   // The new tree is created in the current directory (memory if gROOT).
   //

   if (!li) return 0;
   TIter next(li);
   TTree *newtree = 0;
   TObject *obj;

   while ((obj=next())) {
      if (!obj->InheritsFrom(TTree::Class())) continue;
      TTree *tree = (TTree*)obj;
      Long64_t nentries = tree->GetEntries();
      if (nentries == 0) continue;
      if (!newtree) {
         newtree = (TTree*)tree->CloneTree();

         // Once the cloning is done, separate the trees,
         // to avoid as many side-effects as possible
         tree->GetListOfClones()->Remove(newtree);
         tree->ResetBranchAddresses();
         newtree->ResetBranchAddresses();
         continue;
      }

      newtree->CopyAddresses(tree);
      for (Long64_t i=0;i<nentries;i++) {
         tree->GetEntry(i);
         newtree->Fill();
      }
      tree->ResetBranchAddresses(); // Disconnect from new tree.
   }
   return newtree;
}

//______________________________________________________________________________
Long64_t TTree::Merge(TCollection* li, Option_t* /* option */)
{
   // -- Merge the trees in the TList into this tree.
   //
   // Returns the total number of entries in the merged tree.
   //

   if (!li) return 0;
   TIter next(li);
   TTree *tree;
   while ((tree = (TTree*)next())) {
      if (tree==this) continue;
      if (!tree->InheritsFrom(TTree::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s", tree->ClassName(), ClassName());
         return -1;
      }

      Long64_t nentries = tree->GetEntries();
      if (nentries == 0) continue;

      CopyAddresses(tree);
      for (Long64_t i=0; i<nentries ; i++) {
         tree->GetEntry(i);
         Fill();
      }
      tree->ResetBranchAddresses();
   }

   return GetEntries();
}

//______________________________________________________________________________
Bool_t TTree::Notify()
{
   // -- Function called when loading a new class library.

   TIter next(GetListOfLeaves());
   TLeaf* leaf = 0;
   while ((leaf = (TLeaf*) next())) {
      leaf->Notify();
      leaf->GetBranch()->Notify();
   }
   return kTRUE;
}

//______________________________________________________________________________
TPrincipal* TTree::Principal(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Interface to the Principal Components Analysis class.
   //
   //   Create an instance of TPrincipal
   //   Fill it with the selected variables
   //   if option "n" is specified, the TPrincipal object is filled with
   //                 normalized variables.
   //   If option "p" is specified, compute the principal components
   //   If option "p" and "d" print results of analysis
   //   If option "p" and "h" generate standard histograms
   //   If option "p" and "c" generate code of conversion functions
   //   return a pointer to the TPrincipal object. It is the user responsability
   //   to delete this object.
   //   The option default value is "np"
   //
   //   see TTree::Draw for explanation of the other parameters.
   //
   //   The created object is  named "principal" and a reference to it
   //   is added to the list of specials Root objects.
   //   you can retrieve a pointer to the created object via:
   //      TPrincipal *principal =
   //        (TPrincipal*)gROOT->GetListOfSpecials()->FindObject("principal");
   //

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Principal(varexp, selection, option, nentries, firstentry);
   }
   return 0;
}

//______________________________________________________________________________
void TTree::Print(Option_t* option) const
{
   // -- Print a summary of the tree contents.
   //
   // If option contains "all" friend trees are also printed.
   // If option contains "toponly" only the top level branches are printed.
   //
   // Wildcarding can be used to print only a subset of the branches, e.g.,
   // T.Print("Elec*") will print all branches with name starting with "Elec".

   // We already have been visited while recursively looking
   // through the friends tree, let's return.
   if (kPrint & fFriendLockStatus) {
      return;
   }
   Int_t s = 0;
   Int_t skey = 0;
   if (fDirectory) {
      TKey* key = fDirectory->GetKey(GetName());
      if (key) {
         skey = key->GetKeylen();
         s = key->GetNbytes();
      }
   }
   Long64_t total = skey;
   if (fZipBytes > 0) {
      total += fTotBytes;
   }
   TBufferFile b(TBuffer::kWrite, 10000);
   TTree::Class()->WriteBuffer(b, (TTree*) this);
   total += b.Length();
   Long64_t file = fZipBytes + s;
   Float_t cx = 1;
   if (fZipBytes) {
      cx = (fTotBytes + 0.00001) / fZipBytes;
   }
   Printf("******************************************************************************");
   Printf("*Tree    :%-10s: %-54s *", GetName(), GetTitle());
   Printf("*Entries : %8lld : Total = %15lld bytes  File  Size = %10lld *", fEntries, total, file);
   Printf("*        :          : Tree compression factor = %6.2f                       *", cx);
   Printf("******************************************************************************");
   Int_t nl = const_cast<TTree*>(this)->GetListOfLeaves()->GetEntries();
   Int_t l;
   TBranch* br = 0;
   TLeaf* leaf = 0;
   if (strstr(option, "toponly")) {
      Long64_t *count = new Long64_t[nl];
      Int_t keep =0;
      for (l=0;l<nl;l++) {
         leaf = (TLeaf *)const_cast<TTree*>(this)->GetListOfLeaves()->At(l);
         br   = leaf->GetBranch();
         if (strchr(br->GetName(),'.')) {
            count[l] = -1;
            count[keep] += br->GetZipBytes();
         } else {
            keep = l;
            count[keep]  = br->GetZipBytes();
         }
      }
      for (l=0;l<nl;l++) {
         if (count[l] < 0) continue;
         leaf = (TLeaf *)const_cast<TTree*>(this)->GetListOfLeaves()->At(l);
         br   = leaf->GetBranch();
         printf("branch: %-20s %9lld\n",br->GetName(),count[l]);
      }
      delete [] count;
   } else {
      TString reg = "*";
      if (strlen(option) && strchr(option,'*')) reg = option;
      TRegexp re(reg,kTRUE);
      TIter next(const_cast<TTree*>(this)->GetListOfBranches());
      TBranch::ResetCount();
      while ((br= (TBranch*)next())) {
         TString s = br->GetName();
         s.ReplaceAll("/","_");
         if (s.Index(re) == kNPOS) continue;
         br->Print(option);
      }
   }

   //print TRefTable (if one)
   if (fBranchRef) fBranchRef->Print(option);

   //print friends if option "all"
   if (!fFriends || !strstr(option,"all")) return;
   TIter nextf(fFriends);
   TFriendLock lock(const_cast<TTree*>(this),kPrint);
   TFriendElement *fr;
   while ((fr = (TFriendElement*)nextf())) {
      TTree * t = fr->GetTree();
      if (t) t->Print(option);
   }
}

//______________________________________________________________________________
Long64_t TTree::Process(const char* filename, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Process this tree executing the code in filename.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   //
   // The code in filename is loaded (interpreted or compiled , see below)
   // filename must contain a valid class implementation derived from TSelector.
   // where TSelector has the following member functions:
   //
   //     void TSelector::Begin(). This function is called before looping on the
   //          events in the Tree. The user can create his histograms in this function.
   //
   //     Bool_t TSelector::ProcessCut(Long64_t entry). This function is called
   //          before processing entry. It is the user's responsability to read
   //          the corresponding entry in memory (may be just a partial read).
   //          The function returns kTRUE if the entry must be processed,
   //          kFALSE otherwise.
   //     void TSelector::ProcessFill(Long64_t entry). This function is called for
   //          all selected events. User fills histograms in this function.
   //     void TSelector::Terminate(). This function is called at the end of
   //          the loop on all events.
   //     void TTreeProcess::Begin(). This function is called before looping on the
   //          events in the Tree. The user can create his histograms in this function.
   //
   //   if filename is of the form file.C, the file will be interpreted.
   //   if filename is of the form file.C++, the file file.C will be compiled
   //      and dynamically loaded.
   //   if filename is of the form file.C+, the file file.C will be compiled
   //      and dynamically loaded. At next call, if file.C is older than file.o
   //      and file.so, the file.C is not compiled, only file.so is loaded.
   //
   //   The function returns the number of processed entries. It returns -1
   //   in case of an error.
   //
   //  NOTE1
   //  It may be more interesting to invoke directly the other Process function
   //  accepting a TSelector* as argument.eg
   //     MySelector *selector = (MySelector*)TSelector::GetSelector(filename);
   //     selector->CallSomeFunction(..);
   //     mytree.Process(selector,..);
   //
   //  NOTE2
   //  One should not call this function twice with the same selector file
   //  in the same script. If this is required, proceed as indicated in NOTE1,
   //  by getting a pointer to the corresponding TSelector,eg
   //    workaround 1
   //    ------------
   //void stubs1() {
   //   TSelector *selector = TSelector::GetSelector("h1test.C");
   //   TFile *f1 = new TFile("stubs_nood_le1.root");
   //   TTree *h1 = (TTree*)f1->Get("h1");
   //   h1->Process(selector);
   //   TFile *f2 = new TFile("stubs_nood_le1_coarse.root");
   //   TTree *h2 = (TTree*)f2->Get("h1");
   //   h2->Process(selector);
   //}
   //  or use ACLIC to compile the selector
   //   workaround 2
   //   ------------
   //void stubs2() {
   //   TFile *f1 = new TFile("stubs_nood_le1.root");
   //   TTree *h1 = (TTree*)f1->Get("h1");
   //   h1->Process("h1test.C+");
   //   TFile *f2 = new TFile("stubs_nood_le1_coarse.root");
   //   TTree *h2 = (TTree*)f2->Get("h1");
   //   h2->Process("h1test.C+");
   //}

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Process(filename, option, nentries, firstentry);
   }
   return -1;
}

//______________________________________________________________________________
Long64_t TTree::Process(TSelector* selector, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // Process this tree executing the code in selector.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   //
   // The TSelector class has the following member functions:
   //
   //   void TSelector::Begin(). This function is called before looping on the
   //        events in the Tree. The user can create his histograms in this function.
   //
   //   Bool_t TSelector::ProcessCut(Long64_t entry). This function is called
   //        before processing entry. It is the user's responsability to read
   //        the corresponding entry in memory (may be just a partial read).
   //        The function returns kTRUE if the entry must be processed,
   //        kFALSE otherwise.
   //   void TSelector::ProcessFill(Long64_t entry). This function is called for
   //        all selected events. User fills histograms in this function.
   //   void TSelector::Terminate(). This function is called at the end of
   //        the loop on all events.
   //   void TTreeProcess::Begin(). This function is called before looping on the
   //        events in the Tree. The user can create his histograms in this function.

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Process(selector, option, nentries, firstentry);
   }
   return -1;
}

//______________________________________________________________________________
Long64_t TTree::Project(const char* hname, const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Make a projection of a tree using selections.
   //
   // Depending on the value of varexp (described in Draw) a 1-D, 2-D, etc.,
   // projection of the tree will be filled in histogram hname.
   // Note that the dimension of hname must match with the dimension of varexp.
   //

   Int_t nch = strlen(hname) + strlen(varexp);
   char* var = new char[nch+5];
   sprintf(var, "%s>>%s", varexp, hname);
   nch = strlen(option) + 10;
   char* opt = new char[nch];
   if (option) {
      sprintf(opt, "%sgoff", option);
   } else {
      strcpy(opt, "goff");
   }
   Long64_t nsel = Draw(var, selection, opt, nentries, firstentry);
   delete[] var;
   var = 0;
   delete[] opt;
   opt = 0;
   return nsel;
}

//______________________________________________________________________________
TSQLResult* TTree::Query(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Loop over entries and return a TSQLResult object containing entries following selection.

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Query(varexp, selection, option, nentries, firstentry);
   }
   return 0;
}

//______________________________________________________________________________
Long64_t TTree::ReadFile(const char* filename, const char* branchDescriptor)
{
   // -- Create or simply read branches from filename.
   //
   // if branchDescriptor = "" (default), it is assumed that the Tree descriptor
   //    is given in the first line of the file with a syntax like
   //     A/D:Table[2]/F:Ntracks/I:astring/C
   //  otherwise branchDescriptor must be specified with the above syntax.
   //
   // Lines in the input file starting with "#" are ignored.
   //
   // A TBranch object is created for each variable in the expression.
   // The total number of rows read from the file is returned.

   gTree = this;
   std::ifstream in;
   in.open(filename);
   if (!in.good()) {
      Error("ReadFile","Cannot open file: %s",filename);
      return 0;
   }

   TBranch *branch;
   char *bdname = new char[1000];
   char *bd = new char[10000];
   Int_t nch = 0;
   if (branchDescriptor) nch = strlen(branchDescriptor);
   // branch Descriptor is null, read its definition from the first line in the file
   if (!nch) {
      in >> bd;
      if (!in.good()) {
         Error("ReadFile","Error reading file: %s",filename);
         return 0;
      }
      in.ignore(8192,'\n');
      nch = strlen(bd);
   } else {
      strcpy(bd,branchDescriptor);
   }

   //parse the branch descriptor and create a branch for each element
   //separated by ":"
   void *address = &bd[9000];
   char *bdcur = bd;
   TString desc="", olddesc="";
   while (bdcur) {
      char *colon = strchr(bdcur,':');
      if (colon) *colon = 0;
      strcpy(bdname,bdcur);
      char *slash = strchr(bdname,'/');
      if (slash) {
         *slash = 0;
         desc = bdcur;
         olddesc = slash+1;
      } else {
         desc = Form("%s/%s",bdname,olddesc.Data());
      }
      branch = new TBranch(bdname,address,desc.Data(),32000);
      if (branch->IsZombie()) {
         delete branch;
         Warning("ReadFile","Illegal branch definition: %s",bdcur);
      } else {
         fBranches.Add(branch);
         branch->SetAddress(0);
      }
      if (!colon)break;
      bdcur = colon+1;
   }

   //loop on all lines in the file
   Int_t nbranches = fBranches.GetEntries();
   Int_t status = 1;
   Long64_t nlines = 0;
   while(status > 0) {

      if ( in.peek() != '#' ) {
         //loop on branches and read the branch values into their buffer
         for (Int_t i=0;i<nbranches;i++) {
            branch = (TBranch*)fBranches.At(i);
            TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
            leaf->ReadValue(in);
            status = in.good();
            if (status <= 0) break;
         }
         if (status <= 0) break;
         //we are now ready to fill the tree
         Fill();
         nlines++;
      }
      in.ignore(8192,'\n');
   }

   delete [] bdname;
   delete [] bd;
   return nlines;
}

//______________________________________________________________________________
void TTree::Refresh()
{
   //  -- Refresh contents of this tree and its branches from the current status on disk.
   //
   //  One can call this function in case the tree file is being
   //  updated by another process.

   if (!fDirectory->GetFile()) {
      return;
   }
   fDirectory->ReadKeys();
   fDirectory->GetList()->Remove(this);
   TTree* tree = (TTree*) fDirectory->Get(GetName());
   if (!tree) {
      return;
   }
   //copy info from tree header into this Tree
   fEntries = tree->fEntries;
   fTotBytes = tree->fTotBytes;
   fZipBytes = tree->fZipBytes;
   fSavedBytes = tree->fSavedBytes;
   fTotalBuffers = tree->fTotalBuffers;

   //loop on all branches and update them
   Int_t nleaves = fLeaves.GetEntriesFast();
   for (Int_t i = 0; i < nleaves; i++)  {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      TBranch* branch = (TBranch*) leaf->GetBranch();
      branch->Refresh(tree->GetBranch(branch->GetName()));
   }
   fDirectory->GetList()->Remove(tree);
   fDirectory->GetList()->Add(this);
   delete tree;
   tree = 0;
}

//______________________________________________________________________________
void TTree::RemoveFriend(TTree* oldFriend)
{
   // -- Remove a friend from the list of friends.

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kRemoveFriend & fFriendLockStatus) {
      return;
   }
   if (!fFriends) {
      return;
   }
   TFriendLock lock(this, kRemoveFriend);
   TIter nextf(fFriends);
   TFriendElement* fe = 0;
   while ((fe = (TFriendElement*) nextf())) {
      TTree* friend_t = fe->GetTree();
      if (friend_t == oldFriend) {
         fFriends->Remove(fe);
         delete fe;
         fe = 0;
      }
   }
}

//______________________________________________________________________________
void TTree::Reset(Option_t* option)
{
   // -- Reset baskets, buffers and entries count in all branches and leaves.

   fNotify = 0;
   fEntries = 0;
   fTotBytes = 0;
   fZipBytes = 0;
   fSavedBytes = 0;
   fTotalBuffers = 0;
   fChainOffset = 0;
   fReadEntry = -1;

   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i)  {
      TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
      branch->Reset(option);
   }

   if (fBranchRef) {
      fBranchRef->Reset();
   }
}

//______________________________________________________________________________
void TTree::ResetBranchAddresses()
{
   // -- Tell all of our branches to drop their current objects and allocate new ones.

   TObjArray* branches = GetListOfBranches();
   Int_t nbranches = branches->GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* branch = (TBranch*) branches->UncheckedAt(i);
      branch->ResetAddress();
   }
}

//______________________________________________________________________________
Long64_t TTree::Scan(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Loop over tree entries and print entries passing selection.
   //
   // If varexp is 0 (or "") then print only first 8 columns.
   // If varexp = "*" print all columns.
   // Otherwise a columns selection can be made using "var1:var2:var3".
   // See TTreePlayer::Scan for more information
   //

   GetPlayer();
   if (fPlayer) {
      return fPlayer->Scan(varexp, selection, option, nentries, firstentry);
   }
   return -1;
}

//______________________________________________________________________________
Bool_t TTree::SetAlias(const char* aliasName, const char* aliasFormula)
{
   // -- Set a tree variable alias.
   //
   //  Set an alias for an expression/formula based on the tree 'variables'.
   //
   //  The content of 'aliasName' can be used in TTreeFormula (i.e. TTree::Draw,
   //  TTree::Scan, TTreeViewer) and will be evaluated as the content of
   //  'aliasFormula'.
   //  If the content of 'aliasFormula' only contains symbol names, periods and
   //  array index specification (for example event.fTracks[3]), then
   //  the content of 'aliasName' can be used as the start of symbol.
   //
   //  If the alias 'aliasName' already existed, it is replaced by the new
   //  value.
   //
   //  When being used, the alias can be preceded by an eventual 'Friend Alias'
   //  (see TTree::GetFriendAlias)
   //
   //  Return true if it was added properly.
   //
   //  For example:
   //     tree->SetAlias("x1","(tdc1[1]-tdc1[0])/49");
   //     tree->SetAlias("y1","(tdc1[3]-tdc1[2])/47");
   //     tree->SetAlias("x2","(tdc2[1]-tdc2[0])/49");
   //     tree->SetAlias("y2","(tdc2[3]-tdc2[2])/47");
   //     tree->Draw("y2-y1:x2-x1");
   //
   //     tree->SetAlias("theGoodTrack","event.fTracks[3]");
   //     tree->Draw("theGoodTrack.fPx"); // same as "event.fTracks[3].fPx"

   if (!aliasName || !aliasFormula) {
      return kFALSE;
   }
   if (!strlen(aliasName) || !strlen(aliasFormula)) {
      return kFALSE;
   }
   if (!fAliases) {
      fAliases = new TList;
   } else {
      TNamed* oldHolder = (TNamed*) fAliases->FindObject(aliasName);
      if (oldHolder) {
         oldHolder->SetTitle(aliasFormula);
         return kTRUE;
      }
   }
   TNamed* holder = new TNamed(aliasName, aliasFormula);
   fAliases->Add(holder);
   return kTRUE;
}

//_______________________________________________________________________
void TTree::SetBasketSize(const char* bname, Int_t buffsize)
{
   // -- Set a branch's basket size.
   //
   // bname is the name of a branch.
   // if bname="*", apply to all branches.
   // if bname="xxx*", apply to all branches with name starting with xxx
   // see TRegexp for wildcarding options
   // buffsize = branc basket size
   //

   Int_t nleaves = fLeaves.GetEntriesFast();
   TRegexp re(bname, kTRUE);
   Int_t nb = 0;
   for (Int_t i = 0; i < nleaves; i++)  {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      TBranch* branch = (TBranch*) leaf->GetBranch();
      TString s = branch->GetName();
      if (strcmp(bname, branch->GetName()) && (s.Index(re) == kNPOS)) {
         continue;
      }
      nb++;
      branch->SetBasketSize(buffsize);
   }
   if (!nb) {
      Error("SetBasketSize", "unknown branch -> '%s'", bname);
   }
}

//_______________________________________________________________________
void TTree::SetBranchAddress(const char* bname, void* addr, TBranch** ptr)
{
   // -- Change branch address, dealing with clone trees properly.

   TBranch* branch = GetBranch(bname);
   if (!branch) {
      Error("SetBranchAddress", "unknown branch -> %s", bname);
      return;
   }
   if (ptr) {
      *ptr = branch;
   }
   if (fClones) {
      void* oldAddr = branch->GetAddress();
      TIter next(fClones);
      TTree* clone = 0;
      while ((clone = (TTree*) next())) {
         TBranch* cloneBr = clone->GetBranch(bname);
         if (cloneBr && (cloneBr->GetAddress() == oldAddr)) {
            cloneBr->SetAddress(addr);
         }
      }
   }
   branch->SetAddress(addr);
}

//_______________________________________________________________________
void TTree::SetBranchAddress(const char* bname, void* add, TClass* ptrClass, EDataType datatype, Bool_t isptr)
{
   // -- Verify the validity of the type of add before calling SetBranchAddress.

   SetBranchAddress(bname, add, 0, ptrClass, datatype, isptr);
}

//_______________________________________________________________________
void TTree::SetBranchAddress(const char* bname, void* add, TBranch** ptr, TClass* ptrClass, EDataType datatype, Bool_t isptr)
{
   // -- Verify the validity of the type of add before calling SetBranchAddress.

   TBranch* branch = GetBranch(bname);
   if (!branch) {
      Error("SetBranchAddress", "unknown branch -> %s", bname);
      return;
   }
   if (ptr) {
      *ptr = branch;
   }
   CheckBranchAddressType(branch, ptrClass, datatype, isptr);
   SetBranchAddress(bname, add);
}

//_______________________________________________________________________
void TTree::SetBranchStatus(const char* bname, Bool_t status, UInt_t* found)
{
   // -- Set branch status to Process or DoNotProcess.
   //
   //  When reading a Tree, by default, all branches are read.
   //  One can speed up considerably the analysis phase by activating
   //  only the branches that hold variables involved in a query.
   //
   //     bname is the name of a branch.
   //     if bname="*", apply to all branches.
   //     if bname="xxx*", apply to all branches with name starting with xxx
   //     see TRegexp for wildcarding options
   //      status = 1  branch will be processed
   //             = 0  branch will not be processed
   //    Example:
   //  Assume a tree T with sub-branches a,b,c,d,e,f,g,etc..
   //  when doing T.GetEntry(i) all branches are read for entry i.
   //  to read only the branches c and e, one can do
   //    T.SetBranchStatus("*",0); //disable all branches
   //    T.SetBranchStatus("c",1);
   //    T.setBranchStatus("e",1);
   //    T.GetEntry(i);
   //
   //  WARNING! WARNING! WARNING!
   //  SetBranchStatus is matching the branch based on regular expression match 
   //  of the branch 'name' and not on the branch hierarchy!
   //  In order to be able to selectively enable a top level object that is 'split' 
   //  you need to make sure the name of the top level branch is prefixed to the 
   //  sub-branches' name(by adding a dot ('.') at the end of the Branch creation 
   //  and use the corresponding regular expression. 
   //
   //  I.e If your Tree has been created in split mode with a parent branch "parent."
   //  (note the trailing dot).
   //     T.SetBranchStatus("parent",1);
   //  will not activate the sub-branches of "parent". You should do:
   //     T.SetBranchStatus("parent*",1); 
   //
   //  Without the trailing dot in the branch creation you have no choice but to
   //  call SetBranchStatus explicitly for each of the sub branches.
   //
   //
   //  An alternative to this function is to read directly and only
   //  the interesting branches. Example:
   //    TBranch *brc = T.GetBranch("c");
   //    TBranch *bre = T.GetBranch("e");
   //    brc->GetEntry(i);
   //    bre->GetEntry(i);
   //
   //  If found is not 0, the number of branch(es) found matching the regular
   //  expression is returned in *found AND the error message 'unknown branch'
   //  is suppressed.

   // We already have been visited while recursively looking
   // through the friends tree, let return
   if (kSetBranchStatus & fFriendLockStatus) {
      return;
   }

   TBranch *branch, *bcount, *bson;
   TLeaf *leaf, *leafcount;

   Int_t i,j;
   Int_t nleaves = fLeaves.GetEntriesFast();
   TRegexp re(bname,kTRUE);
   Int_t nb = 0;

   // first pass, loop on all branches
   // for leafcount branches activate/deactivate in function of status
   for (i=0;i<nleaves;i++)  {
      leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      TString s = branch->GetName();
      if (strcmp(bname,"*")) { //Regexp gives wrong result for [] in name
         TString longname; 
         longname.Form("%s.%s",GetName(),branch->GetName());
         if (strcmp(bname,branch->GetName()) 
             && longname != bname
             && s.Index(re) == kNPOS) continue;
      }
      nb++;
      if (status) branch->ResetBit(kDoNotProcess);
      else        branch->SetBit(kDoNotProcess);
      leafcount = leaf->GetLeafCount();
      if (leafcount) {
         bcount = leafcount->GetBranch();
         if (status) bcount->ResetBit(kDoNotProcess);
         else        bcount->SetBit(kDoNotProcess);
      }
   }

   //search in list of friends
   UInt_t foundInFriend = 0;
   if (fFriends) {
      TFriendLock lock(this,kSetBranchStatus);
      TIter nextf(fFriends);
      TFriendElement *fe;
      TString name;
      while ((fe = (TFriendElement*)nextf())) {
         TTree *t = fe->GetTree();
         if (t==0) continue;

         // If the alias is present replace it with the real name.
         char *subbranch = (char*)strstr(bname,fe->GetName());
         if (subbranch!=bname) subbranch = 0;
         if (subbranch) {
            subbranch += strlen(fe->GetName());
            if ( *subbranch != '.' ) subbranch = 0;
            else subbranch ++;
         }
         if (subbranch) {
            name.Form("%s.%s",t->GetName(),subbranch);
         } else {
            name = bname;
         }
         t->SetBranchStatus(name,status, &foundInFriend);
      }
   }
   if (!nb && !foundInFriend) {
      if (found==0) Error("SetBranchStatus", "unknown branch -> %s", bname);
      return;
   }
   if (found) *found = nb + foundInFriend;

   // second pass, loop again on all branches
   // activate leafcount branches for active branches only
   for (i = 0; i < nleaves; i++) {
      leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      if (!branch->TestBit(kDoNotProcess)) {
         leafcount = leaf->GetLeafCount();
         if (leafcount) {
            bcount = leafcount->GetBranch();
            bcount->ResetBit(kDoNotProcess);
         }
      } else {
         //Int_t nbranches = branch->GetListOfBranches()->GetEntriesFast();
         Int_t nbranches = branch->GetListOfBranches()->GetEntries();
         for (j=0;j<nbranches;j++) {
            bson = (TBranch*)branch->GetListOfBranches()->UncheckedAt(j);
            if (!bson) continue;
            if (!bson->TestBit(kDoNotProcess)) {
               if (bson->GetNleaves() <= 0) continue;
               branch->ResetBit(kDoNotProcess);
               break;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TTree::SetBranchStyle(Int_t style)
{
   // -- Set the current branch style.  (static function)
   //
   // style = 0 old Branch
   // style = 1 new Bronch

   fgBranchStyle = style;
}

 //______________________________________________________________________________
void TTree::SetCacheSize(Long64_t cacheSize)
{
   // -- Set maximum size of the file cache (default is 10000000, i.e., 10 MB).
   //if cachesize <= 0 the existing cache (if any) is deleted

   TFile* file = GetCurrentFile();
   if (!file) {
      fCacheSize = cacheSize;
      return;
   }
   TFileCacheRead* pf = file->GetCacheRead();
   if (pf) {
      if (cacheSize == fCacheSize) {
         return;
      }
      delete pf;
      pf = 0;
      if (cacheSize <= 0) {
         file->SetCacheRead(0);
         fCacheSize=0;
         return;
      }
   }
   fCacheSize = cacheSize;
   if (cacheSize <= 0) {
      return;
   }
   new TTreeCache(this, cacheSize);
}

//______________________________________________________________________________
void TTree::SetCircular(Long64_t maxEntries)
{
   // -- Enable/Disable circularity for this tree.
   //
   // if maxEntries > 0 a maximum of maxEntries is kept in one buffer/basket
   // per branch in memory.
   //   Note that when this function is called (maxEntries>0) the Tree
   //   must be empty or having only one basket per branch.
   // if maxEntries <= 0 the tree circularity is disabled.
   //
   // NOTE 1:
   //  Circular Trees are interesting in online real time environments
   //  to store the results of the last maxEntries events.
   // NOTE 2:
   //  Calling SetCircular with maxEntries <= 0 is necessary before
   //  merging circular Trees that have been saved on files.
   // NOTE 3:
   //  SetCircular with maxEntries <= 0 is automatically called
   //  by TChain::Merge
   // NOTE 4:
   //  A circular Tree can still be saved in a file. When read back,
   //  it is still a circular Tree and can be filled again.

   if (maxEntries <= 0) {
      // -- Disable circularity.
      fMaxEntries = 1000000000;
      fMaxEntries *= 1000;
      ResetBit(kCircular);
      //in case the Tree was originally created in gROOT, the branch
      //compression level was set to -1. If the Tree is now associated to
      //a file, reset the compression level to the file compression level
      if (fDirectory) {
         TFile* bfile = fDirectory->GetFile();
         Int_t compress = 1;
         if (bfile) {
            compress = bfile->GetCompressionLevel();
         }
         Int_t nb = fBranches.GetEntriesFast();
         for (Int_t i = 0; i < nb; i++) {
            TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
            branch->SetCompressionLevel(compress);
         }
      }
   } else {
      // -- Enable circularity.
      fMaxEntries = maxEntries;
      SetBit(kCircular);
   }
}

//______________________________________________________________________________
void TTree::SetDebug(Int_t level, Long64_t min, Long64_t max)
{
   // -- Set the debug level and the debug range.
   //
   // For entries in the debug range, the functions TBranchElement::Fill
   // and TBranchElement::GetEntry will print the number of bytes filled
   // or read for each branch.

   fDebug = level;
   fDebugMin = min;
   fDebugMax = max;
}

//______________________________________________________________________________
void TTree::SetDirectory(TDirectory* dir)
{
   // -- Change the tree's directory.
   //
   // Remove reference to this tree from current directory and
   // add reference to new directory dir.  The dir parameter can
   // be 0 in which case the tree does not belong to any directory.
   //

   if (fDirectory == dir) {
      return;
   }
   if (fDirectory) {
      fDirectory->GetList()->Remove(this);
   }
   fDirectory = dir;
   if (fDirectory) {
      fDirectory->GetList()->Add(this);
   }
   TFile* file = 0;
   if (fDirectory) {
      file = fDirectory->GetFile();
   }
   TBranch* b = 0;
   TIter next(GetListOfBranches());
   while((b = (TBranch*) next())) {
      b->SetFile(file);
   }
}

//_______________________________________________________________________
Long64_t TTree::SetEntries(Long64_t n)
{
   // -- Change number of entries in the tree.
   //
   // If n >= 0, set number of entries in the tree = n.
   //
   // If n < 0, set number of entries in the tree to match the
   // number of entries in each branch. (default for n is -1)
   //
   // This function should be called only when one fills each branch
   // independently via TBranch::Fill without calling TTree::Fill.
   // Calling TTree::SetEntries() make sense only if the number of entries
   // in each branch is identical, a warning is issued otherwise.
   // The function returns the number of entries.
   //

   // case 1 : force number of entries to n
   if (n >= 0) {
      fEntries = n;
      return n;
   }

   // case 2; compute the number of entries from the number of entries in the branches
   TBranch* b = 0;
   Long64_t nMin = 99999999;
   Long64_t nMax = 0;
   TIter next(GetListOfBranches());
   while((b = (TBranch*) next())){
      Long64_t n = b->GetEntries();
      if (n < nMin) {
         nMin = n;
      }
      if (n > nMax) {
         nMax = n;
      }
   }
   if (nMin != nMax) {
      Warning("SetEntries", "Tree branches have different numbers of entries, with %lld maximum.", nMax);
   }
   fEntries = nMax;
   return fEntries;
}

//_______________________________________________________________________
void TTree::SetEntryList(TEntryList *enlist)
{
   //Set an EntryList
   
   if (fEntryList) {
      //check if the previous entry list is owned by the tree
      if (fEntryList->TestBit(kCanDelete)){
         delete fEntryList;
      }
   }
   fEventList = 0;
   if (!enlist) {
      fEntryList = 0;
      return;
   }
   fEntryList = enlist;
   fEntryList->SetTree(this);

}

//_______________________________________________________________________
void TTree::SetEventList(TEventList *evlist)
{
//This function transfroms the given TEventList into a TEntryList
//The new TEntryList is owned by the TTree and gets deleted when the tree
//is deleted. This TEntryList can be returned by GetEntryList() function, and after
//GetEntryList() function is called, the TEntryList is not owned by the tree 
//any more.

   if (!evlist) {
      if (fEntryList){
         if (fEntryList->TestBit(kCanDelete)){
            delete fEntryList;
         }
      }
      fEntryList = 0;
      fEventList = 0;
      return;
   }

   fEventList = evlist;
   fEntryList = new TEntryList(evlist->GetName(), evlist->GetTitle());
   Int_t nsel = evlist->GetN();
   fEntryList->SetTree(this);
   Long64_t entry;
   for (Int_t i=0; i<nsel; i++){
      entry = evlist->GetEntry(i);
      fEntryList->Enter(entry);
   }
   fEntryList->SetBit(kCanDelete, kTRUE);
}

//_______________________________________________________________________
void TTree::SetEstimate(Long64_t n)
{
   // -- Set number of entries to estimate variable limits.

   if (n <= 0) {
      n = 10000;
   }
   fEstimate = n;
   GetPlayer();
   if (fPlayer) {
      fPlayer->SetEstimate(n);
   }
}

//_______________________________________________________________________
void TTree::SetFileNumber(Int_t number)
{
   // -- Set fFileNumber to number.
   // fFileNumber is used by TTree::Fill to set the file name
   // for a new file to be created when the current file exceeds fgTreeMaxSize.
   //    (see TTree::ChangeFile)
   // if fFileNumber=10, the new file name will have a suffix "_11",
   // ie, fFileNumber is incremented before setting the file name

   if (fFileNumber < 0) {
      Warning("SetFileNumber", "file number must be positive. Set to 0");
      fFileNumber = 0;
      return;
   }
   fFileNumber = number;
}

//______________________________________________________________________________
void TTree::SetMaxTreeSize(Long64_t maxsize)
{
   // -- Set the maximum size of a Tree file -- (static function).
   //
   // In TTree::Fill, when the file has a size > fgMaxTreeSize,
   // the function closes the current file and starts writing into
   // a new file with a name of the style "file_1.root" if the original
   // requested file name was "file.root".
   //

   fgMaxTreeSize = maxsize;
}

//______________________________________________________________________________
void TTree::SetName(const char* name)
{
   // -- Change the name of this tree.

   if (gPad) {
      gPad->Modified();
   }
   // Trees are named objects in a THashList.
   // We must update the hashlist if we change the name.
   if (fDirectory) {
      fDirectory->GetList()->Remove(this);
   }
   fName = name;
   if (fDirectory) {
      fDirectory->GetList()->Add(this);
   }
}

//______________________________________________________________________________
void TTree::SetObject(const char* name, const char* title)
{
   // -- Change the name and title of this tree.

   if (gPad) {
      gPad->Modified();
   }

   //  Trees are named objects in a THashList.
   //  We must update the hashlist if we change the name
   if (fDirectory) {
      fDirectory->GetList()->Remove(this);
   }
   fName = name;
   fTitle = title;
   if (fDirectory) {
      fDirectory->GetList()->Add(this);
   }
}

//______________________________________________________________________________
void TTree::SetTreeIndex(TVirtualIndex* index)
{
   // -- The current TreeIndex is replaced by the new index.
   // Note that this function does not delete the previous index.
   // This gives the possibility to play with more than one index, e.g.,
   // TVirtualIndex* oldIndex = tree.GetTreeIndex();
   // tree.SetTreeIndex(newIndex);
   // tree.Draw();
   // tree.SetTreeIndex(oldIndex);
   // tree.Draw(); etc

   fTreeIndex = index;
}

//______________________________________________________________________________
void TTree::SetWeight(Double_t w, Option_t*)
{
   // -- Set tree weight.
   //
   // The weight is used by TTree::Draw to automatically weight each
   // selected entry in the resulting histogram.
   //
   // For example the equivalent of:
   //
   //      T.Draw("x", "w")
   //
   // is:
   //
   //      T.SetWeight(w);
   //      T.Draw("x");
   //
   // This function is redefined by TChain::SetWeight. In case of a
   // TChain, an option "global" may be specified to set the same weight
   // for all trees in the TChain instead of the default behaviour
   // using the weights of each tree in the chain (see TChain::SetWeight).

   fWeight = w;
}

//______________________________________________________________________________
void TTree::Show(Long64_t entry, Int_t lenmax)
{
   // -- Print values of all active leaves for entry.
   //
   // if entry==-1, print current entry (default)
   // if a leaf is an array, a maximum of lenmax elements is printed.
   //
   if (entry != -1) {
      GetEntry(entry);
   }
   printf("======> EVENT:%lld\n", fReadEntry);
   TObjArray* leaves  = GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   Int_t ltype;
   for (Int_t i = 0; i < nleaves; i++) {
      TLeaf* leaf = (TLeaf*) leaves->UncheckedAt(i);
      TBranch* branch = leaf->GetBranch();
      if (branch->TestBit(kDoNotProcess)) {
         continue;
      }
      Int_t len = leaf->GetLen();
      if (len <= 0) {
         continue;
      }
      len = TMath::Min(len, lenmax);
      if (leaf->IsA() == TLeafElement::Class()) {
         leaf->PrintValue(lenmax);
         continue;
      }
      if (branch->GetListOfBranches()->GetEntriesFast() > 0) {
         continue;
      }
      ltype = 10;
      if (leaf->IsA() == TLeafF::Class()) {
         ltype = 5;
      }
      if (leaf->IsA() == TLeafD::Class()) {
         ltype = 5;
      }
      if (leaf->IsA() == TLeafC::Class()) {
         len = 1;
         ltype = 5;
      };
      printf(" %-15s = ", leaf->GetName());
      for (Int_t l = 0; l < len; l++) {
         leaf->PrintValue(l);
         if (l == (len - 1)) {
            printf("\n");
            continue;
         }
         printf(", ");
         if ((l % ltype) == 0) {
            printf("\n                  ");
         }
      }
   }
}

//______________________________________________________________________________
void TTree::StartViewer()
{
   // -- Start the TTreeViewer on this tree.
   //
   //  ww is the width of the canvas in pixels
   //  wh is the height of the canvas in pixels

   GetPlayer();
   if (fPlayer) {
      fPlayer->StartViewer(600, 400);
   }
}

//______________________________________________________________________________
void TTree::Streamer(TBuffer& b)
{
   // -- Stream a class object.
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      gTree = this;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 4) {
         fDirectory = gDirectory;
         TTree::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         if (fTreeIndex) {
            fTreeIndex->SetTree(this);
         }
         if (fIndex.fN) {
            Warning("Streamer", "Old style index in this tree is deleted. Rebuild the index via TTree::BuildIndex");
            fIndex.Set(0);
            fIndexValues.Set(0);
         }
         if (fEstimate <= 10000) {
            fEstimate = 1000000;
         }
         fSavedBytes = fTotBytes;
         gDirectory->Append(this);
         return;
      }
      //====process old versions before automatic schema evolution
      Stat_t djunk;
      Int_t ijunk;
      TNamed::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      TAttMarker::Streamer(b);
      b >> fScanField;
      b >> ijunk; fMaxEntryLoop   = (Long64_t)ijunk;
      b >> ijunk; fMaxVirtualSize = (Long64_t)ijunk;
      b >> djunk; fEntries  = (Long64_t)djunk;
      b >> djunk; fTotBytes = (Long64_t)djunk;
      b >> djunk; fZipBytes = (Long64_t)djunk;
      b >> ijunk; fAutoSave = (Long64_t)ijunk;
      b >> ijunk; fEstimate = (Long64_t)ijunk;
      if (fEstimate <= 10000) fEstimate = 1000000;
      fBranches.Streamer(b);
      fLeaves.Streamer(b);
      fSavedBytes = fTotBytes;
      fDirectory = gDirectory;
      gDirectory->Append(this);
      if (R__v > 1) fIndexValues.Streamer(b);
      if (R__v > 2) fIndex.Streamer(b);
      if (R__v > 3) {
         TList OldInfoList;
         OldInfoList.Streamer(b);
         OldInfoList.Delete();
      }
      b.CheckByteCount(R__s, R__c, TTree::IsA());
      //====end of old versions
   } else {
      if (fBranchRef) {
         fBranchRef->Clear();
      }
      TTree::Class()->WriteBuffer(b, this);
   }
}

//______________________________________________________________________________
Long64_t TTree::UnbinnedFit(const char* funcname, const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
   // -- Unbinned fit of one or more variable(s) from a tree.
   //
   //  funcname is a TF1 function.
   //
   //  See TTree::Draw for explanations of the other parameters.
   //
   //   Fit the variable varexp using the function funcname using the
   //   selection cuts given by selection.
   //
   //   The list of fit options is given in parameter option.
   //      option = "Q" Quiet mode (minimum printing)
   //             = "V" Verbose mode (default is between Q and V)
   //             = "E" Perform better Errors estimation using Minos technique
   //             = "M" More. Improve fit results
   //
   //   You can specify boundary limits for some or all parameters via
   //        func->SetParLimits(p_number, parmin, parmax);
   //   if parmin>=parmax, the parameter is fixed
   //   Note that you are not forced to fix the limits for all parameters.
   //   For example, if you fit a function with 6 parameters, you can do:
   //     func->SetParameters(0,3.1,1.e-6,0.1,-8,100);
   //     func->SetParLimits(4,-10,-4);
   //     func->SetParLimits(5, 1,1);
   //   With this setup, parameters 0->3 can vary freely
   //   Parameter 4 has boundaries [-10,-4] with initial value -8
   //   Parameter 5 is fixed to 100.
   //
   //   For the fit to be meaningful, the function must be self-normalized.
   //
   //   i.e. It must have the same integral regardless of the parameter
   //   settings.  Otherwise the fit will effectively just maximize the
   //   area.
   //
   //   It is mandatory to have a normalization variable
   //   which is fixed for the fit.  e.g.
   //
   //     TF1* f1 = new TF1("f1", "gaus(0)/sqrt(2*3.14159)/[2]", 0, 5);
   //     f1->SetParameters(1, 3.1, 0.01);
   //     f1->SetParLimits(0, 1, 1); // fix the normalization parameter to 1
   //     data->UnbinnedFit("f1", "jpsimass", "jpsipt>3.0");
   //   //
   //
   //   1, 2 and 3 Dimensional fits are supported.
   //   See also TTree::Fit

   GetPlayer();
   if (fPlayer) {
      return fPlayer->UnbinnedFit(funcname, varexp, selection, option, nentries, firstentry);
   }
   return -1;
}

//______________________________________________________________________________
void TTree::UseCurrentStyle()
{
   // -- Replace current attributes by current style.

   if (gStyle->IsReading()) {
      SetFillColor(gStyle->GetHistFillColor());
      SetFillStyle(gStyle->GetHistFillStyle());
      SetLineColor(gStyle->GetHistLineColor());
      SetLineStyle(gStyle->GetHistLineStyle());
      SetLineWidth(gStyle->GetHistLineWidth());
      SetMarkerColor(gStyle->GetMarkerColor());
      SetMarkerStyle(gStyle->GetMarkerStyle());
      SetMarkerSize(gStyle->GetMarkerSize());
   } else {
      gStyle->SetHistFillColor(GetFillColor());
      gStyle->SetHistFillStyle(GetFillStyle());
      gStyle->SetHistLineColor(GetLineColor());
      gStyle->SetHistLineStyle(GetLineStyle());
      gStyle->SetHistLineWidth(GetLineWidth());
      gStyle->SetMarkerColor(GetMarkerColor());
      gStyle->SetMarkerStyle(GetMarkerStyle());
      gStyle->SetMarkerSize(GetMarkerSize());
   }
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFriendLeafIter                                                  //
//                                                                      //
// Iterator on all the leaves in a TTree and its friend                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TTreeFriendLeafIter)

//______________________________________________________________________________
TTreeFriendLeafIter::TTreeFriendLeafIter(const TTree* tree, Bool_t dir)
: fTree(const_cast<TTree*>(tree))
, fLeafIter(0)
, fTreeIter(0)
, fDirection(dir)
{
   // Create a new iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.
}

//______________________________________________________________________________
TTreeFriendLeafIter::TTreeFriendLeafIter(const TTreeFriendLeafIter& iter)
: TIterator(iter)
{
   // Copy constructor

   fTree = iter.fTree;
   fDirection = iter.fDirection;
}

//______________________________________________________________________________
TIterator& TTreeFriendLeafIter::operator=(const TIterator& rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TTreeFriendLeafIter::Class()) {
      const TTreeFriendLeafIter &rhs1 = (const TTreeFriendLeafIter &)rhs;
      fDirection = rhs1.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TTreeFriendLeafIter& TTreeFriendLeafIter::operator=(const TTreeFriendLeafIter& rhs)
{
   // Overridden assignment operator.

   if (this != &rhs) {
      fDirection = rhs.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TObject* TTreeFriendLeafIter::Next()
{
   // Go the next friend element

   if (!fTree) return 0;

   TObject * next;
   TTree * nextTree;

   if (!fLeafIter) {
      TObjArray *list = fTree->GetListOfLeaves();
      if (!list) return 0; // Can happen with an empty chain.
      fLeafIter =  list->MakeIterator(fDirection);
   }

   next = fLeafIter->Next();
   if (!next) {
      if (!fTreeIter) {
         TCollection * list = fTree->GetListOfFriends();
         if (!list) return next;
         fTreeIter = list->MakeIterator(fDirection);
      }
      TFriendElement * nextFriend = (TFriendElement*) fTreeIter->Next();
      ///nextTree = (TTree*)fTreeIter->Next();
      if (nextFriend) {
         nextTree = const_cast<TTree*>(nextFriend->GetTree());
         if (!nextTree) return Next();
         SafeDelete(fLeafIter);
         fLeafIter = nextTree->GetListOfLeaves()->MakeIterator(fDirection);
         next = fLeafIter->Next();
      }
   }
   return next;
}

//______________________________________________________________________________
Option_t* TTreeFriendLeafIter::GetOption() const
{
   // Returns the object option stored in the list.

   if (fLeafIter) return fLeafIter->GetOption();
   return "";
}

