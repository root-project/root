// @(#)root/tree:$Name:  $:$Id: TTree.cxx,v 1.116 2002/03/19 17:05:49 brun Exp $
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
//  ==> TTree *tree = new TTree(name, title, maxvirtualsize)
//     Creates a Tree with name and title. Maxvirtualsize is by default 64Mbytes,
//     maxvirtualsize = 64000000(default) means: Keeps as many buffers in memory until
//     the sum of all buffers is greater than 64 Megabyte. When this happens,
//     memory buffers are written to disk and deleted until the size of all
//     buffers is again below the threshold.
//     maxvirtualsize = 0 means: keep only one buffer in memory.
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
//   Int_t nentries = (Int_t)t3->GetEntries();
//
//   for (Int_t i = 0; i < nentries; i++){
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

#include <string.h>
#include <stdio.h>

#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TEventList.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TLeafObject.h"
#include "TLeaf.h"
#include "TLeafB.h"
#include "TLeafC.h"
#include "TLeafI.h"
#include "TLeafF.h"
#include "TLeafS.h"
#include "TLeafD.h"
#include "TLeafElement.h"
#include "TBasket.h"
#include "TMath.h"
#include "TDirectory.h"
#include "TBranchClones.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBrowser.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TRegexp.h"
#include "TArrayC.h"
#include "TFolder.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TFriendElement.h"
#include "TVirtualFitter.h"
#include "Api.h"

Int_t TTree::fgBranchStyle = 1;  //use new TBranch style with TBranchElement

TTree *gTree;
const Int_t kMaxLen = 512;

ClassImp(TTree)

//______________________________________________________________________________
TTree::TTree(): TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*Default Tree constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   fScanField      = 25;
   fMaxEntryLoop   = 1000000000;
   fMaxVirtualSize = 0;
   fDirectory      = 0;
   fEntries        = 0;
   fTotBytes       = 0;
   fZipBytes       = 0;
   fWeight         = 1;
   fAutoSave       = 100000000;
   fSavedBytes     = 0;
   fTotalBuffers   = 0;
   fChainOffset    = 0;
   fReadEntry      = -1;
   fEstimate       = 1000000;
   fUpdate         = 0;
   fEventList      = 0;
   fPacketSize     = 100;
   fTimerInterval  = 0;
   fPlayer         = 0;
   fDebug          = 0;
   fDebugMin       = 0;
   fDebugMax       = 9999999;
   fFriends        = 0;
   fMakeClass      = 0;
   fNotify         = 0;
}

//______________________________________________________________________________
TTree::TTree(const char *name,const char *title, Int_t splitlevel)
    :TNamed(name,title)
{
//*-*-*-*-*-*-*-*-*-*Normal Tree constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ======================
//
//   The Tree is created in the current directory
//   Use the various functions Branch below to add branches to this Tree.
//
// If the first character of title is a "/", the function assumes a folder name.
// In this case, it creates automatically branches following the folder hierarchy.
// splitlevel may be used in this case to control the split level.

   fScanField      = 25;
   fMaxEntryLoop   = 1000000000;
   fMaxVirtualSize = 0;
   fDirectory      = gDirectory;
   fEntries        = 0;
   fTotBytes       = 0;
   fZipBytes       = 0;
   fWeight         = 1;
   fAutoSave       = 100000000;
   fSavedBytes     = 0;
   fTotalBuffers   = 0;
   fChainOffset    = 0;
   fReadEntry      = -1;
   fEstimate       = 1000000;
   fUpdate         = 0;
   fEventList      = 0;
   fPacketSize     = 100;
   fTimerInterval  = 0;
   fPlayer         = 0;
   fDebug          = 0;
   fDebugMin       = 0;
   fDebugMax       = 9999999;
   fFriends        = 0;
   fMakeClass      = 0;
   fNotify         = 0;

   SetFillColor(gStyle->GetHistFillColor());
   SetFillStyle(gStyle->GetHistFillStyle());
   SetLineColor(gStyle->GetHistLineColor());
   SetLineStyle(gStyle->GetHistLineStyle());
   SetLineWidth(gStyle->GetHistLineWidth());
   SetMarkerColor(gStyle->GetMarkerColor());
   SetMarkerStyle(gStyle->GetMarkerStyle());
   SetMarkerSize(gStyle->GetMarkerSize());

   gDirectory->Append(this);

   // if title starts with ":" and is a valid folder name, a superbranch
   // is created.
   gTree = this;
   if (strlen(title) > 2) {
      if (title[0] == '/') {
         Branch(title+1,32000,splitlevel);
      }
   }
}

//______________________________________________________________________________
TTree::~TTree()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================
   if (fDirectory) {
      if (!fDirectory->TestBit(TDirectory::kCloseDirectory)) {
         fDirectory->GetList()->Remove(this);
      }
   }
   fLeaves.Clear();
   fBranches.Delete();
   delete fPlayer;
   if (fFriends) {
      fFriends->Delete();
      delete fFriends;
      fFriends = 0;
   }
   fDirectory  = 0; //must be done after the destruction of friends
}

//______________________________________________________________________________
TFriendElement *TTree::AddFriend(const char *treename, const char *filename)
{
// Add a TFriendElement to the list of friends.
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

   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,treename,filename);
   if (fe) {
      fFriends->Add(fe);
      TTree *t = fe->GetTree();
      if (t) {
         if (t->GetEntries() < fEntries) {
            Warning("AddFriend","FriendElement %s in file %s has less entries %g than its parent Tree: %g",
                     treename,filename,t->GetEntries(),fEntries);
         }
      } else {
         Warning("AddFriend","Unknown Tree %s in file %s",treename,filename);
      }
   } else {
      Warning("AddFriend","Cannot add FriendElement %s in file %s",treename,filename);
   }
   return fe;
}

//______________________________________________________________________________
TFriendElement *TTree::AddFriend(const char *treename, TFile *file)
{
// Add a TFriendElement to the list of friends. The TFile is managed by
// the user (e.g. the user must delete the file).
// For complete description see AddFriend(const char *, const char *).

   if (!fFriends) fFriends = new TList();
   TFriendElement *fe = new TFriendElement(this,treename,file);
   if (fe) {
      fFriends->Add(fe);
      TTree *t = fe->GetTree();
      if (t) {
         if (t->GetEntries() < fEntries) {
            Warning("AddFriend","FriendElement %s in file %s has less entries %g than its parent tree: %g",
                     treename,file->GetName(),t->GetEntries(),fEntries);
         }
      } else {
         Warning("AddFriend","unknown tree %s in file %s",treename,file->GetName());
      }
   } else {
      Warning("AddFriend","cannot add FriendElement %s in file %s",treename,
              file?file->GetName():"");
   }
   return fe;
}

//______________________________________________________________________________
void TTree::AutoSave()
{
//*-*-*-*-*-*-*-*-*-*-*AutoSave tree header every fAutoSave bytes*-*-*-*-*-*
//*-*                  ==========================================
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
   if (!fDirectory) return;
//   printf("AutoSave Tree:%s after %g bytes written\n",GetName(),fTotBytes);
   fSavedBytes = fTotBytes;
   TDirectory *dirsav = gDirectory;
   fDirectory->cd();
   TKey *key = (TKey*)fDirectory->GetListOfKeys()->FindObject(GetName());
   Int_t wOK = Write(); //wOK will be 0 if Write failed (disk space exceeded)
   if (wOK && key) {
      key->Delete();
      delete key;
   }
   // save StreamerInfo
   TFile *file = fDirectory->GetFile();
   if (file) file->WriteStreamerInfo();
   dirsav->cd();
}

//______________________________________________________________________________
Int_t TTree::Branch(TList *list, Int_t bufsize, Int_t splitlevel)
{
//   Deprecated function. Use next function instead.
   return Branch((TCollection*)list,bufsize,splitlevel);
}

//______________________________________________________________________________
Int_t TTree::Branch(TCollection *list, Int_t bufsize, Int_t splitlevel, const char *name)
{
//   This function creates one branch for each element in the collection.
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
   if (list == 0) return 0;
   TObject *obj;
   Int_t nbranches = GetListOfBranches()->GetEntries();
   if (list->InheritsFrom(TClonesArray::Class())) {
         Error("Branch", "Cannot call this constructor for a TClonesArray");
         return 0;
   }

   Int_t nch = strlen(name);
   char branchname[kMaxLen];
   TIter next(list);

   while ((obj = next())) {
      if (splitlevel > 1 &&  obj->InheritsFrom(TCollection::Class())
                         && !obj->InheritsFrom(TClonesArray::Class())) {
         TCollection *col = (TCollection*)obj;
         if (nch) sprintf(branchname,"%s_%s_",name,col->GetName());
         else     sprintf(branchname,"%s_",col->GetName());
         Branch(col,bufsize,splitlevel-1,branchname);
      } else {
         if (nch && name[nch-1] == '_') sprintf(branchname,"%s%s",name,obj->GetName());
         else {
            if (nch)  sprintf(branchname,"%s_%s",name,obj->GetName());
            else      sprintf(branchname,"%s",obj->GetName());
         }
         if (splitlevel > 1) strcat(branchname,".");
         Bronch(branchname,obj->ClassName(),
                list->GetObjectRef(obj),bufsize,splitlevel-1);
      }
   }
   return GetListOfBranches()->GetEntries() - nbranches;
}

//______________________________________________________________________________
Int_t TTree::Branch(const char *foldername, Int_t bufsize, Int_t splitlevel)
{
//   This function creates one branch for each element in the folder.
//   The function returns the total number of branches created.

   TObject *ob = gROOT->FindObjectAny(foldername);
   if (!ob) return 0;
   if (ob->IsA() != TFolder::Class()) return 0;
   Int_t nbranches = GetListOfBranches()->GetEntries();
   TFolder *folder = (TFolder*)ob;
   TIter next(folder->GetListOfFolders());
   TObject *obj;
   char *curname = new char[1000];
   char occur[20];
   while ((obj=next())) {
      sprintf(curname,"%s/%s",foldername,obj->GetName());
      if (obj->IsA() == TFolder::Class()) {
        Branch(curname, bufsize, splitlevel-1);
      } else {
        void *add = (void*)folder->GetListOfFolders()->GetObjectRef(obj);
        for (Int_t i=0;i<1000;i++) {
           if (curname[i] == 0) break;
           if (curname[i] == '/') curname[i] = '.';
        }
        Int_t noccur = folder->Occurence(obj);
        if (noccur > 0) {
           sprintf(occur,"_%d",noccur);
           strcat(curname,occur);
        }
        TBranchElement *br;
        br = (TBranchElement*)Bronch(curname,obj->ClassName(), add, bufsize, splitlevel-1);
        br->SetBranchFolder();
      }
   }
   delete [] curname;
   return GetListOfBranches()->GetEntries() - nbranches;
}

//______________________________________________________________________________
TBranch *TTree::Branch(const char *name, void *address, const char *leaflist,Int_t bufsize)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new TTree Branch*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//
//     This Branch constructor is provided to support non-objects in
//     a Tree. The variables described in leaflist may be simple variables
//     or structures.
//    See the two following constructors for writing objects in a Tree.
//
//    By default the branch buffers are stored in the same file as the Tree.
//    use TBranch::SetFile to specify a different file

   gTree = this;
   TBranch *branch = new TBranch(name,address,leaflist,bufsize);
   if (branch->IsZombie()) {
      delete branch;
      return 0;
   }
   fBranches.Add(branch);
   return branch;
}

//______________________________________________________________________________
TBranch *TTree::Branch(const char *name, void *clonesaddress, Int_t bufsize, Int_t splitlevel)
{
// Special constructor for TClonesArray.
// Note that this function is only provided for backward compatibility.
// The Branch or Bronch methods can automatically detect the case of TClonesArray.
//
//
//    name:    global name of this BranchClones
//    bufsize: buffersize in bytes of each individual data member buffer
//    clonesaddress is the address of a pointer to a TClonesArray.
//
//    This Tree option is provided in case each entry consists of one
//    or more arrays of same class objects (tracks, hits,etc).
//    This function creates as many branches as there are public data members
//    in the objects pointed by the TClonesArray. Note that these data members
//    can be only basic data types, not pointers or objects.
//
//    BranchClones have the following advantages compared to the two other
//    solutions (Branch and BranchObject).
//      - When reading the Tree data, it is possible to read selectively
//        a subset of one object (may be just one single data member).
//      - This solution minimizes the number of objects created/destructed.
//      - Data members of the same type are consecutive in the basket buffers,
//        therefore optimizing the compression algorithm.
//      - Array processing notation becomes possible in the query language.
//
//    By default the branch buffers are stored in the same file as the Tree.
//    use TBranch::SetFile to specify a different file
//
//    By default the two members of TObject (fBits and fUniqueID) are stored
//    on individual branches. If the splitlevel > 1, these two branches
//    will not be created.

   if (clonesaddress == 0) return 0;

   char *cpointer =(char*)clonesaddress;
   char **ppointer =(char**)cpointer;
   TClonesArray *list = (TClonesArray*)(*ppointer);
   if (list == 0) return 0;
   gTree = this;
   if (fgBranchStyle == 1) {
      return Bronch(name,"TClonesArray",clonesaddress,bufsize,splitlevel);
   }
   if (splitlevel > 0) {
      TBranchClones *branch = new TBranchClones(name,clonesaddress,bufsize,-1,splitlevel);
      fBranches.Add(branch);
      return branch;
   } else {
      TBranchObject *branch = new TBranchObject(name,list->ClassName(),clonesaddress,bufsize,0);
      fBranches.Add(branch);
      return branch;
   }
}

//______________________________________________________________________________
TBranch *TTree::Branch(const char *name, const char *classname, void *addobj, Int_t bufsize, Int_t splitlevel)
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

   if (fgBranchStyle == 1) {
      return Bronch(name,classname,addobj,bufsize,splitlevel);
   } else {
      if (splitlevel < 0) splitlevel = 0;
      return BranchOld(name,classname,addobj,bufsize,splitlevel);
   }
}
//______________________________________________________________________________
TBranch *TTree::BranchOld(const char *name, const char *classname, void *addobj, Int_t bufsize, Int_t splitlevel)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new TTree BranchObject*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
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

   gTree = this;
   TClass *cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("BranchObject","Cannot find class:%s",classname);
      return 0;
   }
   TBranch *branch = new TBranchObject(name,classname,addobj,bufsize,splitlevel);
   fBranches.Add(branch);
   if (!splitlevel) return branch;

   TObjArray *blist = branch->GetListOfBranches();
   const char *rdname;
   const char *dname;
   char branchname[128];
   char **apointer = (char**)(addobj);
   TObject *obj = (TObject*)(*apointer);
   Bool_t delobj = kFALSE;
   if (!obj) {
      obj = (TObject*)cl->New();
      delobj = kTRUE;
   }
   //build the StreamerInfo if first time for the class
   BuildStreamerInfo(cl,obj);

//*-*- Loop on all public data members of the class and its base classes
   Int_t lenName = strlen(name);
   Int_t isDot = 0;
   if (name[lenName-1] == '.') isDot = 1;
   TBranch *branch1 = 0;
   TRealData *rd, *rdi;
   TIter      nexti(cl->GetListOfRealData());
   TIter      next(cl->GetListOfRealData());
   while ((rd = (TRealData *) next())) {
      TDataMember *dm = rd->GetDataMember();
      if (!dm->IsPersistent()) continue; //do not process members with a ! as the first
                                         // character in the comment field
      if (rd->IsObject()) {
         TClass *clm = gROOT->GetClass(dm->GetFullTypeName());
         if (clm) BuildStreamerInfo(clm,(char*)obj+rd->GetThisOffset());
         continue;
      }
      rdname = rd->GetName();
      dname  = dm->GetName();
      if (cl->CanIgnoreTObjectStreamer()) {
         if (strcmp(dname,"fBits") == 0) continue;
         if (strcmp(dname,"fUniqueID") == 0) continue;
      }

      TDataType *dtype = dm->GetDataType();
      Int_t code = 0;
      if (dtype) code = dm->GetDataType()->GetType();

//*-*- Encode branch name. Use real data member name
      sprintf(branchname,"%s",rdname);
      if (isDot) {
         if (dm->IsaPointer()) sprintf(branchname,"%s%s",name,&rdname[1]);
         else                  sprintf(branchname,"%s%s",name,&rdname[0]);
      }
      char leaflist[128];
      Int_t offset    = rd->GetThisOffset();
      char *pointer   = (char*)obj + offset;
      if (dm->IsaPointer()) {
         TClass *clobj = 0;
         if (!dm->IsBasic()) clobj = gROOT->GetClass(dm->GetTypeName());
         if (clobj && clobj->InheritsFrom("TClonesArray")) {
            char *cpointer  =(char*)pointer;
            char **ppointer =(char**)cpointer;
            TClonesArray *list = (TClonesArray*)(*ppointer);
            if (splitlevel != 2) {
               if (isDot) branch1 = new TBranchClones(&branchname[0],pointer,bufsize);
               else       branch1 = new TBranchClones(&branchname[1],pointer,bufsize);
               blist->Add(branch1);
            } else {
               if (isDot) branch1 = new TBranchObject(&branchname[0],list->ClassName(),pointer,bufsize);
               else       branch1 = new TBranchObject(&branchname[1],list->ClassName(),pointer,bufsize);
               blist->Add(branch1);
            }
         } else {
            if (!clobj) {
               const char * index = dm->GetArrayIndex();
               if (strlen(index)!=0) {
                  //check that index is a valid data member name
                  //if member is part of an object (eg fA and index=fN)
                  //index must be changed from fN to fA.fN
                  char aindex[128];
                  strcpy(aindex,rd->GetName());
                  char *rdot = strrchr(aindex,'.');
                  if (rdot) strcpy(rdot+1,index);
                  nexti.Reset();
                  while ((rdi = (TRealData *) nexti())) {
                     if (strcmp(rdi->GetName(),index) == 0) break;
                     if (strcmp(rdi->GetName(),aindex) == 0) {index = rdi->GetName(); break;}
                  }
                  if      (code ==  1)
                     // Note that we differentiate between strings and
                     // char array by the fact that there is NO specified
                     // size for a string (see next if (code == 1)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"B");
                  else if (code == 11)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"b");
                  else if (code ==  2)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"S");
                  else if (code == 12)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"s");
                  else if (code ==  3)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"I");
                  else if (code == 13)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"i");
                  else if (code ==  5)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"F");
                  else if (code ==  8)
                     sprintf(leaflist,"%s[%s]/%s",&rdname[0],index,"D");
                  else {
                     printf("Cannot create branch for rdname=%s, code=%d\n",branchname, code);
                     leaflist[0] = 0;
                  }
               } else {
                  if (code == 1) {
                     sprintf(leaflist,"%s/%s",dname,"C");
                  } else {
                     continue;
                  }
               }

               // there are '*' in both the branchname and rdname
               char bname[128];
               UInt_t cursor,pos;
               for(cursor = 0, pos = 0;
                   cursor< strlen(branchname);
                   cursor++ ) {
                  if (branchname[cursor]!='*') {
                     bname[pos++]=branchname[cursor];
                  }
               }
               bname[pos] = '\0';
               for(cursor = 0, pos = 0; cursor< strlen(leaflist); cursor++ ) {
                  if (leaflist[cursor]!='*') {
                     leaflist[pos++]=leaflist[cursor];
                  }
               }
               leaflist[pos] = '\0';

               // Add the branch to the tree and indicate that the address
               // is that of a pointer to be dereferenced before using.
               branch1 = new TBranch(bname,*(void**)pointer,leaflist,bufsize);
               TLeaf *leaf = (TLeaf*) branch1->GetListOfLeaves()->At(0);
               leaf->SetBit(TLeaf::kIndirectAddress);
               leaf->SetAddress((void**)pointer);
               blist->Add(branch1);
            } else {
               if (!clobj->InheritsFrom(TObject::Class())) continue;
               branch1 = new TBranchObject(dname,clobj->GetName(),pointer,bufsize,0);
               if (isDot) branch1->SetName(&branchname[0]);
               else       branch1->SetName(&branchname[1]);  //do not use the first character (*)
               blist->Add(branch1);
            }
         }
      } else {
//*-*-------------Data Member is a basic data type----------
         if (dm->IsBasic()) {
            if      (code ==  1) sprintf(leaflist,"%s/%s",rdname,"B");
            else if (code == 11) sprintf(leaflist,"%s/%s",rdname,"b");
            else if (code ==  2) sprintf(leaflist,"%s/%s",rdname,"S");
            else if (code == 12) sprintf(leaflist,"%s/%s",rdname,"s");
            else if (code ==  3) sprintf(leaflist,"%s/%s",rdname,"I");
            else if (code == 13) sprintf(leaflist,"%s/%s",rdname,"i");
            else if (code ==  5) sprintf(leaflist,"%s/%s",rdname,"F");
            else if (code ==  8) sprintf(leaflist,"%s/%s",rdname,"D");
            else {
               printf("Cannot create branch for rdname=%s, code=%d\n",branchname, code);
               leaflist[0] = 0;
            }
            branch1 = new TBranch(branchname,pointer,leaflist,bufsize);
            branch1->SetTitle(rdname);
            blist->Add(branch1);
         }
      }
      if (branch1) branch1->SetOffset(offset);
      else Warning("Branch","Cannot process member:%s",rdname);

   }
   if (delobj) delete obj;
   return branch;
}

//______________________________________________________________________________
TBranch *TTree::Bronch(const char *name, const char *classname, void *add, Int_t bufsize, Int_t splitlevel)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new TTree BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================
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
//   Use splitlevel < 0 instead of splitlevel=0 when the class
//   has a custom Streamer

   gTree = this;
   TClass *cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("Bronch","Cannot find class:%s",classname);
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

   char **ppointer = (char**)add;
   char *objadd = *ppointer;
   if (cl == TClonesArray::Class()) {
      TClonesArray *clones = (TClonesArray *)objadd;
      if (!clones) {
         Error("Bronch","Pointer to TClonesArray is null");
         return 0;
      }
      if (!clones->GetClass()) {
         Error("Bronch","TClonesArray with no class defined in branch: %s",name);
         return 0;
      }
      if (splitlevel > 0) {
         if (clones->GetClass()->GetClassInfo()->RootFlag() & 1)
            Warning("Bronch","Using split mode on a class: %s with a custom Streamer",clones->GetClass()->GetName());
      } else {
         if (clones->GetClass()->GetClassInfo()->RootFlag() & 1) clones->BypassStreamer(kFALSE);
         TBranchObject *branch = new TBranchObject(name,classname,add,bufsize,0);
         fBranches.Add(branch);
         return branch;
      }
   }

   Bool_t hasCustomStreamer = kFALSE;
   if (cl->GetClassInfo()->RootFlag() & 1)  hasCustomStreamer = kTRUE;
   if (splitlevel < 0 || (splitlevel == 0 && hasCustomStreamer)) {
      TBranchObject *branch = new TBranchObject(name,classname,add,bufsize,0);
      fBranches.Add(branch);
      return branch;
   }

   //hopefully normal case
   Bool_t delobj = kFALSE;
   //====> special case of TClonesArray
   if(cl == TClonesArray::Class()) {
      TBranchElement *branch = new TBranchElement(name,(TClonesArray*)objadd,bufsize,splitlevel);
      fBranches.Add(branch);
      branch->SetAddress(add);
      return branch;
   }
   //====>

   if (!objadd) {
      objadd = (char*)cl->New();
      *ppointer = objadd;
      delobj = kTRUE;
   }
   //build the StreamerInfo if first time for the class
   Bool_t optim = TStreamerInfo::CanOptimize();
   if (splitlevel > 0) TStreamerInfo::Optimize(kFALSE);
   TStreamerInfo *sinfo = BuildStreamerInfo(cl,objadd);
   TStreamerInfo::Optimize(optim);

   // create a dummy top level  branch object
   Int_t id = -1;
   if (splitlevel > 0) id = -2;
   char *dot = (char*)strchr(name,'.');
   Int_t nch = strlen(name);
   Bool_t dotlast = kFALSE;
   if (nch && name[nch-1] == '.') dotlast = kTRUE;
   TBranchElement *branch = new TBranchElement(name,sinfo,id,objadd,bufsize,splitlevel);
   fBranches.Add(branch);
   if (splitlevel > 0) {

      // Loop on all public data members of the class and its base classes
      TObjArray *blist = branch->GetListOfBranches();
      TIter next(sinfo->GetElements());
      TStreamerElement *element;
      id = 0;
      char *bname = new char[1000];
      while ((element = (TStreamerElement*)next())) {
         char *pointer = (char*)objadd + element->GetOffset();
         Bool_t isBase = element->IsA() == TStreamerBase::Class();
         if (isBase) {
            TClass *clbase = element->GetClassPointer();
            if (clbase == TObject::Class() && cl->CanIgnoreTObjectStreamer()) continue;
         }
         if (dot) {
            if (dotlast) {
               sprintf(bname,"%s%s",name,element->GetFullName());
            } else {
               if (isBase) sprintf(bname,"%s",name);
               else        sprintf(bname,"%s.%s",name,element->GetFullName());
            }
         } else {
            sprintf(bname,"%s",element->GetFullName());
         }
         TBranchElement *bre = new TBranchElement(bname,sinfo,id,pointer,bufsize,splitlevel-1);
         blist->Add(bre);
         id++;
      }
      delete [] bname;
   }

   branch->SetAddress(add);

   if (delobj) {delete objadd; *ppointer=0;}
   return branch;
}


//______________________________________________________________________________
void TTree::Browse(TBrowser *b)
{
   fBranches.Browse(b);
}

//______________________________________________________________________________
void TTree::BuildIndex(const char *majorname, const char *minorname)
{
   // Build an index table using the leaves with name: major & minor name
   // The index is built in the following way:
   //    A pass on all entries is made using TTree::Draw
   //    var1 = majorname
   //    var2 = minorname
   //    sel  = majorname +minorname*1e-9
   //    The standard result from TTree::Draw is stored in fV1, fV2 and fW
   //    The array fW is sorted into fIndex
   //  Once the index is computed, one can retrieve one entry via
   //    TTree:GetEntryWithIndex(majornumber, minornumber)
   // Example:
   //  tree.BuildIndex("Run","Event"); //creates an index using leaves Run and Event
   //  tree.GetEntryWithIndex(1234,56789); //reads entry corresponding to
   //                                        Run=1234 and Event=56789
   //
   // Note that majorname and minorname may be expressions using original
   // Tree variables eg: "run-90000", "event +3*xx"
   // In case an expression is specified, the equivalent expression must be computed
   // when calling GetEntryWithIndex.
   //
   // Note that once the index is built, it can be saved with the TTree object
   // with tree.Write(); //if the file has been open in "update" mode.
   //
   // The most convenient place to create the index is at the end of
   // the filling process just before saving the Tree header.
   // If a previous index was computed, it is redefined by this new call.
   //
   // Note that this function can also be applied to a TChain.

   Int_t nch = strlen(majorname) + strlen(minorname) + 10;
   char *varexp = new char[nch];
   sprintf(varexp,"%s+%s*1e-9",majorname,minorname);

   Int_t oldEstimate = fEstimate;
   Int_t n = (Int_t)GetEntries(); //must use GetEntries instead of fEntries in case of a chain
   if (n <= 0) return;

   if (n > fEstimate) SetEstimate(n);

   Draw(varexp,"","goff");

   // Sort array fV1 (contains  majorname +minorname*1e-9) into fIndex
   Double_t *w = GetPlayer()->GetV1();
   Int_t *ind = new Int_t[n];
   TMath::Sort(n,w,ind,0);
   fIndexValues.Set(n);
   fIndex.Set(n);
   for (Int_t i=0;i<n;i++) {
      fIndexValues.fArray[i] = w[ind[i]];
      fIndex.fArray[i] = ind[i];
   }
   if (n > oldEstimate) SetEstimate(oldEstimate);

   // clean up
   delete [] ind;
   delete [] varexp;
}

//______________________________________________________________________________
TStreamerInfo *TTree::BuildStreamerInfo(TClass *cl, void *pointer)
{
  // Build StreamerInfo for class cl
  // pointer is an optional argument that may contain a pointer to an object of cl

   if (!cl) return 0;
   cl->BuildRealData(pointer);
   TStreamerInfo *sinfo = cl->GetStreamerInfo(cl->GetClassVersion());
   if(fDirectory) sinfo->ForceWriteInfo(fDirectory->GetFile());

   // Create StreamerInfo for all base classes
   TBaseClass *base;
   TIter nextb(cl->GetListOfBases());
   while((base = (TBaseClass*)nextb())) {
      TClass *clm = gROOT->GetClass(base->GetName());
      BuildStreamerInfo(clm);
   }
   return sinfo;
}

//______________________________________________________________________________
TTree *TTree::CloneTree(Int_t nentries, Option_t *option)
{
// Create a clone of this tree and copy nentries
// By default copy all entries
// option is reserved for future use
// Note that only active branches are copied.
//
// IMPORTANT: Before invoking this function, the branch addresses
//            of this TTree must have been set if one or more branches
//            is not a basic type.
// For examples of CloneTree, see tutorials
//  -copytree:
//    Example of Root macro to copy a subset of a Tree to a new Tree
//    The input file has been generated by the program in $ROOTSYS/test/Event
//    with   Event 1000 1 1 1
//  -copytree2:
//    Example of Root macro to copy a subset of a Tree to a new Tree
//    One branch of the new Tree is written to a separate file
//    The input file has been generated by the program in $ROOTSYS/test/Event
//    with   Event 1000 1 1 1

  // we make a full copy of this tree
   TTree *tree = (TTree*)Clone();
   if (tree == 0) return 0;

   tree->Reset();

  // delete non active branches from the clone
   Int_t i,j,k,l,nb1,nb2;
   TObjArray *lb, *lb1;
   TBranch *branch, *b1, *b2;
   TObjArray *leaves = tree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      if (!leaf) continue;
      branch = leaf->GetBranch();
      if (!branch || !branch->TestBit(kDoNotProcess)) continue;
      TObjArray *branches = tree->GetListOfBranches();
      Int_t nb = branches->GetEntriesFast();
      for (i=0;i<nb;i++) {
         TBranch *br = (TBranch*)branches->UncheckedAt(i);
         if (br == branch) {branches->RemoveAt(i); delete br; branches->Compress(); break;}
         lb = br->GetListOfBranches();
         nb1 = lb->GetEntriesFast();
         for (j=0;j<nb1;j++) {
            b1 = (TBranch*)lb->UncheckedAt(j);
            if (!b1) continue;
            if (b1 == branch) {lb->RemoveAt(j); delete b1; lb->Compress(); break;}
            lb1 = b1->GetListOfBranches();
            nb2 = lb1->GetEntriesFast();
            for (k=0;k<nb2;k++) {
               b2 = (TBranch*)lb1->UncheckedAt(k);
               if (!b2) continue;
               if (b2 == branch) {lb1->RemoveAt(k); delete b2; lb1->Compress(); break;}
            }
         }
      }
   }
   leaves->Compress();

  // copy branch addresses
   CopyAddresses(tree);

  // may be copy some entries
   if (nentries < 0) nentries = Int_t(fEntries);
   if (nentries > fEntries) nentries = Int_t(fEntries);
   for (i=0;i<nentries;i++) {
      GetEntry(i);
      tree->Fill();
   }
   return tree;
}

//______________________________________________________________________________
void TTree::CopyAddresses(TTree *tree)
{
// Set branch addresses of tree equal to the ones of this tree

  // copy branch addresses starting from branches
   Int_t i;
   TObjArray *branches  = GetListOfBranches();
   Int_t nbranches = branches->GetEntriesFast();
   for (i=0;i<nbranches;i++) {
      TBranch *branch = (TBranch*)branches->UncheckedAt(i);
      if (branch->TestBit(kDoNotProcess)) continue;
      if (branch->GetAddress()) {
         TBranch *br = tree->GetBranch(branch->GetName());
         if (br) br->SetAddress(branch->GetAddress());
      }
   }
  // copy branch addresses
   TObjArray *tleaves = tree->GetListOfLeaves();
   Int_t nleaves = tleaves->GetEntriesFast();
   for (i=0;i<nleaves;i++) {
      TLeaf *leaf2 = (TLeaf*)tleaves->UncheckedAt(i);
      TLeaf *leaf  = GetLeaf(leaf2->GetName());
      if (!leaf) continue;
      TBranch *branch  = leaf->GetBranch();
      if (!branch) continue;
      if (branch->TestBit(kDoNotProcess)) continue;
      if (branch->GetAddress()) {
         tree->SetBranchAddress(branch->GetName(),branch->GetAddress());
      } else {
         leaf2->SetAddress(leaf->GetValuePointer());
      }
   }
}

//______________________________________________________________________________
Int_t TTree::CopyEntries(TTree *tree, Int_t nentries)
{
// Copy nentries from tree to this tree
// By default copy all entries
// Return number of bytes copied to this tree.

   if (tree == 0) return 0;

   Int_t nbytes = 0;
   Int_t treeEntries = Int_t(tree->GetEntries());
   if (nentries < 0) nentries = treeEntries;
   if (nentries > treeEntries) nentries = treeEntries;
   for (Int_t i=0;i<nentries;i++) {
      tree->GetEntry(i);
      nbytes += Fill();
   }
   return nbytes;
}

//______________________________________________________________________________
TTree *TTree::CopyTree(const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*copy a Tree with selection*-*-*-*-*-*
//*-*              ==========================
//
// IMPORTANT: Before invoking this function, the branch addresses
//            of this TTree must have been set if one or more branches
//            is not a basic type.
// For examples of CloneTree, see tutorials
//  -copytree:
//    Example of Root macro to copy a subset of a Tree to a new Tree
//    The input file has been generated by the program in $ROOTSYS/test/Event
//    with   Event 1000 1 1 1
//  -copytree2:
//    Example of Root macro to copy a subset of a Tree to a new Tree
//    One branch of the new Tree is written to a separate file
//    The input file has been generated by the program in $ROOTSYS/test/Event
//    with   Event 1000 1 1 1
//
// NOTE that only the active branches are copied.

   GetPlayer();
   if (fPlayer) return fPlayer->CopyTree(selection,option,nentries,firstentry);
   return 0;
}

//______________________________________________________________________________
void TTree::Delete(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Delete this tree from memory or/and disk
//*-*              ========================================
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
            Int_t pos = branch->GetBasketSeek(i);
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
Int_t TTree::Draw(const char *varexp, TCut selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*-*-*Draw expression varexp for specified entries-*-*-*-*-*
//*-*                  ===========================================
//
//      This function accepts TCut objects as arguments.
//      Useful to use the string operator +
//         example:
//            ntuple.Draw("x",cut1+cut2+cut3);
//

   return TTree::Draw(varexp, selection.GetTitle(), option, nentries, firstentry);
}

//______________________________________________________________________________
Int_t TTree::Draw(const char *varexp, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*-*-*Draw expression varexp for specified entries-*-*-*-*-*
//*-*                  ===========================================
//
//  varexp is an expression of the general form e1:e2:e3
//    where e1,etc is a formula referencing a combination of the columns
//  Example:
//     varexp = x     simplest case: draw a 1-Dim distribution of column named x
//            = sqrt(x)            : draw distribution of sqrt(x)
//            = x*y/z
//            = y:sqrt(x) 2-Dim dsitribution of y versus sqrt(x)
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
//  option is the drawing option
//      see TH1::Draw for the list of all drawing options.
//      If option contains the string "goff", no graphics is generated.
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
//     Saving the result of Draw to an histogram
//     =========================================
//  By default the temporary histogram created is called htemp.
//  One can retrieve a pointer to this histogram with:
//    TH1F *htemp = (TH1F*)gPad->GetPrimitive("htemp");
//
//  If varexp0 contains >>hnew (following the variable(s) name(s),
//  the new histogram created is called hnew and it is kept in the current
//  directory (and also the current pad).
//  Example:
//    tree.Draw("sqrt(x)>>hsqrt","y>0")
//    will draw sqrt(x) and save the histogram as "hsqrt" in the current
//    directory.
//      TH1F *hnew = (TH1F*)gDirectory->Get("hnew");
//
//  By default, the specified histogram is reset.
//  To continue to append data to an existing histogram, use "+" in front
//  of the histogram name;
//    tree.Draw("sqrt(x)>>+hsqrt","y>0")
//      will not reset hsqrt, but will continue filling.
//
//     Special functions and variables
//     ===============================
//
//  'ENTRY':  A TTree::Draw formula can use the special variable ENTRY
//  to access the entry number being read.  For example to draw every 
//  other entry use:
//    tree.Draw("myvar","ENTRY%2==0");
//
//     Making a Profile histogram
//     ==========================
//  In case of a 2-Dim expression, one can generate a TProfile histogram
//  instead of a TH2F histogram by specyfying option=prof or option=profs.
//  The option=prof is automatically selected in case of y:x>>pf
//  where pf is an existing TProfile histogram.
//
//     Saving the result of Draw to a TEventList
//     =========================================
//  TTree::Draw can be used to fill a TEventList object (list of entry numbers)
//  instead of histogramming one variable.
//  If varexp0 has the form >>elist , a TEventList object named "elist"
//  is created in the current directory. elist will contain the list
//  of entry numbers satisfying the current selection.
//  Example:
//    tree.Draw(">>yplus","y>0")
//    will create a TEventList object named "yplus" in the current directory.
//    In an interactive session, one can type (after TTree::Draw)
//       yplus.Print("all")
//    to print the list of entry numbers in the list.
//
//  By default, the specified entry list is reset.
//  To continue to append data to an existing list, use "+" in front
//  of the list name;
//    tree.Draw(">>+yplus","y>0")
//      will not reset yplus, but will enter the selected entries at the end
//      of the existing list.
//
//      Using a TEventList as Input
//      ===========================
//  Once a TEventList object has been generated, it can be used as input
//  for TTree::Draw. Use TTree::SetEventList to set the current event list
//  Example:
//     TEventList *elist = (TEventList*)gDirectory->Get("yplus");
//     tree->SetEventList(elist);
//     tree->Draw("py");
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
//  use TEventList::SetReapplyCut.
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
   if (fPlayer) return fPlayer->DrawSelect(varexp,selection,option,nentries,firstentry);
   else return -1;
}


//______________________________________________________________________________
void TTree::DropBuffers(Int_t)
{
//*-*-*-*-*Drop branch buffers to accomodate nbytes below MaxVirtualsize*-*-*-*

// Be careful not to remove current read/write buffers
   Int_t i,j;
   Int_t ndrop = 0;
   Int_t nleaves = fLeaves.GetEntriesFast();
   TLeaf *leaf;
   TBranch *branch;
   TBasket *basket;
   for (i=0;i<nleaves;i++)  {
      leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      Int_t nbaskets = branch->GetListOfBaskets()->GetEntriesFast();
      for (j=0;j<nbaskets-1;j++)  {
         if (j == branch->GetReadBasket() || j == branch->GetWriteBasket()) continue;
         basket = branch->GetBasket(j);
         ndrop += basket->DropBuffers();
         if (fTotalBuffers  < fMaxVirtualSize) return;
      }
   }
}

//______________________________________________________________________________
Int_t TTree::Fill()
{
//*-*-*-*-*Fill all branches of a Tree*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ===========================
//
//   This function loops on all the branches of this tree.
//   For each branch, it copies to the branch buffer (basket) the current
//   values of the leaves data types.
//   If a leaf is a simple data type, a simple conversion to a machine
//   independent format has to be done.
//
   Int_t i;
   Int_t nbytes = 0;
   Int_t nb = fBranches.GetEntriesFast();
   TBranch *branch;

    //case of one single super branch. Automatically update
    // all the branch addresses if a new object was created
   if (nb == 1) {
      branch = (TBranch*)fBranches.UncheckedAt(0);
      branch->UpdateAddress();
   }

   for (i=0;i<nb;i++)  {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      if (branch->TestBit(kDoNotProcess)) continue;
      nbytes += branch->Fill();
   }
   fEntries++;
   if (fTotBytes-fSavedBytes > fAutoSave) AutoSave();
   return nbytes;
}

//______________________________________________________________________________
TBranch *TTree::FindBranch(const char* branchname)
{
   char name[kMaxLen];
   TIter next(GetListOfBranches());

   // This will allow the branchname to be preceded by
   // the name of this tree.
   char *subbranch = (char*)strstr(branchname,GetName());
   if (subbranch!=branchname) subbranch = 0;
   if (subbranch) {
     subbranch += strlen(GetName());
     if ( *subbranch != '.' ) subbranch = 0;
     else subbranch ++;
   }
   TBranch *branch;
   while ((branch = (TBranch*)next())) {
      strcpy(name,branch->GetName());
      char *dim = (char*)strstr(name,"[");
      if (dim) dim[0]='\0';
      if (!strcmp(branchname,name)) return branch;
      if (subbranch && !strcmp(subbranch,name)) return branch;
      TBranch *nestedbranch = branch->FindBranch(branchname);
      if (nestedbranch) return nestedbranch;
   }

   //search in list of friends
   if (!fFriends) return 0;
   TIter nextf(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)nextf())) {
      TTree *t = fe->GetTree();
      // If the alias is present replace it with the real name.
      char *subbranch = (char*)strstr(branchname,fe->GetName());
      if (subbranch!=branchname) subbranch = 0;
      if (subbranch) {
         subbranch += strlen(fe->GetName());
         if ( *subbranch != '.' ) subbranch = 0;
         else subbranch ++;
      }
      if (subbranch) {
         sprintf(name,"%s.%s",t->GetName(),subbranch);
      } else {
         strcpy(name,branchname);
      }
      branch = t->FindBranch(name);
      if (branch) return branch;
   }
   return 0;
}


//______________________________________________________________________________
TLeaf *TTree::FindLeaf(const char* searchname)
{
   char leafname[kMaxLen];
   char leaftitle[kMaxLen];
   char longname[kMaxLen];
   char longtitle[kMaxLen];

   // This will allow the branchname to be preceded by
   // the name of this tree.
   char *subsearchname = (char*)strstr(searchname,GetName());
   if (subsearchname!=searchname) subsearchname = 0;
   if (subsearchname) {
     subsearchname += strlen(GetName());
     if ( *subsearchname != '.' ) subsearchname = 0;
     else subsearchname ++;
   }

   // For leaves we allow for one level up to be prefixed to the
   // name

   TIter next (GetListOfLeaves());
   TLeaf *leaf;
   while ((leaf = (TLeaf*)next())) {
      strcpy(leafname,leaf->GetName());
      char *dim = (char*)strstr(leafname,"[");
      if (dim) dim[0]='\0';

      if (!strcmp(searchname,leafname)) return leaf;
      if (subsearchname && !strcmp(subsearchname,leafname)) return leaf;

      // The TLeafElement contains the branch name in its name,
      // let's use the title....
      strcpy(leaftitle,leaf->GetTitle());
      dim = (char*)strstr(leaftitle,"[");
      if (dim) dim[0]='\0';

      if (!strcmp(searchname,leaftitle)) return leaf;
      if (subsearchname && !strcmp(subsearchname,leaftitle)) return leaf;

      TBranch * branch = leaf->GetBranch();
      if (branch) {
         sprintf(longname,"%s.%s",branch->GetName(),leafname);
         char *dim = (char*)strstr(longname,"[");
         if (dim) dim[0]='\0';
         if (!strcmp(searchname,longname)) return leaf;
         if (subsearchname && !strcmp(subsearchname,longname)) return leaf;

         sprintf(longtitle,"%s.%s",branch->GetName(),leaftitle);
         dim = (char*)strstr(longtitle,"[");
         if (dim) dim[0]='\0';
         if (!strcmp(searchname,longtitle)) return leaf;
         if (subsearchname && !strcmp(subsearchname,longtitle)) return leaf;

         // The following is for the case where the branch is only
         // a sub-branch.  Since we do not see it through
         // TTree::GetListOfBranches, we need to see it indirectly.
         // This is the less sturdy part of this search ... it may
         // need refining ...
         if (strstr(searchname,".")
             && !strcmp(searchname,branch->GetName())) return leaf;
         if (subsearchname && strstr(subsearchname,".")
             && !strcmp(subsearchname,branch->GetName())) return leaf;

         //printf("found leaf3=%s/%s, branch=%s, i=%d\n",leaf->GetName(),leaf->GetTitle(),branch->GetName(),i);
      }
   }

   //search in list of friends
   if (!fFriends) return 0;
   TIter nextf(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)nextf())) {
      TTree *t = fe->GetTree();
      // If the alias is present replace it with the real name.
      char *subsearchname = (char*)strstr(searchname,fe->GetName());
      if (subsearchname!=searchname) subsearchname = 0;
      if (subsearchname) {
         subsearchname += strlen(fe->GetName());
         if ( *subsearchname != '.' ) subsearchname = 0;
         else subsearchname ++;
      }
      if (subsearchname) {
         sprintf(leafname,"%s.%s",t->GetName(),subsearchname);
      } else {
         strcpy(leafname,searchname);
      }
      leaf = t->FindLeaf(leafname);
      if (leaf) return leaf;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TTree::Fit(const char *funcname ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Fit  a projected item(s) from a Tree*-*-*-*-*-*-*-*-*-*
//*-*              ======================================
//
//  funcname is a TF1 function.
//
//  See TTree::Draw for explanations of the other parameters.
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
   if (fPlayer) return fPlayer->Fit(funcname,varexp,selection,option,goption,nentries,firstentry);
   else         return -1;
}

//______________________________________________________________________________
TBranch *TTree::GetBranch(const char *name)
{
// Return pointer to the branch with name in this Tree or the list
// of friends of this tree.

   Int_t i,j,k,nb1,nb2;
   TObjArray *lb, *lb1;
   TBranch *branch, *b1, *b2;
   Int_t nb = fBranches.GetEntriesFast();
   for (i=0;i<nb;i++) {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      if (!strcmp(branch->GetName(),name)) return branch;
      lb = branch->GetListOfBranches();
      nb1 = lb->GetEntriesFast();
      for (j=0;j<nb1;j++) {
         b1 = (TBranch*)lb->UncheckedAt(j);
         if (!b1) continue;
         if (!strcmp(b1->GetName(),name)) return b1;
         lb1 = b1->GetListOfBranches();
         nb2 = lb1->GetEntriesFast();
         for (k=0;k<nb2;k++) {
            b2 = (TBranch*)lb1->UncheckedAt(k);
            if (!b2) continue;
            if (!strcmp(b2->GetName(),name)) return b2;
         }
      }
   }
   TObjArray *leaves = ((TTree*)this)->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (i=0;i<nleaves;i++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(i);
      branch = leaf->GetBranch();
      if (!strcmp(branch->GetName(),name)) return branch;
   }

   //search in list of friends
   if (!fFriends) return 0;
   TIter next(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      branch = t->GetBranch(name);
      if (branch) return branch;
   }

   //second pass in the list of friends when the branch name
   //is prefixed by the tree name
   next.Reset();
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      char *subname = (char*)strstr(name,fe->GetName());
      if (subname != name) continue;
      Int_t l = strlen(fe->GetName());
      subname += l;
      if (*subname != '.') continue;
      subname++;
      branch = t->GetBranch(subname);
      if (branch) return branch;
   }
   return 0;
}


//______________________________________________________________________________
Int_t TTree::GetBranchStyle()
{
  // static function returning the current branch style
  // style = 0 old Branch
  // style = 1 new Bronch

   return fgBranchStyle;
}

//______________________________________________________________________________
TFile *TTree::GetCurrentFile() const
{
//*-*-*-*-*-*Return pointer to the current file*-*-*-*-*-*-*-*
//*-*        ==================================

   if (!fDirectory) return 0;
   return fDirectory->GetFile();
}

//______________________________________________________________________________
Stat_t TTree::GetEntriesFriend() const
{
// return number of entries of this Tree if not zero
// otherwise return the number of entries in the first friend Tree.

   if (fEntries) return fEntries;
   if (!fFriends) return 0;
   TFriendElement *fr = (TFriendElement*)fFriends->At(0);
   if (!fr) return 0;
   return fr->GetTree()->GetEntriesFriend();
}

//______________________________________________________________________________
Int_t TTree::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*-*Read all branches of entry and return total number of bytes*-*-*
//*-*        ===========================================================
//     getall = 0 : get only active branches
//     getall = 1 : get all branches
//
//  The function returns the number of bytes read from the input buffer.
//  If entry does not exist or an I/O error occurs, the function returns 0.
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
//    for (Int_t i=0;i<nentries;i++) {
//       T.GetEntry(i);
//       // the objrect event has been filled at this point
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
//    for (Int_t i=0;i<nentries;i++) {
//       T.GetEntry(i);
//       // the objrect event has been filled at this point
//    }
//   In this case, at each iteration, the object event is deleted by GetEntry
//   and a new instance of Event is created and filled.
//
//   OPTION 3
//   --------
//   Same as option 1, but you delete yourself the event.
//    for (Int_t i=0;i<nentries;i++) {
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

   if (entry < 0 || entry >= fEntries) return 0;
   Int_t i;
   Int_t nbytes = 0;
   fReadEntry = entry;
   TBranch *branch;

   Int_t nb = fBranches.GetEntriesFast();
   for (i=0;i<nb;i++)  {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      nbytes += branch->GetEntry(entry, getall);
   }

   // GetEntry in list of friends
   if (!fFriends) return nbytes;
   TIter nextf(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)nextf())) {
      TTree *t = fe->GetTree();
        if (t) nbytes+=t->GetEntry(entry, getall);
   }
   return nbytes;
}


//______________________________________________________________________________
Int_t TTree::GetEntryNumber(Int_t entry) const
{
//*-*-*-*-*-*Return entry number corresponding to entry*-*-*
//*-*        ==========================================
//     if no selection list returns entry
//     else returns the entry number corresponding to the list index=entry

   if (!fEventList) return entry;
   return fEventList->GetEntry(entry);
}


//______________________________________________________________________________
Int_t TTree::GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const
{
// Return entry number corresponding to major and minor number
// Note that this function returns only the entry number, not the data
// To read the data corresponding to an entry number, use TTree::GetEntryWithIndex
// the BuildIndex function has created a table of Double_t* of sorted values
// corresponding to val = major + minor*1e-9;
// The function performs binary search in this sorted table.
// If it finds a pair that maches val, it returns directly the
// index in the table.
// If an entry corresponding to major and minor is not found, the function
// returns the index of the major,minor pair immediatly lower than the
// requested value, ie it will return -1 if the pair is lower than
// the first entry in the index.
//
// See also GetEntryNumberWithIndex

   if (fIndex.fN == 0) return -1;
   Double_t value = major + minor*1e-9;
   Int_t i = TMath::BinarySearch(Int_t(fEntries), fIndexValues.fArray, value);
   if (i < 0) return -1;
   return fIndex.fArray[i];
}


//______________________________________________________________________________
Int_t TTree::GetEntryNumberWithIndex(Int_t major, Int_t minor) const
{
// Return entry number corresponding to major and minor number
// Note that this function returns only the entry number, not the data
// To read the data corresponding to an entry number, use TTree::GetEntryWithIndex
// the BuildIndex function has created a table of Double_t* of sorted values
// corresponding to val = major + minor*1e-9;
// The function performs binary search in this sorted table.
// If it finds a pair that maches val, it returns directly the
// index in the table, otherwise it returns -1.
//
// See also GetEntryNumberWithBestIndex

   if (fIndex.fN == 0) return -1;
   Double_t value = major + minor*1e-9;
   Int_t i = TMath::BinarySearch(Int_t(fEntries), fIndexValues.fArray, value);
   if (i < 0) return -1;
   if (TMath::Abs(fIndexValues.fArray[i] - value) > 1.e-10) return -1;
   return fIndex.fArray[i];
}


//______________________________________________________________________________
Int_t TTree::GetEntryWithIndex(Int_t major, Int_t minor)
{
// Return entry  corresponding to major and minor number
// For example:
//     Int_t run   = 1234;
//     Int_t event = 345;
//     Int_t serial= tree.GetEntryNumberWithIndex(run,event);
//    now the variable serial is in the range [0,nentries] and one can do
//    tree.GetEntry(serial);

   Int_t serial = GetEntryNumberWithIndex(major, minor);
   if (serial < 0) return -1;
   return GetEntry(serial);
}

//______________________________________________________________________________
TIterator* TTree::GetIteratorOnAllLeaves(Bool_t dir)
{
// Creates a new iterator that will go through all the leaves on the tree
// itself and its friend.

   return new TTreeFriendLeafIter(this,dir);

}

//______________________________________________________________________________
TLeaf *TTree::GetLeaf(const char *aname)
{
// Return pointer to the 1st Leaf named name in any Branch of this Tree
// or any branch in the list of friend trees.
//
//  aname may be of the form branchname/leafname

   char *slash = (char*)strchr(aname,'/');
   char *name;
   Int_t nbch = 0;
   if (slash) {
      name = slash+1;
      nbch = slash-aname;
   } else {
      name = (char*)aname;
   }
   TLeaf *leaf = 0;
   TIter nextl(GetListOfLeaves());
   while ((leaf = (TLeaf*)nextl())) {
      if (strcmp(leaf->GetName(),name)) continue;
      if (slash) {
         if (strncmp(leaf->GetBranch()->GetName(),aname,nbch)) continue;
      }
      return leaf;
   }

   if (!fFriends) return 0;
   TIter next(fFriends);
   TFriendElement *fe;
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      leaf = t->GetLeaf(name);
      if (leaf) return leaf;
   }

   //second pass in the list of friends when the leaf name
   //is prefixed by the tree name
   next.Reset();
   while ((fe = (TFriendElement*)next())) {
      TTree *t = fe->GetTree();
      char *subname = (char*)strstr(name,fe->GetName());
      if (subname != name) continue;
      Int_t l = strlen(fe->GetName());
      subname += l;
      if (*subname != '.') continue;
      subname++;
      leaf = t->GetLeaf(subname);
      if (leaf) return leaf;
   }
   return 0;
}




//______________________________________________________________________________
Double_t TTree::GetMaximum(const char *columname)
{
//*-*-*-*-*-*-*-*-*Return maximum of column with name columname*-*-*-*-*-*-*
//*-*              ============================================

   TLeaf *leaf = this->GetLeaf(columname);
   if (!leaf) return 0;
   TBranch *branch = leaf->GetBranch();
   Double_t cmax = -FLT_MAX; //in float.h
   for (Int_t i=0;i<fEntries;i++) {
      branch->GetEntry(i);
      Double_t val = leaf->GetValue();
      if (val > cmax) cmax = val;
   }
   return cmax;
}


//______________________________________________________________________________
Double_t TTree::GetMinimum(const char *columname)
{
//*-*-*-*-*-*-*-*-*Return minimum of column with name columname*-*-*-*-*-*-*
//*-*              ============================================

   TLeaf *leaf = this->GetLeaf(columname);
   if (!leaf) return 0;
   TBranch *branch = leaf->GetBranch();
   Double_t cmin = FLT_MAX; //in float.h
   for (Int_t i=0;i<fEntries;i++) {
      branch->GetEntry(i);
      Double_t val = leaf->GetValue();
      if (val < cmin) cmin = val;
   }
   return cmin;
}


//______________________________________________________________________________
const char *TTree::GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex) const
{
//*-*-*-*-*-*-*-*-*Return name corresponding to colindex in varexp*-*-*-*-*-*
//*-*              ===============================================
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
TVirtualTreePlayer *TTree::GetPlayer()
{
   // Load the TTreePlayer (if not already done)
   // Pointer to player is fPlayer

   if (fPlayer) return fPlayer;
   fPlayer = TVirtualTreePlayer::TreePlayer(this);
   return fPlayer;
}

//______________________________________________________________________________
Int_t TTree::LoadTree(Int_t entry)
{
//*-*-*-*-*-*-*-*-*Set current Tree entry
//*-*              ======================

// this function is overloaded in TChain

   if (fNotify) {
      if (fReadEntry < 0) fNotify->Notify();
   }
   fReadEntry = entry;
   return fReadEntry;

}

//______________________________________________________________________________
void TTree::Loop(Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Loop on nentries of this tree starting at firstentry
//*-*              ===================================================

   GetPlayer();
   if (!fPlayer) return;
   fPlayer->Loop(option,nentries,firstentry);
}

//______________________________________________________________________________
Int_t TTree::MakeSelector(const char *selector)
{
//====>
//*-*-*-*-*-*-*Generate skeleton selector class for this Tree*-*-*-*-*-*-*
//*-*          ==============================================
//
//   The following files are produced: selector.h and selector.C
//   if selector is NULL, selector will be nameoftree.
//
//   The generated code in selector.h includes the following:
//      - Identification of the original Tree and Input file name
//      - Definition of selector class (data and functions)
//      - the following class functions:
//         - constructor and destructor
//         - void    Begin(TTree *tree)
//         - void    Init(TTree *tree)
//         - Bool_t  Notify()
//         - Bool_t  ProcessCut(Int-t entry)
//         - void    ProcessFill(Int-t entry)
//         - void    Terminate
//
//   The class selector derives from TSelector.
//   The generated code in selector.C includes empty functions defined above:
//
//   To use this function:
//      - connect your Tree file (eg: TFile f("myfile.root");)
//      - T->MakeSelector("myselect");
//    where T is the name of the Tree in file myfile.root
//    and myselect.h, myselect.C the name of the files created by this function.
//   In a Root session, you can do:
//      Root > T->Process("select.C")
//
//====>

   return MakeClass(selector,"selector");
}

//______________________________________________________________________________
Int_t TTree::MakeClass(const char *classname, Option_t *option)
{
//====>
//*-*-*-*-*-*-*Generate skeleton analysis class for this Tree*-*-*-*-*-*-*
//*-*          ==============================================
//
//   The following files are produced: classname.h and classname.C
//   if classname is NULL, classname will be nameoftree.
//
//   When the option "anal" is specified, the function generates the
//   analysis class described in TTree::makeAnal.
//
//   The generated code in classname.h includes the following:
//      - Identification of the original Tree and Input file name
//      - Definition of analysis class (data and functions)
//      - the following class functions:
//         -constructor (connecting by default the Tree file)
//         -GetEntry(Int_t entry)
//         -Init(TTree *tree) to initialize a new TTree
//         -Show(Int_t entry) to read and Dump entry
//
//   The generated code in classname.C includes only the main
//   analysis function Loop.
//
//   To use this function:
//      - connect your Tree file (eg: TFile f("myfile.root");)
//      - T->MakeClass("MyClass");
//    where T is the name of the Tree in file myfile.root
//    and MyClass.h, MyClass.C the name of the files created by this function.
//   In a Root session, you can do:
//      Root > .L MyClass.C
//      Root > MyClass t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//
//====>

   GetPlayer();
   if (!fPlayer) return 0;
   return fPlayer->MakeClass(classname,option);
}


//______________________________________________________________________________
Int_t TTree::MakeCode(const char *filename)
{
//====>
//*-*-*-*-*-*-*-*-*Generate skeleton function for this Tree*-*-*-*-*-*-*
//*-*              ========================================
//
//   The function code is written on filename
//   if filename is NULL, filename will be nameoftree.C
//
//   The generated code includes the following:
//      - Identification of the original Tree and Input file name
//      - Connection of the Tree file
//      - Declaration of Tree variables
//      - Setting of branches addresses
//      - a skeleton for the entry loop
//
//   To use this function:
//      - connect your Tree file (eg: TFile f("myfile.root");)
//      - T->MakeCode("anal.C");
//    where T is the name of the Tree in file myfile.root
//    and anal.C the name of the file created by this function.
//
//   NOTE: Since the implementation of this function, a new and better
//         function TTree::MakeClass has been developped.
//
//          Author: Rene Brun
//====>

   Warning("MakeCode","MakeCode is obsolete. Use MakeClass or MakeSelector instead");

   GetPlayer();
   if (!fPlayer) return 0;
   return fPlayer->MakeCode(filename);
}

//______________________________________________________________________________
void TTree::MakeIndex(TString &varexp, Int_t *index)
{
//*-*-*-*-*-*-*-*-*Build Index array for names in varexp*-*-*-*-*-*-*-*-*-*-*
//*-*              =====================================

   Int_t ivar = 1;
   index[0]  = -1;
   for (Int_t i=0;i<varexp.Length();i++) {
      if (varexp[i] == ':') {
         index[ivar] = i;
         ivar++;
      }
   }
   index[ivar] = varexp.Length();
}

//______________________________________________________________________________
Bool_t TTree::MemoryFull(Int_t nbytes)
{
//*-*-*-*-*-*Check if adding nbytes to memory we are still below MaxVirtualsize
//*-*        ==================================================================

   if (fTotalBuffers + nbytes < fMaxVirtualSize) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TTree::Notify()
{
// function called when loading a new class library

   TIter next(GetListOfLeaves());
   TLeaf *leaf;
   while ((leaf = (TLeaf*)next())) {
      leaf->Notify();
      leaf->GetBranch()->Notify();
   }
   return kTRUE;
}

//______________________________________________________________________________
TPrincipal *TTree::Principal(const char *varexp, const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Interface to the Principal Components Analysis class*-*-*
//*-*              ====================================================
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
   if (fPlayer) return fPlayer->Principal(varexp,selection,option,nentries,firstentry);
   else         return 0;
}

//______________________________________________________________________________
void TTree::Print(Option_t *option) const
{
   // Print a summary of the Tree contents.
   // if option contains "all" friend trees are also printed.
   // if option contains "toponly" only the top level branches are printed.
   //
   // Wildcarding can be used to print only a subset of the branches
   // eg, T.Print("Elec*") will print all branches with name starting with "Elec"

  if (!strcasecmp(option, "p") || !strcasecmp(option, "pa")) {
#ifdef NEVER
          TPacketGenerator *t = GetPacketGenerator();
     if (!t) {
        Printf("No TPacketGenerator object available");
        return;
     }
     if (!strcasecmp(option, "p"))
        t->Print();
     else
        t->Print("all");
     return;
#endif
  }

  Int_t s = 0;
  if (fDirectory) {
     TKey *key = fDirectory->GetKey(GetName());
     if (key) s = key->GetNbytes();
  }
  Double_t total = fTotBytes + s;
  Int_t file     = Int_t(fZipBytes) + s;
  Float_t cx     = 1;
  if (fZipBytes) cx = fTotBytes/fZipBytes;
  Printf("******************************************************************************");
  Printf("*Tree    :%-10s: %-54s *",GetName(),GetTitle());
  Printf("*Entries : %8d : Total = %15.10g bytes  File  Size = %10d *",Int_t(fEntries),total,file);
  Printf("*        :          : Tree compression factor = %6.2f                       *",cx);
  Printf("******************************************************************************");

  TBranch *br;
  if (strstr(option,"toponly")) {
     Int_t nl = ((TTree*)this)->GetListOfLeaves()->GetEntries();
     TLeaf *leaf;
     Int_t *count = new Int_t[nl];
     Int_t l;
     Int_t keep =0;
     for (l=0;l<nl;l++) {
        leaf = (TLeaf *)((TTree*)this)->GetListOfLeaves()->At(l);
        br   = leaf->GetBranch();
        if (strchr(br->GetName(),'.')) {
           count[l] = -1;
           count[keep] += (Int_t)br->GetZipBytes();
        } else {
           keep = l;
           count[keep]  = (Int_t)br->GetZipBytes();
        }
     }
     for (l=0;l<nl;l++) {
        if (count[l] < 0) continue;
        leaf = (TLeaf *)((TTree*)this)->GetListOfLeaves()->At(l);
        br   = leaf->GetBranch();
        printf("branch: %-20s %9d\n",br->GetName(),count[l]);
     }
     delete [] count;
  } else {
     TString reg = "*";
     if (strlen(option) && strchr(option,'*')) reg = option;
     TRegexp re(reg,kTRUE);
     TIter next(((TTree*)this)->GetListOfBranches());
     TBranch::ResetCount();
     while ((br= (TBranch*)next())) {
        TString s = br->GetName();
        if (s.Index(re) == kNPOS) continue;
        br->Print(option);
     }
  }

  //print friends if option "all"
  if (!fFriends || !strstr(option,"all")) return;
  TIter nextf(fFriends);
  TFriendElement *fr;
  while ((fr = (TFriendElement*)nextf())) {
     fr->GetTree()->Print(option);
  }
}

//______________________________________________________________________________
Int_t TTree::Process(const char *filename,Option_t *option,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Process this tree executing the code in filename*-*-*-*-*
//*-*              ================================================
//
//   The code in filename is loaded (interpreted or compiled , see below)
//   filename must contain a valid class implementation derived from TSelector.
//   where TSelector has the following member functions:
//
//     void TSelector::Begin(). This function is called before looping on the
//          events in the Tree. The user can create his histograms in this function.
//
//     Bool_t TSelector::ProcessCut(Int_t entry). This function is called
//          before processing entry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise.
//     void TSelector::ProcessFill(Int_t entry). This function is called for
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

   GetPlayer();
   if (fPlayer) return fPlayer->Process(filename,option,nentries,firstentry);
   else         return -1;
}

//______________________________________________________________________________
Int_t TTree::Process(TSelector *selector,Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Process this tree executing the code in selector*-*-*-*-*
//*-*              ================================================
//
//   The TSelector class has the following member functions:
//
//     void TSelector::Begin(). This function is called before looping on the
//          events in the Tree. The user can create his histograms in this function.
//
//     Bool_t TSelector::ProcessCut(Int_t entry). This function is called
//          before processing entry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise.
//     void TSelector::ProcessFill(Int_t entry). This function is called for
//          all selected events. User fills histograms in this function.
//     void TSelector::Terminate(). This function is called at the end of
//          the loop on all events.
//     void TTreeProcess::Begin(). This function is called before looping on the
//          events in the Tree. The user can create his histograms in this function.

   GetPlayer();
   if (fPlayer) return fPlayer->Process(selector,option,nentries,firstentry);
   else         return -1;
}

//______________________________________________________________________________
Int_t TTree::Project(const char *hname, const char *varexp, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Make a projection of a Tree using selections*-*-*-*-*-*-*
//*-*              =============================================
//
//   Depending on the value of varexp (described in Draw) a 1-D,2-D,etc
//   projection of the Tree will be filled in histogram hname.
//   Note that the dimension of hname must match with the dimension of varexp.
//

   Int_t nch = strlen(hname) + strlen(varexp);
   char *var = new char[nch+5];
   sprintf(var,"%s>>%s",varexp,hname);
   nch = strlen(option) + 10;
   char *opt = new char[nch];
   if (option) sprintf(opt,"%sgoff",option);
   else        strcpy(opt,"goff");

   Int_t nsel = Draw(var,selection,opt,nentries,firstentry);

   delete [] var;
   delete [] opt;
   return nsel;
}

//______________________________________________________________________________
TSQLResult *TTree::Query(const char *varexp, const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
   // Loop on Tree & return TSQLResult object containing entries following selection

   GetPlayer();
   if (fPlayer) return fPlayer->Query(varexp,selection,option,nentries,firstentry);
   return 0;
}

//______________________________________________________________________________
void TTree::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset buffers and entries count in all branches/leaves*-*-*
//*-*            ======================================================

   fNotify         = 0;
   fEntries        = 0;
   fTotBytes       = 0;
   fZipBytes       = 0;
   fSavedBytes     = 0;
   fTotalBuffers   = 0;
   fChainOffset    = 0;
   fReadEntry      = -1;

   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nb;i++)  {
      TBranch *branch = (TBranch*)fBranches.UncheckedAt(i);
      branch->Reset(option);
   }
}

//______________________________________________________________________________
Int_t  TTree::Scan(const char *varexp, const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Loop on Tree & print entries following selection*-*-*-*-*-*
//*-*              ===============================================

   GetPlayer();
   if (fPlayer) return fPlayer->Scan(varexp,selection,option,nentries,firstentry);
   else         return -1;
}

//_______________________________________________________________________
 void TTree::SetBasketSize(const char *bname, Int_t buffsize)
{
//*-*-*-*-*-*-*-*-*Set branc(es) basket size*-*-*-*-*-*-*-*
//*-*              =========================
//
//     bname is the name of a branch.
//     if bname="*", apply to all branches.
//     if bname="xxx*", apply to all branches with name starting with xxx
//     see TRegexp for wildcarding options
//     buffsize = branc basket size

   TBranch *branch;
   TLeaf *leaf;

   Int_t i;
   Int_t nleaves = fLeaves.GetEntriesFast();
   TRegexp re(bname,kTRUE);
   Int_t nb = 0;
   for (i=0;i<nleaves;i++)  {
      leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      branch = (TBranch*)leaf->GetBranch();
      TString s = branch->GetName();
      if (strcmp(bname,branch->GetName()) && s.Index(re) == kNPOS) continue;
      nb++;
      branch->SetBasketSize(buffsize);
   }
   if (!nb) {
      Error("SetBasketSize", "unknown branch -> %s", bname);
   }
}

//_______________________________________________________________________
 void TTree::SetBranchAddress(const char *bname, void *add)
{
//*-*-*-*-*-*-*-*-*Set branch address*-*-*-*-*-*-*-*
//*-*              ==================
//
//      If object is a TTree, this function is only an interface to TBranch::SetAddress
//      Function overloaded by TChain.

   TBranch *branch = GetBranch(bname);
   if (branch) branch->SetAddress(add);
   else        Error("SetBranchAddress", "unknown branch -> %s", bname);
}

//_______________________________________________________________________
void TTree::SetBranchStatus(const char *bname, Bool_t status)
{
//*-*-*-*-*-*-*-*-*Set branch status Process or DoNotProcess*-*-*-*-*-*-*-*
//*-*              =========================================
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
//  An alternative to this function is to read directly and only
//  the interesting branches. Example:
//    TBranch *brc = T.GetBranch("c");
//    TBranch *bre = T.GetBranch("e");
//    brc->GetEntry(i);
//    bre->GetEntry(i);


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
      if (strcmp(bname,branch->GetName()) && s.Index(re) == kNPOS) continue;
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
   if (!nb) {
      Error("SetBranchStatus", "unknown branch -> %s", bname);
      return;
   }


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
  // static function setting the current branch style
  // style = 0 old Branch
  // style = 1 new Bronch

   fgBranchStyle = style;
}

//______________________________________________________________________________
void TTree::SetDebug(Int_t level, Int_t min, Int_t max)
{
   // Set the debug level and the debug range
   // for entries in the debug range, the functions TBranchElement::Fill
   // and TBranchElement::GetEntry will print the number of bytes filled
   // or read for each branch.

   fDebug    = level;
   fDebugMin = min;
   fDebugMax = max;
}

//______________________________________________________________________________
void TTree::SetDirectory(TDirectory *dir)
{
   // Remove reference to this tree from current directory and add
   // reference to new directory dir. dir can be 0 in which case the tree
   // does not belong to any directory.

   if (fDirectory == dir) return;
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fDirectory = dir;
   if (fDirectory) fDirectory->GetList()->Add(this);
   TFile *file = 0;
   if (fDirectory) file = fDirectory->GetFile();
   TBranch * b;
   TIter next(GetListOfBranches());
   while((b = (TBranch*)next())){
      b->SetFile(file);
   }
}

//_______________________________________________________________________
void TTree::SetEstimate(Int_t n)
{
//*-*-*-*-*-*-*-*-*Set number of entries to estimate variable limits*-*-*-*
//*-*              ================================================

   if (n<=0) n = 10000;
   fEstimate = n;
   GetPlayer();
   if (fPlayer) fPlayer->SetEstimate(n);
}

//______________________________________________________________________________
void TTree::SetName(const char *name)
{
// Change the name of this Tree
//
   if (gPad) gPad->Modified();

//  Trees are named objects in a THashList.
//  We must update the hashlist if we change the name
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fName = name;
   if (fDirectory) fDirectory->GetList()->Add(this);
}

//______________________________________________________________________________
void TTree::SetObject(const char *name, const char *title)
{
// Change the name and title of this Tree
//
   if (gPad) gPad->Modified();

//  Trees are named objects in a THashList.
//  We must update the hashlist if we change the name
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fName  = name;
   fTitle = title;
   if (fDirectory) fDirectory->GetList()->Add(this);
}

//______________________________________________________________________________
void TTree::SetWeight(Double_t w, Option_t *)
{
//  Set tree weight.
//  The weight is used by TTree::Draw to automatically weight each
//  selected entry in the resulting histogram.
//  For example the equivalent of
//     T.Draw("x","w")
//  is
//     T.SetWeight(w);
//     T.Draw("x");
//
// This function is redefined by TChain::SetWeight. In case of a TChain,
// an option "global" may be specified to set the same weight
// for all Trees in the TChain instead of the default behaviour
// using the weights of each Tree in the chain. (see TChain::SetWeight)

   fWeight = w;
}

//_______________________________________________________________________
void TTree::Show(Int_t entry)
{
//*-*-*-*-*-*Print values of all active leaves for entry*-*-*-*-*-*-*-*
//*-*        ===========================================
// if entry==-1, print current entry (default)

   if (entry != -1) GetEntry(entry);
   printf("======> EVENT:%d\n",fReadEntry);
   TObjArray *leaves  = GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (Int_t i=0;i<nleaves;i++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(i);
      TBranch *branch = leaf->GetBranch();
      if (branch->TestBit(kDoNotProcess)) continue;
      Int_t len = leaf->GetLen();
      if (len <= 0) continue;
      len = TMath::Min(len,10);
      if (leaf->IsA() == TLeafElement::Class()) {leaf->PrintValue(len); continue;}
      if (branch->GetListOfBranches()->GetEntriesFast() > 0) continue;
      if (leaf->IsA() == TLeafF::Class()) len = TMath::Min(len,5);
      if (leaf->IsA() == TLeafD::Class()) len = TMath::Min(len,5);
      if (leaf->IsA() == TLeafC::Class()) len = 1;
      printf(" %-15s = ",leaf->GetName());
      for (Int_t l=0;l<len;l++) {
         leaf->PrintValue(l);
         if (l == len-1) printf("\n");
         else            printf(", ");
      }
   }
}

//_______________________________________________________________________
void TTree::StartViewer()
{
//*-*-*-*-*-*-*-*-*Start the TTreeViewer on this TTree*-*-*-*-*-*-*-*-*-*
//*-*              ===================================
//
//  ww is the width of the canvas in pixels
//  wh is the height of the canvas in pixels

   GetPlayer();
   if (fPlayer) fPlayer->StartViewer(600,400);
}

//_______________________________________________________________________
void TTree::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      gTree = this;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 4) {
         fDirectory = gDirectory;
         TTree::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         if (fEstimate <= 10000) fEstimate = 1000000;
         fSavedBytes = fTotBytes;
         gDirectory->Append(this);
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      TAttMarker::Streamer(b);
      b >> fScanField;
      b >> fMaxEntryLoop;
      b >> fMaxVirtualSize;
      b >> fEntries;
      b >> fTotBytes;
      b >> fZipBytes;
      b >> fAutoSave;
      b >> fEstimate;
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
      TTree::Class()->WriteBuffer(b,this);
   }
}

//______________________________________________________________________________
Int_t TTree::UnbinnedFit(const char *funcname ,const char *varexp, const char *selection,Option_t *option ,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*Unbinned fit of one or more variable(s) from a Tree*-*-*-*-*-*
//*-*        ===================================================
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
//   In practice it is convenient to have a normalization variable
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
   if (fPlayer) return fPlayer->UnbinnedFit(funcname,varexp,selection,option,nentries,firstentry);
   else         return -1;
}

//______________________________________________________________________________
void TTree::UseCurrentStyle()
{
//*-*-*-*-*-*Replace current attributes by current style*-*-*-*-*
//*-*        ===========================================

   SetFillColor(gStyle->GetHistFillColor());
   SetFillStyle(gStyle->GetHistFillStyle());
   SetLineColor(gStyle->GetHistLineColor());
   SetLineStyle(gStyle->GetHistLineStyle());
   SetLineWidth(gStyle->GetHistLineWidth());
   SetMarkerColor(gStyle->GetMarkerColor());
   SetMarkerStyle(gStyle->GetMarkerStyle());
   SetMarkerSize(gStyle->GetMarkerSize());
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
TTreeFriendLeafIter::TTreeFriendLeafIter(const TTree * tree, Bool_t dir)
  : fTree((TTree*)tree),fLeafIter(0),fTreeIter(0),fDirection(dir)
{
   // Create a new iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.
}

//______________________________________________________________________________
TTreeFriendLeafIter::TTreeFriendLeafIter(const TTreeFriendLeafIter&iter)
{
  // Copy constructor

  fTree = iter.fTree;
  fDirection = iter.fDirection;

}

//______________________________________________________________________________
TIterator &TTreeFriendLeafIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TTreeFriendLeafIter::Class()) {
      const TTreeFriendLeafIter &rhs1 = (const TTreeFriendLeafIter &)rhs;
      fDirection = rhs1.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TTreeFriendLeafIter &TTreeFriendLeafIter::operator=(const TTreeFriendLeafIter &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs ) {
      fDirection = rhs.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TTreeFriendLeafIter::Next()
{
   if (!fTree) return 0;

   TObject * next;
   TTree * nextTree;

   if (!fLeafIter) {
     fLeafIter =  fTree->GetListOfLeaves()->MakeIterator(fDirection);
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
         nextTree = (TTree*)nextFriend->GetTree();
         if (!nextTree) return 0;
         SafeDelete(fLeafIter);
         fLeafIter = nextTree->GetListOfLeaves()->MakeIterator(fDirection);
         next = fLeafIter->Next();
      }
   }
   return next;
}

//______________________________________________________________________________
Option_t *TTreeFriendLeafIter::GetOption() const
{
   // Returns the object option stored in the list.

   if (fLeafIter) return fLeafIter->GetOption();
   return "";
}

