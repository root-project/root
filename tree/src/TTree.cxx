// @(#)root/tree:$Name:  $:$Id: TTree.cxx,v 1.43 2001/01/16 16:25:58 brun Exp $
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
// #include "TROOT.h"
// #include "TFile.h"
// #include "TH1.h"
// #include "TH2.h"
// #include "TProfile.h"
// #include "TRandom.h"
// #include "TTree.h"
//
//
// TROOT simple("simple","Histograms and trees");
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
#include <float.h>

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
#include "TBasket.h"
#include "TMath.h"
#include "TDirectory.h"
#include "TBranchClones.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBrowser.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TRegexp.h"
#include "TArrayC.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TVirtualFitter.h"

TTree *gTree;

ClassImp(TTree)

//______________________________________________________________________________
TTree::TTree(): TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*Default Tree constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   fDirectory      = 0;
   fTotalBuffers   = 0;
   fChainOffset    = 0;
   fReadEntry      = -1;
   fUpdate         = 0;
   fEventList      = 0;
   fPacketSize     = 100;
   fTimerInterval  = 0;
   fPlayer         = 0;
}

//______________________________________________________________________________
TTree::TTree(const char *name,const char *title, Int_t maxvirtualsize)
    :TNamed(name,title)
{
//*-*-*-*-*-*-*-*-*-*Normal Tree constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ======================
//
//   The Tree is created in the current directory
//   Use the various functions Branch below to add branches to this Tree.

   fScanField      = 25;
   fMaxEntryLoop   = 1000000000;
   fMaxVirtualSize = maxvirtualsize;
   fDirectory      = gDirectory;
   fEntries        = 0;
   fTotBytes       = 0;
   fZipBytes       = 0;
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

   SetFillColor(gStyle->GetHistFillColor());
   SetFillStyle(gStyle->GetHistFillStyle());
   SetLineColor(gStyle->GetHistLineColor());
   SetLineStyle(gStyle->GetHistLineStyle());
   SetLineWidth(gStyle->GetHistLineWidth());
   SetMarkerColor(gStyle->GetMarkerColor());
   SetMarkerStyle(gStyle->GetMarkerStyle());
   SetMarkerSize(gStyle->GetMarkerSize());

   gDirectory->Append(this);
}

//______________________________________________________________________________
TTree::~TTree()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   if (fDirectory) {
      if (!fDirectory->TestBit(TDirectory::kCloseDirectory))
         fDirectory->GetList()->Remove(this);
   }
   fBranches.Delete();
   fDirectory  = 0;
   delete fPlayer;
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
   fBranches.Add(branch);
   return branch;
}

//______________________________________________________________________________
TBranch *TTree::Branch(const char *name, const char *classname, void *addobj, Int_t bufsize, Int_t splitlevel)
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
TBranch *TTree::Branch(const char *name, void *clonesaddress, Int_t bufsize, Int_t splitlevel)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new TTree BranchClones*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
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
   if (splitlevel) {
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
Int_t TTree::Branch(TList *list, Int_t bufsize)
{
//   This new function creates one branch for each element in the list.
//   Two cases are supported:
//      list[i] is a TObject*: a TBranchObject is created with a branch name
//                             being the name of the object.
//      list[i] is a TClonesArray*: A TBranchClones is created.
//      if list[i]->TestBit(TClonesArray::kNoSplit) is 1, the TClonesArray
//      is not split.
//      if list[i]->TestBit(TClonesArray::kForgetBits) is 1 and the TClonesArray
//      is split, then no branches are created for the fBits and fUniqueID
//      of the TObject part of the class referenced by the TClonesArray.
//   The function returns the total number of branches created.

   if (list == 0) return 0;
   TObject *obj;
   Int_t nbranches = GetListOfBranches()->GetEntries();
   TObjLink *lnk = list->FirstLink();

   Int_t splitlevel = 1;
   while (lnk) {
      obj = lnk->GetObject();
      if (obj->InheritsFrom(TClonesArray::Class())) {
         TClonesArray *clones = (TClonesArray*)obj;
         splitlevel = 1;
         if (clones->TestBit(TClonesArray::kForgetBits)) splitlevel = 2;
         if (clones->TestBit(TClonesArray::kNoSplit))    splitlevel = 0;
         Branch(clones->GetName(),lnk->GetObjectRef(),bufsize,splitlevel);
      } else {
         splitlevel = 0;
         Branch(obj->GetName(),obj->ClassName(),lnk->GetObjectRef(),bufsize,splitlevel);
      }
      lnk = lnk->Next();
   }
   return GetListOfBranches()->GetEntries() - nbranches;
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
   Int_t n = Int_t(fEntries);
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
void TTree::BuildStreamerInfo(TClass *cl, void *pointer)
{
  // Build StreamerInfo for class cl
  // pointer is an optional argument that may contain a pointer to an object of cl
   
   cl->BuildRealData(pointer);
   TStreamerInfo *sinfo = cl->GetStreamerInfo(cl->GetClassVersion());
   //if (!sinfo->GetTypes() || sinfo->IsOptimized()) sinfo->BuildOld();
   if (gFile) {
      TArrayC *cindex = gFile->GetClassIndex();
      Int_t number = sinfo->GetNumber();
      if (cindex->fArray[number] == 0) {
         cindex->fArray[0]       = 1;
         cindex->fArray[number]  = 1;
      }
   }
}

//______________________________________________________________________________
TTree *TTree::CloneTree(Int_t nentries, Option_t *option)
{
// Create a clone of this tree and copy nentries
// By default copy all entries
// option is reserved for future use
//        plan to implement option "ACTIVE" to copy only active branches
//
// IMPORTANT: Before invoking this function, the branch addresses
//            of this TTree must have been set.
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

  // delete non active branches from the clone if option "ACTIVE" has been specified
   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("ACTIVE")) {
      // this block should be implemented
   }

  // copy branch addresses starting from branches
   Int_t i;
   TObjArray *branches  = GetListOfBranches();
   Int_t nbranches = branches->GetEntriesFast();
   for (i=0;i<nbranches;i++) {
      TBranch *branch = (TBranch*)branches->UncheckedAt(i);
      if (branch->GetAddress()) {
         tree->SetBranchAddress(branch->GetName(),branch->GetAddress());
      }
   }
  // copy branch addresses starting from leaves
   TObjArray *leaves  = GetListOfLeaves();
   TObjArray *tleaves = tree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   for (i=0;i<nleaves;i++) {
      TLeaf *leaf2 = (TLeaf*)tleaves->UncheckedAt(i);
      TLeaf *leaf  = (TLeaf*)leaves->UncheckedAt(i);
      TBranch *branch  = leaf->GetBranch();
  //    TBranch *branch2 = leaf2->GetBranch();
  //    branch2->SetCompressionLevel(branch2->GetFile()->GetCompressionLevel());
      if (branch->GetAddress()) {
         tree->SetBranchAddress(branch->GetName(),branch->GetAddress());
      } else {
         leaf2->SetAddress(leaf->GetValuePointer());
      }
   }

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
      if (!gFile->IsWritable()) {
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
//      selection2 = "(x+y)*(sqrt(z)>3.2"
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
//  If varexp0 contains >>hnew (following the variable(s) name(s),
//  the new histogram created is called hnew and it is kept in the current
//  directory.
//  Example:
//    tree.Draw("sqrt(x)>>hsqrt","y>0")
//    will draw sqrt(x) and save the histogram as "hsqrt" in the current
//    directory.
//
//  By default, the specified histogram is reset.
//  To continue to append data to an existing histogram, use "+" in front
//  of the histogram name;
//    tree.Draw("sqrt(x)>>+hsqrt","y>0")
//      will not reset hsqrt, but will continue filling.
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
//*-*-*-*-*-*Return pointer to the branch with name*-*-*-*-*-*-*-*
//*-*        ======================================

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
         if (!strcmp(b1->GetName(),name)) return b1;
         lb1 = b1->GetListOfBranches();
         nb2 = lb1->GetEntriesFast();
         for (k=0;k<nb2;k++) {
            b2 = (TBranch*)lb1->UncheckedAt(k);
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
   return 0;
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
Int_t TTree::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*-*Read all branches of entry and return total number of bytes*-*-*
//*-*        ===========================================================
//     getall = 0 : get only active branches
//     getall = 1 : get all branches

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
Int_t TTree::GetEntryNumberWithIndex(Int_t major, Int_t minor) const
{
// Return entry number corresponding to major and minor number
// Note that this function returns only the entry number, not the data
// To read the data corresponding to an entry number, use TTree::GetEntryWithIndex
// the BuildIndex function has created a table of Double_t* of sorted values
// corresponding to val = major + minor*1e-9;
// The function performs binary search in this sorted table.
// If it find an array value that maches val, it returns directly the
// index in the table.
// If an entry corresponding to major and minor is not found, the function
// returns a value = -lowest -1 where lowest is the entry number in the table
// immediatly lower than the requested value.
   
   if (fIndex.fN == 0) return -1;
   Double_t value = major + minor*1e-9;
   Int_t i = TMath::BinarySearch(Int_t(fEntries), fIndexValues.fArray, value);
   if (TMath::Abs(fIndexValues.fArray[i] - value) > 1.e-10) return -1-i;
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
TLeaf *TTree::GetLeaf(const char *name) 
{
//*-*-*-*-*-*Return pointer to the 1st Leaf named name in any Branch-*-*-*-*-*
//*-*        =======================================================

   return (TLeaf*)fLeaves.FindObject(name);
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
   // Print a summary of the Tree contents. In case options are "p" or "pa"
   // print information about the TPacketGenerator ("pa" is equivalent to
   // TPacketGenerator::Print("all")).

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

  TIter next(((TTree*)this)->GetListOfBranches());
  TBranch *br;
  TBranch::ResetCount();
  while ((br= (TBranch*)next())) {
     br->Print(option);
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
//      and dynamically loaded. The corresponding binary file and shared library
//      will be deleted at the end of the function.
//   if filename is of the form file.C+, the file file.C will be compiled
//      and dynamically loaded. The corresponding binary file and shared library
//      will be kept at the end of the function. At next call, if file.C
//      is older than file.o and file.so, the file.C is not compiled, only
//      file.so is loaded.
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
//     bname is the name of a branch.
//     if bname="*", apply to all branches.
//     if bname="xxx*", apply to all branches with name starting with xxx
//     see TRegexp for wildcarding options
//      status = 1  branch will be processed
//             = 0  branch will not be processed

   
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
         Int_t nbranches = branch->GetListOfBranches()->GetEntriesFast();
         for (j=0;j<nbranches;j++) {
            bson = (TBranch*)branch->GetListOfBranches()->UncheckedAt(j);
            if (!bson->TestBit(kDoNotProcess)) {
               branch->ResetBit(kDoNotProcess);
               break;
            }
         }
      }
   }
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
      if (branch->GetListOfBranches()->GetEntriesFast() > 0) continue;
      Int_t len = leaf->GetLen();
      if (len <= 0) continue;
      if (leaf->IsA() == TLeafF::Class()) len = TMath::Min(len,5);
      if (leaf->IsA() == TLeafD::Class()) len = TMath::Min(len,5);
      len = TMath::Min(len,10);
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
         TTree::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         if (fEstimate <= 10000) fEstimate = 1000000;
         fSavedBytes = fTotBytes;
         fDirectory = gDirectory;
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
TBranch *TTree::Trunk(const char *name, const char *classname, void *addobj, Int_t bufsize, Int_t splitlevel)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new TTree BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================
//
//    Build a TBranchElement for an object of class classname.
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
      Error("Trunk","Cannot find class:%s",classname);
      return 0;
   }
   TBranch *branch;
   if (splitlevel == 0) {
      branch = new TBranchObject(name,classname,addobj,bufsize,0);
      fBranches.Add(branch);
      return branch;
   }

   char **apointer = (char**)(addobj);
   TObject *obj = (TObject*)(*apointer);
   Bool_t delobj = kFALSE;
   if (!obj) {
      obj = (TObject*)cl->New();
      delobj = kTRUE;
   }
   //build the StreamerInfo if first time for the class
   TStreamerInfo::Optimize(kFALSE);
   BuildStreamerInfo(cl,obj);

   // create a dummy top level trunk branch
   TStreamerInfo *sinfo = cl->GetStreamerInfo();
   branch = new TBranchElement(name,sinfo,-1,addobj,bufsize,0);
   branch->SetName(name);
   fBranches.Add(branch);
   TObjArray *blist = branch->GetListOfBranches();   
   
//*-*- Loop on all public data members of the class and its base classes
   TIter next(sinfo->GetElements());
   TStreamerElement *element;
   Int_t id = 0;
   while ((element = (TStreamerElement*)next())) {
      TBranch *branch = new TBranchElement(element->GetName(),sinfo,id,addobj,bufsize,splitlevel-1);
      blist->Add(branch);
      id++;
   }
         
   if (delobj) delete obj;
   return branch;
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
