// @(#)root/treeplayer:$Name:  $:$Id: TTreePlayer.cxx,v 1.88 2002/01/29 17:33:47 brun Exp $
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
//        if splitlevel=1 (default), this branch will automatically be split
//          into subbranches, with one subbranch for each data member or object
//          of the object itself. In case the object member is a TClonesArray,
//          the mechanism described in case C is applied to this array.
//        if splitlevel=0, the object is serialized in the branch buffer.
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
//         See also the class TNtuple (a simple Tree with only one branch)
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

#include "snprintf.h"

#include "Riostream.h"
#include "TTreePlayer.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TSelector.h"
#include "TEventList.h"
#include "TBranchObject.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TLeafObject.h"
#include "TLeafF.h"
#include "TLeafD.h"
#include "TLeafC.h"
#include "TLeafB.h"
#include "TLeafI.h"
#include "TLeafS.h"
#include "TMath.h"
#include "TH2.h"
#include "TH3.h"
#include "TView.h"
#include "TPolyMarker.h"
#include "TPolyMarker3D.h"
#include "TDirectory.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TVirtualPad.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TTreeFormula.h"
#include "TGaxis.h"
#include "TBrowser.h"
#include "TStyle.h"
#include "TSocket.h"
#include "TSlave.h"
#include "TMessage.h"
#include "TPacketGenerator.h"
#include "TInterpreter.h"
#include "Foption.h"
#include "TTreeResult.h"
#include "TTreeRow.h"
#include "TPrincipal.h"
#include "Api.h"
#include "TChain.h"
#include "TChainElement.h"
#include "TF1.h"
#include "TVirtualFitter.h"
#include "TEnv.h"
#include "THLimitsFinder.h"

R__EXTERN Foption_t Foption;
R__EXTERN  TTree *gTree;

TVirtualFitter *tFitter=0;

extern void TreeUnbinnedFitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);

ClassImp(TTreePlayer)

//______________________________________________________________________________
TTreePlayer::TTreePlayer()
{
//*-*-*-*-*-*-*-*-*-*-*Default Tree constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   fTree           = 0;
   fV1             = 0;
   fV2             = 0;
   fV3             = 0;
   fW              = 0;
   fVar1           = 0;
   fVar2           = 0;
   fVar3           = 0;
   fVar4           = 0;
   fMultiplicity   = 0;
   fScanFileName   = 0;
   fScanRedirect   = kFALSE;
   fSelect         = 0;
   fSelectedRows   = 0;
   fDraw           = 0;
   fPacketGen      = 0;
   fPacketSize     = 100;
   fHistogram      = 0;
}

//______________________________________________________________________________
TTreePlayer::~TTreePlayer()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   ClearFormula();
   if (fV1)    delete [] fV1;
   if (fV2)    delete [] fV2;
   if (fV3)    delete [] fV3;
   if (fW)     delete [] fW;
   delete fPacketGen;
}

//______________________________________________________________________________
void TTreePlayer::ClearFormula()
{
//*-*-*-*-*-*-*Delete internal buffers*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*          =======================

   delete fVar1;   fVar1 = 0;
   delete fVar2;   fVar2 = 0;
   delete fVar3;   fVar3 = 0;
   delete fVar4;   fVar4 = 0;
   delete fSelect; fSelect = 0;
}

//______________________________________________________________________________
void TTreePlayer::CompileVariables(const char *varexp, const char *selection)
{
//*-*-*-*-*-*-*Compile input variables and selection expression*-*-*-*-*-*
//*-*          ================================================
//
//  varexp is an expression of the general form e1:e2:e3
//    where e1,etc is a formula referencing a combination of the columns
//  Example:
//     varexp = x     simplest case: draw a 1-Dim distribution of column named x
//            = sqrt(x)            : draw distribution of sqrt(x)
//            = x*y/z
//            = y:sqrt(x) 2-Dim dsitribution of y versus sqrt(x)
//
//  selection is an expression with a combination of the columns
//  Example:
//      selection = "x<y && sqrt(z)>3.2"
//       in a selection all the C++ operators are authorized
//
//

   const Int_t MAXCOL = 4;
   TString title;
   Int_t i,nch,ncols;
   Int_t index[MAXCOL];
//*-*- Compile selection expression if there is one
   fDimension = 0;
   ClearFormula();
   fMultiplicity = 0;
   Int_t force = 0;
   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return; }
      if (fSelect->GetMultiplicity() >= 1) fMultiplicity = fSelect;
      if (fSelect->GetMultiplicity() == -1) force = 4;
   }
//*-*- if varexp is empty, take first column by default
   nch = strlen(varexp);
   if (nch == 0) {fDimension = 0; return;}
   title = varexp;

//*-*- otherwise select only the specified columns
   ncols  = 1;
   for (i=0;i<nch;i++)  if (title[i] == ':') ncols++;
   if (ncols > 3 ) return;
   MakeIndex(title,index);

   fTree->ResetBit(TTree::kForceRead);
   if (ncols >= 1) {
      fVar1 = new TTreeFormula("Var1",GetNameByIndex(title,index,0),fTree);
      if (!fVar1->GetNdim()) { ClearFormula(); return;}
      if (!fMultiplicity && fVar1->GetMultiplicity() >= 1) fMultiplicity = fVar1;
      if (!force && fVar1->GetMultiplicity() == -1) force = 1;
   }
   if (ncols >= 2) {
      fVar2 = new TTreeFormula("Var2",GetNameByIndex(title,index,1),fTree);
      if (!fVar2->GetNdim()) { ClearFormula(); return;}
      if (!fMultiplicity && fVar2->GetMultiplicity() >= 1) fMultiplicity = fVar2;
      if (!force && fVar2->GetMultiplicity() == -1) force = 2;
   }
   if (ncols >= 3) {
      fVar3 = new TTreeFormula("Var3",GetNameByIndex(title,index,2),fTree);
      if (!fVar3->GetNdim()) { ClearFormula(); return;}
      if (!fMultiplicity && fVar3->GetMultiplicity()  >= 1) fMultiplicity = fVar3;
      if (!force && fVar3->GetMultiplicity() == -1) force = 3;
   }
   if (force) fTree->SetBit(TTree::kForceRead);

   fDimension    = ncols;
}

//______________________________________________________________________________
TTree *TTreePlayer::CopyTree(const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
{
   // copy a Tree with selection
   // make a clone of this Tree header.
   // then copy the selected entries
   //
   // selection is a standard selection expression (see TTreePlayer::Draw)
   // option is reserved for possible future use
   // nentries is the number of entries to process (default is all)
   // first is the first entry to process (default is 0)
   //
   // Note that the branch addresses must be correctly set before calling this function
   // The following example illustrates how to copy some events from the Tree
   // generated in $ROOTSYS/test/Event
   //
   //   gSystem->Load("libEvent");
   //   TFile f("Event.root");
   //   TTree *T = (TTree*)f.Get("T");
   //   Event *event = new Event();
   //   T->SetBranchAddress("event",&event);
   //   TFile f2("Event2.root","recreate");
   //   TTree *T2 = T->CopyTree("fNtrack<595");
   //   T2->Write();


  // we make a copy of the tree header
   TTree *tree;
   if (fTree->InheritsFrom("TChain")) {
      fTree->LoadTree(0);
      tree = (TTree*)fTree->GetTree()->CloneTree(0);
   } else {
      tree = (TTree*)fTree->CloneTree(0);
   }
   if (tree == 0) return 0;

   Int_t entry,entryNumber;
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > fTree->GetEntriesFast()-1) {
      lastentry  = (Int_t)fTree->GetEntriesFast() -1;
      nentries   = lastentry - firstentry + 1;
   }

   // Compile selection expression if there is one
   ClearFormula();
   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      if (!fSelect || !fSelect->GetNdim()) { delete fSelect; fSelect = 0; }
   }

   //loop on the specified entries
   Int_t tnumber = -1;
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Int_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         fTree->CopyAddresses(tree);
      }
      if (fSelect) {
         fSelect->GetNdata();
         if (fSelect->EvalInstance(0) == 0) continue;
      }
      fTree->GetEntry(entryNumber);
      tree->Fill();
   }
   delete fSelect; fSelect = 0;
   return tree;
}

//_______________________________________________________________________
void TTreePlayer::CreatePacketGenerator(Int_t nentries, Stat_t firstEntry)
{
   // Create or reset the packet generator.

}

//______________________________________________________________________________
Int_t TTreePlayer::DrawSelect(const char *varexp0, const char *selection, Option_t *option,Int_t nentries, Int_t firstentry)
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
// You can also specify the index as an expression of any other variables from the
// tree.
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
//  "fMatrix[][fResult[][]]"           30      on 1st dim of fMatrix then on both
//                                             dimensions of fResults.  The value
//                                             if fResults[j][k] is used as the second
//                                             index of fMatrix.
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
//    for (Int_t i0; i0 < min(3,2); i0++) {
//       use the value of (fMatrix[i0][2] - fMatrix[3][i0])
//    }
//
// So the loop equivalent to "fMatrix[][2] - fResults[][]" is:
//
//    for (Int_t i0; i0 < min(3,5); i0++) {
//       for (Int_t i1; i1 < 2; i1++) {
//          use the value of (fMatrix[i0][2] - fMatrix[i0][i1])
//       }
//    }
//
// So the loop equivalent to "fMatrix[][] - fResults[][]" is:
//
//    for (Int_t i0; i0 < min(3,5); i0++) {
//       for (Int_t i1; i1 < min(3,2); i1++) {
//          use the value of (fMatrix[i0][i1] - fResults[i0][i1])
//       }
//    }
//
// So the loop equivalent to "fMatrix[][fResults[][]]" is:
//
//    for (Int_t i0; i0 < 3; i0++) {
//       for (Int_t j2; j2 < 5; j2++) {
//          for (Int_t j3; j3 < 2; j3++) {
//             i1 = fResults[j2][j3];
//             use the value of fMatrix[i0][i1]
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
//    tree.Draw("sqrt(x)>>hsqrt(500,10,20)"
//          // plot sqrt(x) between 10 and 20 using 500 bins
//    tree.Draw("sqrt(x):sin(y)>>hsqrt(100,10,60,50,.1,.5)"
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
//     Making a Profile histogram
//     ==========================
//  In case of a 2-Dim expression, one can generate a TProfile histogram
//  instead of a TH2F histogram by specyfying option=prof or option=profs.
//  The option=prof is automatically selected in case of y:x>>pf
//  where pf is an existing TProfile histogram.
//
//     Making a 2D Profile histogram
//     ==========================
//  In case of a 3-Dim expression, one can generate a TProfile2D histogram
//  instead of a TH3F histogram by specyfying option=prof or option=profs.
//  The option=prof is automatically selected in case of z:y:x>>pf
//  where pf is an existing TProfile2D histogram.
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
//  If arrays are used in the selection critera, the event entered in the
//  list are all the event that have at least one element of the array that
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
//    -GetV1()              //returns a pointer to the float array of V1
//    -GetV2()              //returns a pointer to the float array of V2
//    -GetV3()              //returns a pointer to the float array of V3
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
   if (fTree->GetEntriesFriend() == 0) return 0;
   TString  opt;
   char *hdefault = (char *)"htemp";
   char *varexp;
   Int_t i,j,hkeep, action;
   opt = option;
   opt.ToLower();
   TH1 *oldh1 = 0;
   TEventList *elist = 0;
   char htitle[2560]; htitle[0] = '\0';
   Bool_t profile = kFALSE;

   TCut realSelection(selection);
   TEventList *inElist = fTree->GetEventList();
   Bool_t cleanElist = kFALSE;
   if ( inElist && inElist->GetReapplyCut() ) {
      realSelection = realSelection && inElist->GetTitle();
   }

// what each variable should contain:
//   varexp0   - original expression eg "a:b>>htest"
//   hname     - name of new or old histogram
//   hkeep     - flag if to keep new histogram
//   hnameplus - flag if to add to current histo
//   i         - length of variable expression stipped of everything after ">>"
//   varexp    - variable expression stipped of everything after ">>"
//   oldh1     - pointer to hist hname
//   elist     - pointer to selection list of hname

   Bool_t CanRebin = kTRUE;
   if (opt.Contains("same")) CanRebin = kFALSE;
  
   Int_t nbinsx=0, nbinsy=0, nbinsz=0;
   Double_t xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0;
   
   fHistogram = 0;
   char *hname = 0;
   for(UInt_t k=strlen(varexp0)-1;k>0;k--) {
      if (varexp0[k]=='>' && varexp0[k-1]=='>') {
         hname = (char*) &(varexp0[k-1]);
         break;
      }
   }
   //   char *hname = (char*)strstr(varexp0,">>");
   if (hname) {
      i = (int)( hname - varexp0 );  //  length of varexp0 before ">>"
      hname += 2;
      hkeep  = 1;
      varexp = new char[i+1];
      varexp[0] = 0; //necessary if i=0
      Bool_t hnameplus = kFALSE;
      while (*hname == ' ') hname++;
      if (*hname == '+') {
         hnameplus = kTRUE;
         hname++;
         while (*hname == ' ') hname++; //skip ' '
      }
      j = strlen(hname) - 1;   // skip ' '  at the end
      while (j) {
	if (hname[j] != ' ') break;
	hname[j] = 0;
	j--;
      }

      if (i) {
         strncpy(varexp,varexp0,i); varexp[i]=0;

	 Int_t mustdelete=0;

	 // parse things that follow the name of the histo between '(' and ')'.
	 // At this point hname contains the name of the specified histogram.
	 //   Now the syntax is exended to handle an hname of the following format
	 //   hname(nBIN [[,[xlow]][,xhigh]],...)
	 //   so enclosed in brackets is the binning information, xlow, xhigh, and
	 //   the same for the other dimensions

	 char *pstart;    // pointer to '('
	 char *pend;      // pointer to ')'
	 char *cdummy;    // dummy pointer
	 int ncomma;      // number of commas between '(' and ')', later number of arguments
	 int ncols;       // number of columns in varexpr
	 Double_t value;  // parsed value (by sscanf)

	 const Int_t maxvalues=9;

	 pstart= strchr(hname,'(');
	 pend =  strchr(hname,')');
	 if (pstart != NULL ){  // found the bracket

	   mustdelete=1;

	   // check that there is only one open and close bracket
	   if (pstart == strrchr(hname,'(')  &&  pend == strrchr(hname,')')) {

	     // count number of ',' between '(' and ')'
	     ncomma=0;
	     cdummy = pstart;
	     cdummy = strchr(&cdummy[1],',');
	     while (cdummy != NULL) {
	       cdummy = strchr(&cdummy[1],',');
	       ncomma++;
	     }

	     if (ncomma+1 > maxvalues) {
	       Error("DrawSelect","ncomma+1>maxvalues, ncomma=%d, maxvalues=%d",ncomma,maxvalues);
	       ncomma=maxvalues-1;
	     }

	     ncomma++; 	       // number of arguments
	     cdummy = pstart;
             
	     //   number of columns
	     ncols  = 1;
	     for (j=0;j<i;j++)  if (varexp[j] == ':') ncols++;
	     if (ncols > 3 ) {  // max 3 columns
	       Error("DrawSelect","ncols > 3, ncols=%d",ncols);
	       ncols = 0;
	     }

	     // check dimensions before and after ">>"
	     if (ncols*3 < ncomma) {
	       Error("DrawSelect","ncols*3 < ncomma ncols=%d, ncomma=%d",ncols,ncomma);
	       ncomma = ncols*3;
	     }

	     // scan the values one after the other
	     for (j=0;j<ncomma;j++) {
	       cdummy++;  // skip '(' or ','
	       if (sscanf(cdummy," %lf ",&value) == 1) {
                  cdummy=strchr(&cdummy[1],',');

		  switch (j) {  // do certain settings depending on position of argument
		  case 0:  // binning x-axis
                     nbinsx = (Int_t)value;
		     if      (ncols<2) {
                        gEnv->SetValue("Hist.Binning.1D.x",nbinsx);
                     } else if (ncols<3) {
		        gEnv->SetValue("Hist.Binning.2D.x",nbinsx);
		        gEnv->SetValue("Hist.Binning.2D.Prof",nbinsx);
		     } else {
		        gEnv->SetValue("Hist.Binning.3D.x",nbinsx);
		        gEnv->SetValue("Hist.Binning.3D.Profx",nbinsx);
		     }

		     break;
		  case 1:  // lower limit x-axis
                     xmin = value;
		     break;
		  case 2:  // upper limit x-axis
                     xmax = value;
		     break;
		  case 3:  // binning y-axis
                     nbinsy = (Int_t)value;
		     if (ncols<3) gEnv->SetValue("Hist.Binning.2D.y",nbinsy);
		     else {
		       gEnv->SetValue("Hist.Binning.3D.y",nbinsy);
		       gEnv->SetValue("Hist.Binning.3D.Profy",nbinsy);
		     }
		     break;
		  case 4:  // lower limit y-axis
                     ymin = value;
		     break;
		  case 5:  // upper limit y-axis
                     ymax = value;
		     break;
		  case 6:  // binning z-axis
                     nbinsz = (Int_t)value;
		     gEnv->SetValue("Hist.Binning.3D.z",nbinsz);
		     break;
		  case 7:  // lower limit z-axis
                     zmin = value;
		     break;
		  case 8:  // upper limit z-axis
                     zmax = value;
		     break;
		  default:
		     Error("DrawSelect","j>8");
		     break;
		  }
	       }  // if sscanf == 1
	     } // for j=0;j<ncomma;j++
	   } else {
	     Error("DrawSelect","Two open or close brackets found, hname=%s",hname);
	   }

	   // fix up hname
	   pstart[0]='\0';   // removes things after (and including) '('
	 } // if '(' is found

	 j = strlen(hname) - 1; // skip ' '  at the end
	 while (j) {
	   if (hname[j] != ' ') break; // skip ' '  at the end
	   hname[j] = 0;
	   j--;
	 }

         oldh1 = (TH1*)gDirectory->Get(hname);  // if hname contains '(...)' the return values is NULL, which is what we want
         if (oldh1 && !hnameplus) oldh1->Reset();  // reset unless adding is wanted

	 if (mustdelete) {
	   if (gDebug) {
              Warning("Draw","Deleting old histogram, since (possibly new) limits and binnings have been given");
           }
           delete oldh1; oldh1=0;
	 }

      } else { // if (i)                       // make selection list (i.e. varexp0 starts with ">>")
         elist = (TEventList*)gDirectory->Get(hname);
         if (!elist) {
            elist = new TEventList(hname,realSelection.GetTitle(),1000,0);
         }
         if (elist) {
            if (!hnameplus) {
               if (elist==inElist) {
                  // We have been asked to reset the input list!!
                  // Let's set it aside for now ...
                  inElist = new TEventList(*elist);
                  cleanElist = kTRUE;
                  fTree->SetEventList(inElist);
               }
               elist->Reset();
               elist->SetTitle(realSelection.GetTitle());
            } else {
               TCut old = elist->GetTitle();
               TCut upd = old || realSelection.GetTitle();
               elist->SetTitle(upd.GetTitle());
            }
         }
      }  // if (i)
   } else { // if (hname)
      hname  = hdefault;
      hkeep  = 0;
      varexp = (char*)varexp0;
      if (gDirectory) {
         oldh1 = (TH1*)gDirectory->Get(hname);
         if (oldh1) { oldh1->Delete(); oldh1 = 0;}
      }
   }
//*-* Do not process more than fMaxEntryLoop entries
   if (nentries > fTree->GetMaxEntryLoop()) nentries = fTree->GetMaxEntryLoop();

//*-*- Decode varexp and selection

   CompileVariables(varexp, realSelection.GetTitle());
   if (!fVar1 && !elist) return -1;

//*-*- In case oldh1 exists, check dimensionality
   Int_t nsel = strlen(selection);
   if (nsel > 1) {
      sprintf(htitle,"%s {%s}",varexp,selection);
   } else {
      sprintf(htitle,"%s",varexp);
   }
   if (oldh1) {
      Int_t olddim = oldh1->GetDimension();
      Int_t mustdelete = 0;
      if (oldh1->InheritsFrom("TProfile")) {
         profile = kTRUE;
         olddim = 2;
      }
      if (oldh1->InheritsFrom("TProfile2D")) {
         profile = kTRUE;
         olddim = 3;
      }
      if (opt.Contains("prof") && fDimension>1) {
        // ignore "prof" for 1D.
         if (!profile || olddim != fDimension) mustdelete = 1;
      } else {
         if (olddim != fDimension) mustdelete = 1;
      }
      if (mustdelete) {
         Warning("Draw","Deleting old histogram with different dimensions");
         delete oldh1; oldh1 = 0;
      }
   }

//*-*- Create a default canvas if none exists
   fDraw = 0;
   if (!gPad && !opt.Contains("goff") && fDimension > 0) {
      if (!gROOT->GetMakeDefCanvas()) return -1;
      (gROOT->GetMakeDefCanvas())();
   }

//*-*- 1-D distribution
   if (fDimension == 1) {
      action = 1;
      if (!oldh1) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.1D.x",100);
         if (gPad && opt.Contains("same")) {
            TListIter np(gPad->GetListOfPrimitives());
            TObject *op;
            TH1 *oldhtemp = 0;
            while ((op = np()) && !oldhtemp) {
              if (op->InheritsFrom("TH1")) oldhtemp = (TH1 *)op;
            }
            if (oldhtemp) {
                fNbins[0] = oldhtemp->GetXaxis()->GetNbins();
                fVmin[0]  = oldhtemp->GetXaxis()->GetXmin();
                fVmax[0]  = oldhtemp->GetXaxis()->GetXmax();
             } else {
                fVmin[0]  = gPad->GetUxmin();
                fVmax[0]  = gPad->GetUxmax();
             }
         } else {
             action   = -1;
             fVmin[0] = xmin;
             fVmax[0] = xmax;
             if (xmin < xmax) CanRebin = kFALSE;
         }
      }
      TH1F *h1;
      if (oldh1) {
         h1 = (TH1F*)oldh1;
         fNbins[0] = h1->GetXaxis()->GetNbins();
      } else {
         h1 = new TH1F(hname,htitle,fNbins[0],fVmin[0],fVmax[0]);
         h1->SetLineColor(fTree->GetLineColor());
         h1->SetLineWidth(fTree->GetLineWidth());
         h1->SetLineStyle(fTree->GetLineStyle());
         h1->SetFillColor(fTree->GetFillColor());
         h1->SetFillStyle(fTree->GetFillStyle());
         h1->SetMarkerStyle(fTree->GetMarkerStyle());
         h1->SetMarkerColor(fTree->GetMarkerColor());
         h1->SetMarkerSize(fTree->GetMarkerSize());
         //if (action == -1) h1->SetBuffer(fTree->GetEstimate());
         if (CanRebin)h1->SetBit(TH1::kCanRebin);
         if (!hkeep) {
            h1->SetBit(kCanDelete);
            if (!opt.Contains("goff")) h1->SetDirectory(0);
         }
         if (opt.Length() && opt[0] == 'e') h1->Sumw2();
      }
      fVar1->SetAxis(h1->GetXaxis());


      EntryLoop(action, h1, nentries, firstentry, option);

      if (fVar1->IsInteger()) h1->LabelsDeflate("X");
      if (!fDraw && !opt.Contains("goff")) h1->Draw(option);

//*-*- 2-D distribution
   } else if (fDimension == 2) {
      action = 2;
     // if (!opt.Contains("same") && !opt.Contains("goff") && gPad)  gPad->Clear();
      if (!oldh1 || !opt.Contains("same")) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.2D.y",40);
         fNbins[1] = gEnv->GetValue("Hist.Binning.2D.x",40);
         if (opt.Contains("prof")) fNbins[1] = gEnv->GetValue("Hist.Binning.2D.Prof",100);
         if (opt.Contains("same")) {
             TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
             if (oldhtemp) {
                fNbins[1] = oldhtemp->GetXaxis()->GetNbins();
                fVmin[1]  = oldhtemp->GetXaxis()->GetXmin();
                fVmax[1]  = oldhtemp->GetXaxis()->GetXmax();
                fNbins[0] = oldhtemp->GetYaxis()->GetNbins();
                fVmin[0]  = oldhtemp->GetYaxis()->GetXmin();
                fVmax[0]  = oldhtemp->GetYaxis()->GetXmax();
             } else {
                fNbins[1] = gEnv->GetValue("Hist.Binning.2D.x",40);
                fVmin[1]  = gPad->GetUxmin();
                fVmax[1]  = gPad->GetUxmax();
                fNbins[0] = gEnv->GetValue("Hist.Binning.2D.y",40);
                fVmin[0]  = gPad->GetUymin();
                fVmax[0]  = gPad->GetUymax();
             }
         } else {
             if (!oldh1) action = -2;
             fVmin[1] = xmin;
             fVmax[1] = xmax;
             fVmin[0] = ymin;
             fVmax[0] = ymax;
             if (xmin < xmax && ymin < ymax) CanRebin = kFALSE;
         }
      }
      if (profile || opt.Contains("prof")) {
         TProfile *hp;
         if (oldh1) {
            action = 4;
            hp = (TProfile*)oldh1;
         } else {
            if (action < 0) {
               action = -4;
               fVmin[1] = xmin;
               fVmax[1] = xmax;
               if (xmin < xmax) CanRebin = kFALSE;
            }
            if (opt.Contains("profs"))
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"s");
            else
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"");
            if (!hkeep) {
               hp->SetBit(kCanDelete);
               if (!opt.Contains("goff")) hp->SetDirectory(0);
            }
            hp->SetLineColor(fTree->GetLineColor());
            hp->SetLineWidth(fTree->GetLineWidth());
            hp->SetLineStyle(fTree->GetLineStyle());
            hp->SetFillColor(fTree->GetFillColor());
            hp->SetFillStyle(fTree->GetFillStyle());
            hp->SetMarkerStyle(fTree->GetMarkerStyle());
            hp->SetMarkerColor(fTree->GetMarkerColor());
            hp->SetMarkerSize(fTree->GetMarkerSize());
            if (CanRebin)hp->SetBit(TH1::kCanRebin);
         }
         fVar2->SetAxis(hp->GetXaxis());

         EntryLoop(action,hp,nentries, firstentry, option);

         if (fVar2->IsInteger()) hp->LabelsDeflate("X");
         if (!fDraw && !opt.Contains("goff")) hp->Draw(option);
      } else {
         TH2F *h2;
         if (oldh1) {
            h2 = (TH2F*)oldh1;
         } else {
            h2 = new TH2F(hname,htitle,fNbins[1],fVmin[1], fVmax[1], fNbins[0], fVmin[0], fVmax[0]);
            h2->SetLineColor(fTree->GetLineColor());
            h2->SetFillColor(fTree->GetFillColor());
            h2->SetFillStyle(fTree->GetFillStyle());
            h2->SetMarkerStyle(fTree->GetMarkerStyle());
            h2->SetMarkerColor(fTree->GetMarkerColor());
            h2->SetMarkerSize(fTree->GetMarkerSize());
            if (CanRebin)h2->SetBit(TH1::kCanRebin);
            if (!hkeep) {
               h2->SetBit(TH1::kNoStats);
               h2->SetBit(kCanDelete);
               if (!opt.Contains("goff")) h2->SetDirectory(0);
            }
         }
         fVar1->SetAxis(h2->GetYaxis());
         fVar2->SetAxis(h2->GetXaxis());
         Int_t noscat = strlen(option);
         if (opt.Contains("same")) noscat -= 4;
         if (noscat) {
            EntryLoop(action,h2,nentries, firstentry, option);
            if (fVar1->IsInteger()) h2->LabelsDeflate("Y");
            if (fVar2->IsInteger()) h2->LabelsDeflate("X");
            if (!fDraw && !opt.Contains("goff")) h2->Draw(option);
         } else {
            action = 12;
            if (!oldh1 && !opt.Contains("same")) action = -12;
            EntryLoop(action,h2,nentries, firstentry, option);
            if (fVar1->IsInteger()) h2->LabelsDeflate("Y");
            if (fVar2->IsInteger()) h2->LabelsDeflate("X");
            if (oldh1 && !fDraw && !opt.Contains("goff")) h2->Draw(option);
         }
      }

//*-*- 3-D distribution
   } else if (fDimension == 3) {
      action = 3;
     // if (!opt.Contains("same") && !opt.Contains("goff") && gPad)  gPad->Clear();
      if (!oldh1 || !opt.Contains("same")) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.3D.z",20);
         fNbins[1] = gEnv->GetValue("Hist.Binning.3D.y",20);
         fNbins[2] = gEnv->GetValue("Hist.Binning.3D.x",20);
         if (opt.Contains("same")) {
             TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
             if (oldhtemp) {
                fNbins[2] = oldhtemp->GetXaxis()->GetNbins();
                fVmin[2]  = oldhtemp->GetXaxis()->GetXmin();
                fVmax[2]  = oldhtemp->GetXaxis()->GetXmax();
                fNbins[1] = oldhtemp->GetYaxis()->GetNbins();
                fVmin[1]  = oldhtemp->GetYaxis()->GetXmin();
                fVmax[1]  = oldhtemp->GetYaxis()->GetXmax();
                fNbins[0] = oldhtemp->GetZaxis()->GetNbins();
                fVmin[0]  = oldhtemp->GetZaxis()->GetXmin();
                fVmax[0]  = oldhtemp->GetZaxis()->GetXmax();
             } else {
                TView *view = gPad->GetView();
                Double_t *rmin = view->GetRmin();
                Double_t *rmax = view->GetRmax();
                fNbins[2] = gEnv->GetValue("Hist.Binning.3D.z",20);
                fVmin[2]  = rmin[0];
                fVmax[2]  = rmax[0];
                fNbins[1] = gEnv->GetValue("Hist.Binning.3D.y",20);
                fVmin[1]  = rmin[1];
                fVmax[1]  = rmax[1];
                fNbins[0] = gEnv->GetValue("Hist.Binning.3D.x",20);
                fVmin[0]  = rmin[2];
                fVmax[0]  = rmax[2];
             }
         } else {
             if (!oldh1) action = -3;
             fVmin[2] = xmin;
             fVmax[2] = xmax;
             fVmin[1] = ymin;
             fVmax[1] = ymax;
             fVmin[0] = zmin;
             fVmax[0] = zmax;
             if (xmin < xmax && ymin < ymax && zmin < zmax) CanRebin = kFALSE;
         }
      }
      if (profile || opt.Contains("prof")) {
         TProfile2D *hp;
         if (oldh1) {
            action = 23;
            hp = (TProfile2D*)oldh1;
         } else {
            if (action < 0) {
               action = -23;
               fVmin[2] = xmin;
               fVmax[2] = xmax;
               fVmin[1] = ymin;
               fVmax[1] = ymax;
             if (xmin < xmax && ymin < ymax) CanRebin = kFALSE;
            }
            if (opt.Contains("profs"))
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"s");
            else
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"");
            if (!hkeep) {
               hp->SetBit(kCanDelete);
               if (!opt.Contains("goff")) hp->SetDirectory(0);
            }
            hp->SetLineColor(fTree->GetLineColor());
            hp->SetLineWidth(fTree->GetLineWidth());
            hp->SetLineStyle(fTree->GetLineStyle());
            hp->SetFillColor(fTree->GetFillColor());
            hp->SetFillStyle(fTree->GetFillStyle());
            hp->SetMarkerStyle(fTree->GetMarkerStyle());
            hp->SetMarkerColor(fTree->GetMarkerColor());
            hp->SetMarkerSize(fTree->GetMarkerSize());
            if (CanRebin)hp->SetBit(TH1::kCanRebin);
         }
         fVar2->SetAxis(hp->GetYaxis());
         fVar3->SetAxis(hp->GetXaxis());

         EntryLoop(action,hp,nentries, firstentry, option);

         if (fVar2->IsInteger()) hp->LabelsDeflate("Y");
         if (fVar3->IsInteger()) hp->LabelsDeflate("X");
         if (!fDraw && !opt.Contains("goff")) hp->Draw(option);

      } else {
         TH3F *h3;
         if (oldh1) {
            h3 = (TH3F*)oldh1;
         } else {
            h3 = new TH3F(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1], fNbins[0], fVmin[0], fVmax[0]);
            h3->SetLineColor(fTree->GetLineColor());
            h3->SetFillColor(fTree->GetFillColor());
            h3->SetFillStyle(fTree->GetFillStyle());
            h3->SetMarkerStyle(fTree->GetMarkerStyle());
            h3->SetMarkerColor(fTree->GetMarkerColor());
            h3->SetMarkerSize(fTree->GetMarkerSize());
            if (CanRebin)h3->SetBit(TH1::kCanRebin);
            if (!hkeep) {
               h3->SetBit(kCanDelete);
               h3->SetBit(TH1::kNoStats);
               if (!opt.Contains("goff")) h3->SetDirectory(0);
            }
         }
         fVar1->SetAxis(h3->GetZaxis());
         fVar2->SetAxis(h3->GetYaxis());
         fVar3->SetAxis(h3->GetXaxis());
         Int_t noscat = strlen(option);
         if (opt.Contains("same")) noscat -= 4;
         if (noscat) {
            EntryLoop(action,h3,nentries, firstentry, option);
            if (fVar1->IsInteger()) h3->LabelsDeflate("Z");
            if (fVar2->IsInteger()) h3->LabelsDeflate("Y");
            if (fVar3->IsInteger()) h3->LabelsDeflate("X");
            if (!fDraw && !opt.Contains("goff")) h3->Draw(option);
         } else {
            action = 13;
            if (!oldh1 && !opt.Contains("same")) action = -13;
            EntryLoop(action,h3,nentries, firstentry, option);
            if (fVar1->IsInteger()) h3->LabelsDeflate("Z");
            if (fVar2->IsInteger()) h3->LabelsDeflate("Y");
            if (fVar3->IsInteger()) h3->LabelsDeflate("X");
            if (oldh1 && !fDraw && !opt.Contains("goff")) h3->Draw(option);
         }
      }

//*-* an Event List
   } else {
      action = 5;
      Int_t oldEstimate = fTree->GetEstimate();
      fTree->SetEstimate(1);
      EntryLoop(action,elist,nentries, firstentry, option);
      fTree->SetEstimate(oldEstimate);
   }
   if (hkeep) delete [] varexp;
   if (cleanElist) {
     // We are in the case where the input list was reset!
     fTree->SetEventList(elist);
     delete inElist;
   }
   return fSelectedRows;
}

//______________________________________________________________________________
void TTreePlayer::EntryLoop(Int_t &action, TObject *obj, Int_t nentries, Int_t firstentry, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Loop on all entries*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==================
//
//  nentries is the number of entries to process (default is all)
//  first is the first entry to process (default is 0)
//
//  action =  1  Fill 1-D histogram obj
//         =  2  Fill 2-D histogram obj
//         =  3  Fill 3-D histogram obj
//         =  4  Fill Profile histogram obj
//         =  5  Fill a TEventlist
//         = 11  Estimate Limits
//         = 12  Fill 2-D PolyMarker obj
//         = 13  Fill 3-D PolyMarker obj
//  action < 0   Evaluate Limits for case abs(action)
//

   Int_t i,entry,entryNumber,localEntry, lastentry,ndata,nfill0;
   Double_t ww;
   Int_t  npoints;
   lastentry = firstentry + nentries - 1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = (Int_t)fTree->GetEntriesFriend() - 1;
      nentries   = lastentry - firstentry + 1;
   }

   TDirectory *cursav = gDirectory;

   fNfill = 0;

   //Create a timer to get control in the entry loop(s)
   TProcessEventTimer *timer = 0;
   Int_t interval = fTree->GetTimerInterval();
   if (!gROOT->IsBatch() && interval)
      timer = new TProcessEventTimer(interval);

   Double_t treeWeight = fTree->GetWeight();
   npoints = 0;
   if (!fV1 && fVar1)   fV1 = new Double_t[fTree->GetEstimate()];
   if (!fV2 && fVar2)   fV2 = new Double_t[fTree->GetEstimate()];
   if (!fV3 && fVar3)   fV3 = new Double_t[fTree->GetEstimate()];
   if (!fW)             fW  = new Double_t[fTree->GetEstimate()];
   Int_t force = fTree->TestBit(TTree::kForceRead);
   if (!fMultiplicity) {
      for (entry=firstentry;entry<firstentry+nentries;entry++) {
         entryNumber = fTree->GetEntryNumber(entry);
         if (entryNumber < 0) break;
         if (timer && timer->ProcessEvents()) break;
         if (gROOT->IsInterrupted()) break;
         localEntry = fTree->LoadTree(entryNumber);
         if (localEntry < 0) break;
         if (fSelect) {
            if (force && fSelect->GetNdata()<=0) continue;
            fW[fNfill] = treeWeight*fSelect->EvalInstance(0);
            if (!fW[fNfill]) continue;
         } else fW[fNfill] = treeWeight;
         if (fVar1) {
            if (force && fVar1->GetNdata()<=0) continue;
            fV1[fNfill] = fVar1->EvalInstance(0);
         }
         if (fVar2) {
            if (force && fVar2->GetNdata()<=0) continue;
            fV2[fNfill] = fVar2->EvalInstance(0);
            if (fVar3) {
               if (force && fVar3->GetNdata()<=0) continue;
               fV3[fNfill] = fVar3->EvalInstance(0);
            }
         }
         fNfill++;
         if (fNfill >= fTree->GetEstimate()) {
            TakeAction(fNfill,npoints,action,obj,option);
            fNfill = 0;
         }
      }

      if (fNfill) {
         TakeAction(fNfill,npoints,action,obj,option);
      }
      fSelectedRows = npoints;
      if (npoints == 0) fDraw = 1; // do not draw
      delete timer;
      return;
   }

   Bool_t Var1Multiple = kFALSE;
   Bool_t Var2Multiple = kFALSE;
   Bool_t Var3Multiple = kFALSE;
   Bool_t SelectMultiple = kFALSE;
   if (fVar1 && fVar1->GetMultiplicity()) Var1Multiple = kTRUE;
   if (fVar2 && fVar2->GetMultiplicity()) Var2Multiple = kTRUE;
   if (fVar3 && fVar3->GetMultiplicity()) Var3Multiple = kTRUE;
   if (fSelect && fSelect->GetMultiplicity()) SelectMultiple = kTRUE;

   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      if (timer && timer->ProcessEvents()) break;
      if (gROOT->IsInterrupted()) break;
      localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      nfill0 = fNfill;

      // Look for the lowest common array size amongst the
      // variable and selection cut.
      ndata = fMultiplicity->GetNdata();
      if (Var1Multiple && (fMultiplicity!=fVar1))
        ndata = TMath::Min(ndata,fVar1->GetNdata());
      if (Var2Multiple && (fMultiplicity!=fVar2))
        ndata = TMath::Min(ndata,fVar2->GetNdata());
      if (Var3Multiple && (fMultiplicity!=fVar3))
        ndata = TMath::Min(ndata,fVar3->GetNdata());

      // no data at all, let's move on to the next entry.
      if (!ndata) continue;

      // Calculate the first values
      if (fSelect) {
         fW[fNfill] = treeWeight*fSelect->EvalInstance(0);
         if (!fW[fNfill]  && !SelectMultiple) continue;
      } else fW[fNfill] = treeWeight;
      if (fVar1) {
         fV1[fNfill] = fVar1->EvalInstance(0);
         if (fVar2) {
            fV2[fNfill] = fVar2->EvalInstance(0);
            if (fVar3) {
               fV3[fNfill] = fVar3->EvalInstance(0);
            }
         }
      }
      if (fW[fNfill]) {
         fNfill++;
         if (fNfill >= fTree->GetEstimate()) {
            TakeAction(fNfill,npoints,action,obj,option);
            fNfill = 0;
         }
      }
      ww = treeWeight;

      for (i=1;i<ndata;i++) {
         if (SelectMultiple) {
            ww = treeWeight*fSelect->EvalInstance(i);
            if (ww == 0) continue;
         }
         if (fVar1) {
            if (Var1Multiple) fV1[fNfill] = fVar1->EvalInstance(i);
            else              fV1[fNfill] = fV1[nfill0];
            if (fVar2) {
               if (Var2Multiple) fV2[fNfill] = fVar2->EvalInstance(i);
               else              fV2[fNfill] = fV2[nfill0];
               if (fVar3) {
                  if (Var3Multiple) fV3[fNfill] = fVar3->EvalInstance(i);
                  else              fV3[fNfill] = fV3[nfill0];
               }
            }
         }
         fW[fNfill] = ww;

         fNfill++;
         if (fNfill >= fTree->GetEstimate()) {
            TakeAction(fNfill,npoints,action,obj,option);
            fNfill = 0;
         }
      }
   }

   delete timer;

   if (fNfill) {
      TakeAction(fNfill,npoints,action,obj,option);
   }

   fSelectedRows = npoints;
   if (npoints == 0) fDraw = 1; // do not draw
   if (cursav) cursav->cd();
}

//______________________________________________________________________________
Int_t TTreePlayer::Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption,Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Fit  a projected item(s) from a Tree*-*-*-*-*-*-*-*-*-*
//*-*              ======================================
//
//  formula is a TF1 expression.
//
//  See TTree::Draw for explanations of the other parameters.
//
//  By default the temporary histogram created is called htemp.
//  If varexp contains >>hnew , the new histogram created is called hnew
//  and it is kept in the current directory.
//  Example:
//    tree.Fit("pol4","sqrt(x)>>hsqrt","y>0")
//    will fit sqrt(x) and save the histogram as "hsqrt" in the current
//    directory.
//

   Int_t nch = strlen(option) + 10;
   char *opt = new char[nch];
   if (option) sprintf(opt,"%sgoff",option);
   else        strcpy(opt,"goff");

   Int_t nsel = DrawSelect(varexp,selection,opt,nentries,firstentry);

   delete [] opt;

   if (fHistogram) {
      fHistogram->Fit(formula,option,goption);
   }
   return nsel;
}

//______________________________________________________________________________
const char *TTreePlayer::GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex)
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
void TTreePlayer::GetNextPacket(TSlave *sl, Int_t &nentries, Stat_t &firstentry, Stat_t &processed)
{
   // Return in nentries and firstentry the optimal range of entries (packet)
   // to be processed by slave sl. See TPacketGenerator for the algorithm
   // used to get the packet size.

   fPacketGen->GetNextPacket(sl, nentries, firstentry);
   processed = fPacketGen->GetEntriesProcessed();
}

//______________________________________________________________________________
void TTreePlayer::Loop(Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Loop on nentries of this tree starting at firstentry
//*-*              ===================================================

   Warning("Loop","Obsolete function");
}

//______________________________________________________________________________
Int_t TTreePlayer::MakeClass(const char *classname, const char *option)
{
//====>
//*-*-*-*-*-*-*Generate skeleton analysis class for this Tree*-*-*-*-*-*-*
//*-*          ==============================================
//
//   The following files are produced: classname.h and classname.C
//   if classname is NULL, classname will be nameoftree.
//
//   When the option "selector" is specified, the function generates the
//   selector class described in TTree::MakeSelector.
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

   TString opt = option;
   opt.ToLower();

   // Connect output files
   char *thead = new char[256];

   if (!classname) classname = fTree->GetName();
   sprintf(thead,"%s.h",classname);
   FILE *fp = fopen(thead,"w");
   if (!fp) {
      printf("Cannot open output file:%s\n",thead);
      delete [] thead;
      return 3;
   }
   char *tcimp = new char[256];
   sprintf(tcimp,"%s.C",classname);
   FILE *fpc = fopen(tcimp,"w");
   if (!fpc) {
      printf("Cannot open output file:%s\n",tcimp);
      delete [] thead;
      delete [] tcimp;
      return 3;
   }
   char *treefile = new char[1000];
   if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile())
                strcpy(treefile,fTree->GetDirectory()->GetFile()->GetName());
   else         strcpy(treefile,"Memory Directory");
   // In the case of a chain, the GetDirectory information usually does
   // pertain to the Chain itself but to the currently loaded tree.
   // So we can not rely on it.
   Bool_t ischain = fTree->InheritsFrom("TChain");

//======================Generate classname.h=====================
   // Print header
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;
   TDatime td;
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"//   This class has been automatically generated \n");
   fprintf(fp,"//     (%s by ROOT version%s)\n",td.AsString(),gROOT->GetVersion());
   if (!ischain) {
      fprintf(fp,"//   from TTree %s/%s\n",fTree->GetName(),fTree->GetTitle());
      fprintf(fp,"//   found on file: %s\n",treefile);
   } else {
      fprintf(fp,"//   from TChain %s/%s\n",fTree->GetName(),fTree->GetTitle());
   }
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"\n");
   fprintf(fp,"\n");
   fprintf(fp,"#ifndef %s_h\n",classname);
   fprintf(fp,"#define %s_h\n",classname);
   fprintf(fp,"\n");
   fprintf(fp,"#include <TROOT.h>\n");
   fprintf(fp,"#include <TChain.h>\n");
   fprintf(fp,"#include <TFile.h>\n");
   if (opt.Contains("selector")) fprintf(fp,"#include <TSelector.h>\n");

// First loop on all leaves to generate dimension declarations
   Int_t len, lenb,l;
   char blen[128];
   char *bname;
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      strcpy(blen,leaf->GetName());
      bname = &blen[0];
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      lenb = strlen(blen);
      if (blen[lenb-1] == '_') {
         blen[lenb-1] = 0;
         len = leaf->GetMaximum();
         if (len <= 0) len = 1;
         fprintf(fp,"   const Int_t kMax%s = %d;\n",blen,len);
      }
   }


// second loop on all leaves to generate type declarations
   fprintf(fp,"\n");
   if (opt.Contains("selector")) {
      fprintf(fp,"class %s : public TSelector {\n",classname);
      fprintf(fp,"   public :\n");
      fprintf(fp,"   TTree          *fChain;   //!pointer to the analyzed TTree or TChain\n");
   } else {
      fprintf(fp,"class %s {\n",classname);
      fprintf(fp,"   public :\n");
      fprintf(fp,"   TTree          *fChain;   //!pointer to the analyzed TTree or TChain\n");
      fprintf(fp,"   Int_t           fCurrent; //!current Tree number in a TChain\n");
   }
   fprintf(fp,"//Declaration of leaves types\n");
   TLeaf *leafcount;
   TLeafObject *leafobj;
   TBranchElement *bre;
   const char *headOK  = "   ";
   const char *headcom = " //";
   const char *head;
   char branchname[128];
   char aprefix[128];
   TObjArray branches(100);
   TObjArray mustInit(100);
   Int_t *leafStatus = new Int_t[nleaves];
   for (l=0;l<nleaves;l++) {
      Int_t kmax = 0;
      head = headOK;
      leafStatus[l] = 0;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      branchname[0] = 0;
      strcpy(branchname,branch->GetName());
      strcpy(aprefix,branch->GetName());
      if (!branches.FindObject(branch)) branches.Add(branch);
      else leafStatus[l] = 1;
      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strcat(branchname,".");
         strcat(branchname,leaf->GetTitle());
         if (leafcount) {
           // remove any dimension in title
           char *dim =  (char*)strstr(branchname,"["); if (dim) dim[0] = 0;
         }
      } else {
         if (leafcount) strcpy(branchname,branch->GetName());
         else           strcpy(branchname,leaf->GetTitle());
      }
      char *twodim = (char*)strstr(leaf->GetTitle(),"][");
      bname = branchname;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      if (branch->IsA() == TBranchObject::Class()) {
         if (branch->GetListOfBranches()->GetEntriesFast()) {leafStatus[l] = 1; continue;}
         leafobj = (TLeafObject*)leaf;
         if (!leafobj->GetClass()) {leafStatus[l] = 1; head = headcom;}
         fprintf(fp,"%s%-15s *%s;\n",head,leafobj->GetTypeName(), leafobj->GetName());
         mustInit.Add(leafobj);
         continue;
      }
      if (leafcount) {
         len = leafcount->GetMaximum();
         strcpy(blen,leafcount->GetName());
         bname = &blen[0];
         while (*bname) {if (*bname == '.') *bname='_'; bname++;}
         lenb = strlen(blen);
         if (blen[lenb-1] == '_') {blen[lenb-1] = 0; kmax = 1;}
         else                     sprintf(blen,"%d",len);
      }
      if (branch->IsA() == TBranchElement::Class()) {
         bre = (TBranchElement*)branch;
         if (bre->GetType() != 3 && bre->GetStreamerType() <= 0 && bre->GetListOfBranches()->GetEntriesFast()) leafStatus[l] = 0;
         if (bre->GetType() == 3) {
            fprintf(fp,"   %-15s %s;\n","Int_t", branchname);
            continue;
         }
         if (branch->GetListOfBranches()->GetEntriesFast()) {leafStatus[l] = 1; continue;}
         if (bre->GetStreamerType() <= 0) {
            if (!gROOT->GetClass(bre->GetClassName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
            fprintf(fp,"%s%-15s *%s;\n",head,bre->GetClassName(), bre->GetName());
            mustInit.Add(bre);
            continue;
         }
         if (bre->GetStreamerType() > 60) {
            TClass *cle = gROOT->GetClass(bre->GetClassName());
            if (!cle) {leafStatus[l] = 1; continue;}
            if (bre->GetStreamerType() == 66) leafStatus[l] = 0;
            char brename[256];
            strcpy(brename,bre->GetName());
            char *bren = brename;
            char *adot = strrchr(bren,'.');
            if (adot) bren = adot+1;
            char *brack = strchr(bren,'[');
            if (brack) *brack = 0;
            TStreamerElement *elem = (TStreamerElement*)cle->GetStreamerInfo()->GetElements()->FindObject(bren);
            if (elem) {
               if (elem->IsA() == TStreamerBase::Class()) {leafStatus[l] = 1; continue;}
               if (!gROOT->GetClass(elem->GetTypeName())) {leafStatus[l] = 1; continue;}
               if (!gROOT->GetClass(elem->GetTypeName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
               if (leafcount) fprintf(fp,"%s%-15s %s[kMax%s];\n",head,elem->GetTypeName(), branchname,blen);
               else           fprintf(fp,"%s%-15s %s;\n",head,elem->GetTypeName(), branchname);
            } else {
               if (!gROOT->GetClass(bre->GetClassName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
               fprintf(fp,"%s%-15s %s;\n",head,bre->GetClassName(), branchname);
            }
            continue;
         }
       }
     if (strlen(leaf->GetTypeName()) == 0) {leafStatus[l] = 1; continue;}
      if (leafcount) {
         //len = leafcount->GetMaximum();
         //strcpy(blen,leafcount->GetName());
         //bname = &blen[0];
         //while (*bname) {if (*bname == '.') *bname='_'; bname++;}
         //lenb = strlen(blen);
         //Int_t kmax = 0;
         //if (blen[lenb-1] == '_') {blen[lenb-1] = 0; kmax = 1;}
         //else                     sprintf(blen,"%d",len);
	 // Dimensions can be in the branchname for a split Object with a fix length C array.
	 // Theses dimensions HAVE TO be placed after the dimension explicited by leafcount
	 char *dimensions = 0;
         char *dimInName = (char*) strstr(branchname,"[");
	 if ( twodim || dimInName ) {
	   int dimlen = 0;
	   if (dimInName) dimlen += strlen(dimInName) + 1;
	   if (twodim)    dimlen += strlen(twodim) + 1;
	   dimensions = new char[dimlen];
	   if (dimInName) {
	     strcpy(dimensions,dimInName);
	     dimInName[0] = 0; // terminate branchname before the array dimensions.
	   } else dimensions[0] = 0;
	   if (twodim) strcat(dimensions,(char*)(twodim+1));
	 }
         if (dimensions) {
            if (kmax) fprintf(fp,"   %-15s %s[kMax%s]%s;   //[%s]\n",leaf->GetTypeName(),
			                                    branchname,blen,dimensions,leafcount->GetName());
	    else      fprintf(fp,"   %-15s %s[%d]%s;   //[%s]\n",leaf->GetTypeName(),
			                                branchname,len,dimensions,leafcount->GetName());
	    delete dimensions;
         } else {
            if (kmax) fprintf(fp,"   %-15s %s[kMax%s];   //[%s]\n",leaf->GetTypeName(), branchname,blen,leafcount->GetName());
            else      fprintf(fp,"   %-15s %s[%d];   //[%s]\n",leaf->GetTypeName(), branchname,len,leafcount->GetName());
         }
      } else {
         if (strstr(branchname,"[")) len = 1;
         if (len < 2) fprintf(fp,"   %-15s %s;\n",leaf->GetTypeName(), branchname);
         else {
            if (twodim) fprintf(fp,"   %-15s %s%s;\n",leaf->GetTypeName(), branchname,(char*)strstr(leaf->GetTitle(),"["));
            else        fprintf(fp,"   %-15s %s[%d];\n",leaf->GetTypeName(), branchname,len);
         }
      }
  }

// generate list of branches
   fprintf(fp,"\n");
   fprintf(fp,"//List of branches\n");
   for (l=0;l<nleaves;l++) {
      if (leafStatus[l]) continue;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      //if (strlen(leaf->GetTypeName()) == 0) continue;
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      strcpy(branchname,branch->GetName());
      if ( branch->GetNleaves() <= 1 ) {
         if (branch->IsA() != TBranchObject::Class()) {
            if (!leafcount) strcpy(branchname,leaf->GetTitle());
         }
      }
      bname = branchname;
      char *twodim = (char*)strstr(bname,"[");
      if (twodim) *twodim = 0;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      fprintf(fp,"   TBranch        *b_%s;   //!\n",branchname);
   }

// generate class member functions prototypes
   if (opt.Contains("selector")) {
      fprintf(fp,"\n");
      fprintf(fp,"   %s(TTree *tree=0) { }\n",classname) ;
      fprintf(fp,"   ~%s() { }\n",classname);
      fprintf(fp,"   void    Begin(TTree *tree);\n");
      fprintf(fp,"   void    Init(TTree *tree);\n");
      fprintf(fp,"   Bool_t  Notify();\n");
      fprintf(fp,"   Bool_t  ProcessCut(Int_t entry);\n");
      fprintf(fp,"   void    ProcessFill(Int_t entry);\n");
      fprintf(fp,"   void    Terminate();\n");
      fprintf(fp,"};\n");
      fprintf(fp,"\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");
   } else {
      fprintf(fp,"\n");
      fprintf(fp,"   %s(TTree *tree=0);\n",classname);
      fprintf(fp,"   ~%s();\n",classname);
      fprintf(fp,"   Int_t  Cut(Int_t entry);\n");
      fprintf(fp,"   Int_t  GetEntry(Int_t entry);\n");
      fprintf(fp,"   Int_t  LoadTree(Int_t entry);\n");
      fprintf(fp,"   void   Init(TTree *tree);\n");
      fprintf(fp,"   void   Loop();\n");
      fprintf(fp,"   Bool_t Notify();\n");
      fprintf(fp,"   void   Show(Int_t entry = -1);\n");
      fprintf(fp,"};\n");
      fprintf(fp,"\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");
   }
// generate code for class constructor
   fprintf(fp,"#ifdef %s_cxx\n",classname);
   if (!opt.Contains("selector")) {
      fprintf(fp,"%s::%s(TTree *tree)\n",classname,classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// if parameter tree is not specified (or zero), connect the file\n");
      fprintf(fp,"// used to generate this class and read the Tree.\n");
      fprintf(fp,"   if (tree == 0) {\n");
      if (ischain) {
        fprintf(fp,"\n#ifdef SINGLE_TREE\n");
        fprintf(fp,"      // The following code should be used if you want this class to access\n");
        fprintf(fp,"      // a single tree instead of a chain\n");
      }
      fprintf(fp,"      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(\"%s\");\n",treefile);
      fprintf(fp,"      if (!f) {\n");
      fprintf(fp,"         f = new TFile(\"%s\");\n",treefile);
      if (gDirectory != gFile) {
        fprintf(fp,"         f->cd(\"%s\");\n",gDirectory->GetPath());
      }
      fprintf(fp,"      }\n");
      fprintf(fp,"      tree = (TTree*)gDirectory->Get(\"%s\");\n\n",fTree->GetName());
      if (ischain) {
         fprintf(fp,"#else // SINGLE_TREE\n\n");
         fprintf(fp,"      // The following code should be used if you want this class to access a chain\n");
         fprintf(fp,"      // of trees.\n");
         fprintf(fp,"      TChain * chain = new TChain(\"%s\",\"%s\");\n",
                 fTree->GetName(),fTree->GetTitle());
         TIter next(((TChain*)fTree)->GetListOfFiles());
         TChainElement *element;
         while ((element = (TChainElement*)next())) {
            fprintf(fp,"      chain->Add(\"%s/%s\");\n",element->GetTitle(),element->GetName());
         }
         fprintf(fp,"      tree = chain;\n");
         fprintf(fp,"#endif // SINGLE_TREE\n\n");
      }
      fprintf(fp,"   }\n");
      fprintf(fp,"   Init(tree);\n");
      fprintf(fp,"}\n");
      fprintf(fp,"\n");
   }

// generate code for class destructor()
   if (!opt.Contains("selector")) {
      fprintf(fp,"%s::~%s()\n",classname,classname);
      fprintf(fp,"{\n");
      fprintf(fp,"   if (!fChain) return;\n");
      fprintf(fp,"   delete fChain->GetCurrentFile();\n");
      fprintf(fp,"}\n");
      fprintf(fp,"\n");
   }
// generate code for class member function GetEntry()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Int_t %s::GetEntry(Int_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Read contents of entry.\n");

      fprintf(fp,"   if (!fChain) return 0;\n");
      fprintf(fp,"   return fChain->GetEntry(entry);\n");
      fprintf(fp,"}\n");
   }
// generate code for class member function LoadTree()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Int_t %s::LoadTree(Int_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Set the environment to read one entry\n");
      fprintf(fp,"   if (!fChain) return -5;\n");
      fprintf(fp,"   Int_t centry = fChain->LoadTree(entry);\n");
      fprintf(fp,"   if (centry < 0) return centry;\n");
      fprintf(fp,"   if (fChain->IsA() != TChain::Class()) return centry;\n");
      fprintf(fp,"   TChain *chain = (TChain*)fChain;\n");
      fprintf(fp,"   if (chain->GetTreeNumber() != fCurrent) {\n");
      fprintf(fp,"      fCurrent = chain->GetTreeNumber();\n");
      fprintf(fp,"      Notify();\n");
      fprintf(fp,"   }\n");
      fprintf(fp,"   return centry;\n");
      fprintf(fp,"}\n");
      fprintf(fp,"\n");
   }

// generate code for class member function Init(), first pass = get branch pointer
   fprintf(fp,"void %s::Init(TTree *tree)\n",classname);
   fprintf(fp,"{\n");
   if (mustInit.Last()) {
      TIter next(&mustInit);
      TObject *obj;
      fprintf(fp,"//   Set object pointer\n");
      while( (obj = next()) ) {
         if (obj->InheritsFrom(TBranch::Class())) {
            fprintf(fp,"   %s = 0;\n",((TBranch*)obj)->GetName() );
         } else if (obj->InheritsFrom(TLeaf::Class())) {
            fprintf(fp,"   %s = 0;\n",((TLeaf*)obj)->GetName() );
         }
      }
   }
   fprintf(fp,"//   Set branch addresses\n");
   fprintf(fp,"   if (tree == 0) return;\n");
   fprintf(fp,"   fChain    = tree;\n");
   if (!opt.Contains("selector")) fprintf(fp,"   fCurrent = -1;\n");
   fprintf(fp,"   fChain->SetMakeClass(1);\n");
   fprintf(fp,"\n");
   for (l=0;l<nleaves;l++) {
      if (leafStatus[l]) continue;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      strcpy(aprefix,branch->GetName());

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strcpy(branchname,branch->GetName());
         strcat(branchname,".");
         strcat(branchname,leaf->GetTitle());
         if (leafcount) {
           // remove any dimension in title
           char *dim =  (char*)strstr(branchname,"["); if (dim) dim[0] = 0;
         }
      } else {
         if (leafcount) strcpy(branchname,branch->GetName());
         else           strcpy(branchname,leaf->GetTitle());
      }
      bname = branchname;
      char *brak = strstr(branchname,"[");     if (brak) *brak = 0;
      char *twodim = (char*)strstr(bname,"["); if (twodim) *twodim = 0;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      if (branch->IsA() == TBranchObject::Class()) {
         if (branch->GetListOfBranches()->GetEntriesFast()) {
            fprintf(fp,"   fChain->SetBranchAddress(\"%s\",(void*)-1);\n",branch->GetName());
            continue;
         }
         strcpy(branchname,branch->GetName());
      }
      if (branch->IsA() == TBranchElement::Class()) {
         if (((TBranchElement*)branch)->GetType() == 3) len =1;
      }
      if (leafcount) len = leafcount->GetMaximum()+1;
      if (len > 1) fprintf(fp,"   fChain->SetBranchAddress(\"%s\",%s);\n",branch->GetName(),branchname);
      else         fprintf(fp,"   fChain->SetBranchAddress(\"%s\",&%s);\n",branch->GetName(),branchname);
   }
   //must call Notify in case of MakeClass
   if (!opt.Contains("selector")) {
      fprintf(fp,"   Notify();\n");
   }

   fprintf(fp,"}\n");
   fprintf(fp,"\n");

// generate code for class member function Notify()
   fprintf(fp,"Bool_t %s::Notify()\n",classname);
   fprintf(fp,"{\n");
   fprintf(fp,"   // Called when loading a new file.\n");
   fprintf(fp,"   // Get branch pointers.\n");
   for (l=0;l<nleaves;l++) {
      if (leafStatus[l]) continue;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      strcpy(branchname,branch->GetName());
      if ( branch->GetNleaves() <= 1 ) {
         if (branch->IsA() != TBranchObject::Class()) {
            if (!leafcount) strcpy(branchname,leaf->GetTitle());
         }
      }
      bname = branchname;
      char *twodim = (char*)strstr(bname,"["); if (twodim) *twodim = 0;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      if (branch->IsA() == TBranchObject::Class()) {
         if (branch->GetListOfBranches()->GetEntriesFast()) {
            fprintf(fp,"   b_%s = fChain->GetBranch(\"%s\");\n",branchname,branch->GetName());
            continue;
         }
         strcpy(branchname,branch->GetName());
      }
      fprintf(fp,"   b_%s = fChain->GetBranch(\"%s\");\n",branchname,branch->GetName());
   }
   fprintf(fp,"   return kTRUE;\n");
   fprintf(fp,"}\n");
   fprintf(fp,"\n");

// generate code for class member function Show()
   if (!opt.Contains("selector")) {
      fprintf(fp,"void %s::Show(Int_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Print contents of entry.\n");
      fprintf(fp,"// If entry is not specified, print current entry\n");

      fprintf(fp,"   if (!fChain) return;\n");
      fprintf(fp,"   fChain->Show(entry);\n");
      fprintf(fp,"}\n");
   }
// generate code for class member function Cut()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Int_t %s::Cut(Int_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// This function may be called from Loop.\n");
      fprintf(fp,"// returns  1 if entry is accepted.\n");
      fprintf(fp,"// returns -1 otherwise.\n");

      fprintf(fp,"   return 1;\n");
      fprintf(fp,"}\n");
   }
   fprintf(fp,"#endif // #ifdef %s_cxx\n",classname);
   fprintf(fp,"\n");

//======================Generate classname.C=====================
   if (!opt.Contains("selector")) {
      // generate code for class member function Loop()
      fprintf(fpc,"#define %s_cxx\n",classname);
      fprintf(fpc,"#include \"%s\"\n",thead);
      fprintf(fpc,"#include \"%s\"\n","TH2.h");
      fprintf(fpc,"#include \"%s\"\n","TStyle.h");
      fprintf(fpc,"#include \"%s\"\n","TCanvas.h");
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Loop()\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"//   In a ROOT session, you can do:\n");
      fprintf(fpc,"//      Root > .L %s.C\n",classname);
      fprintf(fpc,"//      Root > %s t\n",classname);
      fprintf(fpc,"//      Root > t.GetEntry(12); // Fill t data members with entry number 12\n");
      fprintf(fpc,"//      Root > t.Show();       // Show values of entry 12\n");
      fprintf(fpc,"//      Root > t.Show(16);     // Read and show values of entry 16\n");
      fprintf(fpc,"//      Root > t.Loop();       // Loop on all entries\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"\n//     This is the loop skeleton\n");
      fprintf(fpc,"//       To read only selected branches, Insert statements like:\n");
      fprintf(fpc,"// METHOD1:\n");
      fprintf(fpc,"//    fChain->SetBranchStatus(\"*\",0);  // disable all branches\n");
      fprintf(fpc,"//    fChain->SetBranchStatus(\"branchname\",1);  // activate branchname\n");
      fprintf(fpc,"// METHOD2: replace line\n");
      fprintf(fpc,"//    fChain->GetEntry(i);  // read all branches\n");
      fprintf(fpc,"//by  b_branchname->GetEntry(i); //read only this branch\n");
      fprintf(fpc,"   if (fChain == 0) return;\n");
      fprintf(fpc,"\n   Int_t nentries = Int_t(fChain->GetEntriesFast());\n");
      fprintf(fpc,"\n   Int_t nbytes = 0, nb = 0;\n");
      fprintf(fpc,"   for (Int_t jentry=0; jentry<nentries;jentry++) {\n");
      fprintf(fpc,"      Int_t ientry = LoadTree(jentry); //in case of a TChain, ientry is the entry number in the current file\n");
      fprintf(fpc,"      if (ientry < 0) break;\n");
      fprintf(fpc,"      nb = fChain->GetEntry(jentry);   nbytes += nb;\n");
      fprintf(fpc,"      // if (Cut(ientry) < 0) continue;\n");
      fprintf(fpc,"   }\n");
      fprintf(fpc,"}\n");
   }
   if (opt.Contains("selector")) {
      // generate usage comments and list of includes
      fprintf(fpc,"#define %s_cxx\n",classname);
      fprintf(fpc,"// The class definition in %s.h has been generated automatically\n",classname);
      fprintf(fpc,"// by the ROOT utility TTree::MakeSelector().\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// This class is derived from the ROOT class TSelector.\n");
      fprintf(fpc,"// The following members functions are called by the TTree::Process() functions:\n");
      fprintf(fpc,"//    Begin():       called everytime a loop on the tree starts,\n");
      fprintf(fpc,"//                   a convenient place to create your histograms.\n");
      fprintf(fpc,"//    Notify():      this function is called at the first entry of a new Tree\n");
      fprintf(fpc,"//                   in a chain.\n");
      fprintf(fpc,"//    ProcessCut():  called at the beginning of each entry to return a flag,\n");
      fprintf(fpc,"//                   true if the entry must be analyzed.\n");
      fprintf(fpc,"//    ProcessFill(): called in the entry loop for all entries accepted\n");
      fprintf(fpc,"//                   by Select.\n");
      fprintf(fpc,"//    Terminate():   called at the end of a loop on the tree,\n");
      fprintf(fpc,"//                   a convenient place to draw/fit your histograms.\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"//   To use this file, try the following session on your Tree T\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// Root > T.Process(\"%s.C\")\n",classname);
      fprintf(fpc,"// Root > T.Process(\"%s.C,\"some options\"\")\n",classname);
      fprintf(fpc,"// Root > T.Process(\"%s.C+\")\n",classname);
      fprintf(fpc,"//\n");
      fprintf(fpc,"#include \"%s\"\n",thead);
      fprintf(fpc,"#include \"%s\"\n","TH2.h");
      fprintf(fpc,"#include \"%s\"\n","TStyle.h");
      fprintf(fpc,"#include \"%s\"\n","TCanvas.h");
      fprintf(fpc,"\n");
      // generate code for class member function Begin
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Begin(TTree *tree)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // Function called before starting the event loop.\n");
      fprintf(fpc,"   // Initialize the tree branches.\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   Init(tree);\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   TString option = GetOption();\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function ProcessCut
      fprintf(fpc,"\n");
      fprintf(fpc,"Bool_t %s::ProcessCut(Int_t entry)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // Selection function.\n");
      fprintf(fpc,"   // Entry is the entry number in the current tree.\n");
      fprintf(fpc,"   // Read only the necessary branches to select entries.\n");
      fprintf(fpc,"   // Return kFALSE as soon as a bad entry is detected.\n");
      fprintf(fpc,"   // To read complete event, call fChain->GetTree()->GetEntry(entry).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   return kTRUE;\n");
      fprintf(fpc,"}\n");
      // generate code for class member function ProcessFill
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::ProcessFill(Int_t entry)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // Function called for selected entries only.\n");
      fprintf(fpc,"   // Entry is the entry number in the current tree.\n");
      fprintf(fpc,"   // Read branches not processed in ProcessCut() and fill histograms.\n");
      fprintf(fpc,"   // To read complete event, call fChain->GetTree()->GetEntry(entry).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function Terminate
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Terminate()\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // Function called at the end of the event loop.\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
   }
   Info("MakeClass","Files: %s and %s generated from Tree: %s",thead,tcimp,fTree->GetName());
   delete [] leafStatus;
   delete [] thead;
   delete [] tcimp;
   delete [] treefile;
   fclose(fp);
   fclose(fpc);

   return 0;
}


//______________________________________________________________________________
Int_t TTreePlayer::MakeCode(const char *filename)
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
//      - T->MakeCode("user.C");
//    where T is the name of the Tree in file myfile.root
//    and user.C the name of the file created by this function.
//
//   NOTE: Since the implementation of this function, new and better
//         function TTree::MakeClass and TTree::MakeSelector have been developped.
//
//          Author: Rene Brun
//====>

// Connect output file
   char *tfile = new char[1000];
   if (filename) strcpy(tfile,filename);
   else          sprintf(tfile,"%s.C",fTree->GetName());
   FILE *fp = fopen(tfile,"w");
   if (!fp) {
      printf("Cannot open output file:%s\n",tfile);
      return 3;
   }
   char *treefile = new char[1000];
   if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile())
                strcpy(treefile,fTree->GetDirectory()->GetFile()->GetName());
   else         strcpy(treefile,"Memory Directory");
   // In the case of a chain, the GetDirectory information usually does
   // pertain to the Chain itself but to the currently loaded tree.
   // So we can not rely on it.
   Bool_t ischain = fTree->InheritsFrom("TChain");

// Print header
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;
   TDatime td;
   fprintf(fp,"{\n");
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"//   This file has been automatically generated \n");
   fprintf(fp,"//     (%s by ROOT version%s)\n",td.AsString(),gROOT->GetVersion());
   if (!ischain) {
      fprintf(fp,"//   from TTree %s/%s\n",fTree->GetName(),fTree->GetTitle());
      fprintf(fp,"//   found on file: %s\n",treefile);
   } else {
      fprintf(fp,"//   from TChain %s/%s\n",fTree->GetName(),fTree->GetTitle());
   }
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"\n");
   fprintf(fp,"\n");


// Reset and file connect
   fprintf(fp,"//Reset ROOT and connect tree file\n");
   fprintf(fp,"   gROOT->Reset();\n");
   if (ischain) {
      fprintf(fp,"\n#ifdef SINGLE_TREE\n");
      fprintf(fp,"   // The following code should be used if you want this code to access\n");
      fprintf(fp,"   // a single tree instead of a chain\n");
   }
   fprintf(fp,"   TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(\"%s\");\n",treefile);
   fprintf(fp,"   if (!f) {\n");
   fprintf(fp,"      f = new TFile(\"%s\");\n",treefile);
   if (gDirectory != gFile) {
      fprintf(fp,"      f->cd(\"%s\");\n",gDirectory->GetPath());
   }
   fprintf(fp,"   }\n");
   fprintf(fp,"   TTree *%s = (TTree*)gDirectory->Get(\"%s\");\n\n",fTree->GetName(),fTree->GetName());
   if (ischain) {
      fprintf(fp,"#else // SINGLE_TREE\n\n");
      fprintf(fp,"   // The following code should be used if you want this code to access a chain\n");
      fprintf(fp,"   // of trees.\n");
      fprintf(fp,"   TChain *%s = new TChain(\"%s\",\"%s\");\n",
                 fTree->GetName(),fTree->GetName(),fTree->GetTitle());
      TIter next(((TChain*)fTree)->GetListOfFiles());
      TChainElement *element;
      while ((element = (TChainElement*)next())) {
        fprintf(fp,"   %s->Add(\"%s/%s\");\n",fTree->GetName(),element->GetTitle(),element->GetName());
      }
      fprintf(fp,"#endif // SINGLE_TREE\n\n");
   }

// First loop on all leaves to generate type declarations
   fprintf(fp,"//Declaration of leaves types\n");
   Int_t len, l;
   TLeaf *leafcount;
   TLeafObject *leafobj;
   char *bname;
   const char *headOK  = "   ";
   const char *headcom = " //";
   const char *head;
   char branchname[128];
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      if (branch->GetListOfBranches()->GetEntriesFast() > 0) continue;

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strcpy(branchname,branch->GetName());
         strcat(branchname,".");
         strcat(branchname,leaf->GetTitle());
         if (leafcount) {
           // remove any dimension in title
           char *dim =  (char*)strstr(branchname,"[");
           dim[0] = 0;
         }
      } else {
         if (leafcount) strcpy(branchname,branch->GetName());
         else           strcpy(branchname,leaf->GetTitle());
      }
      char *twodim = (char*)strstr(leaf->GetTitle(),"][");
      bname = branchname;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      if (branch->IsA() == TBranchObject::Class()) {
         leafobj = (TLeafObject*)leaf;
         if (leafobj->GetClass()) head = headOK;
         else                     head = headcom;
         fprintf(fp,"%s%-15s *%s = 0;\n",head,leafobj->GetTypeName(), leafobj->GetName());
         continue;
      }
      if (leafcount) {
         len = leafcount->GetMaximum();
	 // Dimensions can be in the branchname for a split Object with a fix length C array.
	 // Theses dimensions HAVE TO be placed after the dimension explicited by leafcount
	 char *dimInName = (char*) strstr(branchname,"[");
	 char *dimensions = 0;
	 if ( twodim || dimInName ) {
	   int dimlen = 0;
	   if (dimInName) dimlen += strlen(dimInName) + 1;
	   if (twodim)    dimlen += strlen(twodim) + 1;
	   dimensions = new char[dimlen];
	   if (dimInName) {
	     strcpy(dimensions,dimInName);
	     dimInName[0] = 0; // terminate branchname before the array dimensions.
	   } else dimensions[0] = 0;
	   if (twodim) strcat(dimensions,(char*)(twodim+1));
	 }
         if (dimensions) {
            fprintf(fp,"   %-15s %s[%d]%s;\n",leaf->GetTypeName(), branchname,len,dimensions);
	    delete dimensions;
         } else {
            fprintf(fp,"   %-15s %s[%d];\n",leaf->GetTypeName(), branchname,len);
         }
      } else {
         if (strstr(branchname,"[")) len = 1;
         if (len < 2) fprintf(fp,"   %-15s %s;\n",leaf->GetTypeName(), branchname);
         else         fprintf(fp,"   %-15s %s[%d];\n",leaf->GetTypeName(), branchname,len);
      }
   }

// Second loop on all leaves to set the corresponding branch address
   fprintf(fp,"\n//Set branch addresses\n");
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strcpy(branchname,branch->GetName());
         strcat(branchname,".");
         strcat(branchname,leaf->GetTitle());
         if (leafcount) {
           // remove any dimension in title
           char *dim =  (char*)strstr(branchname,"[");
           dim[0] = 0;
         }
      } else {
         if (leafcount) strcpy(branchname,branch->GetName());
         else           strcpy(branchname,leaf->GetTitle());
      }
      bname = branchname;
      while (*bname) {if (*bname == '.') *bname='_'; bname++;}
      char *brak = strstr(branchname,"[");
      if (brak) *brak = 0;
      head = headOK;
      if (branch->IsA() == TBranchObject::Class()) {
         strcpy(branchname,branch->GetName());
         leafobj = (TLeafObject*)leaf;
         if (!leafobj->GetClass()) head = headcom;
      }
      if (leafcount) len = leafcount->GetMaximum()+1;
      if (len > 1 || brak) fprintf(fp,"%s%s->SetBranchAddress(\"%s\",%s);\n",head,fTree->GetName(),branch->GetName(),branchname);
      else                 fprintf(fp,"%s%s->SetBranchAddress(\"%s\",&%s);\n",head,fTree->GetName(),branch->GetName(),branchname);
   }

//Generate instructions to make the loop on entries
   fprintf(fp,"\n//     This is the loop skeleton\n");
   fprintf(fp,"//       To read only selected branches, Insert statements like:\n");
   fprintf(fp,"// %s->SetBranchStatus(\"*\",0);  // disable all branches\n",fTree->GetName());
   fprintf(fp,"// %s->SetBranchStatus(\"branchname\",1);  // activate branchname\n",GetName());
   fprintf(fp,"\n   Int_t nentries = %s->GetEntries();\n",fTree->GetName());
   fprintf(fp,"\n   Int_t nbytes = 0;\n");
   fprintf(fp,"//   for (Int_t i=0; i<nentries;i++) {\n");
   fprintf(fp,"//      nbytes += %s->GetEntry(i);\n",fTree->GetName());
   fprintf(fp,"//   }\n");
   fprintf(fp,"}\n");

   printf("Macro: %s generated from Tree: %s\n",tfile,fTree->GetName());
   delete [] tfile;
   delete [] treefile;
   fclose(fp);

   return 0;
}


//______________________________________________________________________________
void TTreePlayer::MakeIndex(TString &varexp, Int_t *index)
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
TPrincipal *TTreePlayer::Principal(const char *varexp, const char *selection, Option_t *option, Int_t nentries, Int_t firstentry)
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
//   see TTreePlayer::DrawSelect for explanation of the other parameters.

   TTreeFormula **var;
   TString *cnames;
   TString onerow;
   TString opt = option;
   opt.ToLower();
   TPrincipal *principal = 0;
   Int_t entry,entryNumber,i,nch;
   Int_t *index = 0;
   Int_t ncols = 8;   // by default first 8 columns are printed only
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = (Int_t)fTree->GetEntriesFriend() -1;
      nentries   = lastentry - firstentry + 1;
   }

//*-*- Compile selection expression if there is one
   fSelect = 0;
   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      if (!fSelect) return principal;
      if (!fSelect->GetNdim()) { delete fSelect; fSelect = 0; return principal; }
   }
//*-*- if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (!strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      cnames = new TString[ncols];
      for (i=0;i<ncols;i++) {
         cnames[i] = ((TLeaf*)leaves->At(i))->GetName();
      }
//*-*- otherwise select only the specified columns
   } else {
      ncols = 1;
      onerow = varexp;
      for (i=0;i<onerow.Length();i++)  if (onerow[i] == ':') ncols++;
      cnames = new TString[ncols];
      index  = new Int_t[ncols+1];
      MakeIndex(onerow,index);
      for (i=0;i<ncols;i++) {
         cnames[i] = GetNameByIndex(onerow,index,i);
      }
   }
   var = new TTreeFormula* [ncols];
   Double_t *xvars = new Double_t[ncols];

//*-*- Create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
   }

   //*-* Build the TPrincipal object
   if (opt.Contains("n")) principal = new TPrincipal(ncols, "n");
   else                   principal = new TPrincipal(ncols);

//*-*- loop on all selected entries
   fSelectedRows = 0;
   Int_t tnumber = -1;
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Int_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         for (i=0;i<ncols;i++) var[i]->UpdateFormulaLeaves();
      }
      if (fSelect) {
         fSelect->GetNdata();
         if (fSelect->EvalInstance(0) == 0) continue;
      }
      onerow = Form("* %8d ",entryNumber);
      for (i=0;i<ncols;i++) {
         xvars[i] = var[i]->EvalInstance(0);
      }
      principal->AddRow(xvars);
   }

//*-* some actions with principal ?
   if (opt.Contains("p")) {
        principal->MakePrincipals(); // Do the actual analysis
        if (opt.Contains("d")) principal->Print();
        if (opt.Contains("h")) principal->MakeHistograms();
        if (opt.Contains("c")) principal->MakeCode();
   }

//*-*- delete temporary objects
   delete fSelect; fSelect = 0;
   for (i=0;i<ncols;i++) {
      delete var[i];
   }
   delete [] var;
   delete [] cnames;
   delete [] index;
   delete [] xvars;

   return principal;
}

//______________________________________________________________________________
Int_t TTreePlayer::Process(const char *filename,Option_t *option, Int_t nentries, Int_t firstentry)
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
//     Bool_t TSelector::Notify(). This function is called at the first entry
//          of a new file in a chain.
//
//     Bool_t TSelector::ProcessCut(Int_t tentry). This function is called
//          before processing tentry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise. tentry is the entry number in the current Tree.
//
//     void TSelector::ProcessFill(Int_t tentry). This function is called for
//          all selected events. User fills histograms in this function.
//
//     void TSelector::Terminate(). This function is called at the end of
//          the loop on all events.
//
//   if filename is of the form file.C, the file will be interpreted.
//   if filename is of the form file.C++, the file file.C will be compiled
//      and dynamically loaded.
//   if filename is of the form file.C+, the file file.C will be compiled
//      and dynamically loaded. At next call, if file.C is older than file.o
//      and file.so, the file.C is not compiled, only file.so is loaded.


   //Get a pointer to the TSelector object
   static TSelector *selector = 0;
   delete selector; //delete previous selector if any
   // This might reloads the script and delete your option
   // string! so let copy it first:
   TString opt(option);
   TString file(filename);
   selector = TSelector::GetSelector(file);
   if (!selector) return -1;

   Int_t nsel = Process(selector,opt,nentries,firstentry);
   return nsel;
}

//______________________________________________________________________________
Int_t TTreePlayer::Process(TSelector *selector,Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*Process this tree executing the code in selector*-*-*-*-*
//*-*              ================================================
//
//   The TSelector class has the following member functions:
//
//     void TSelector::Begin(). This function is called before looping on the
//          events in the Tree. The user can create his histograms in this function.
//
//     Bool_t TSelector::Notify(). This function is called at the first entry
//          of a new file in a chain.
//
//     Bool_t TSelector::ProcessCut(Int_t tentry). This function is called
//          before processing tentry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise. tentry is the entry number in the current Tree.
//
//     void TSelector::ProcessFill(Int_t tentry). This function is called for
//          all selected events. User fills histograms in this function.
//
//     void TSelector::Terminate(). This function is called at the end of
//          the loop on all events.
//
//  If the Tree (Chain) has an associated EventList, the loop is on the nentries
//  of the EventList, starting at firstentry, otherwise the loop is on the
//  specified Tree entries.

   selector->SetOption(option);

   selector->Begin(fTree); //<===call user initialisation function

   //Create a timer to get control in the entry loop(s)
   TProcessEventTimer *timer = 0;
   Int_t interval = fTree->GetTimerInterval();
   if (!gROOT->IsBatch() && interval)
      timer = new TProcessEventTimer(interval);

   //loop on entries (elist or all entries)
   Int_t treeNumber = -1;
   Long_t entry, entryNumber;
   fSelectedRows = 0;
   Int_t nent = Int_t(fTree->GetEntriesFriend());
   TEventList *elist = fTree->GetEventList();
   if (elist) nent = elist->GetN();
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > nent) {
      lastentry  = nent -1;
      nentries   = lastentry - firstentry + 1;
   }
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      if (timer && timer->ProcessEvents()) break;
      if (gROOT->IsInterrupted()) break;
      if (elist) entryNumber = fTree->LoadTree(elist->GetEntry(entry));
      else       entryNumber = fTree->LoadTree(entry);
      if (entryNumber < 0) break;
      if (fTree->GetTreeNumber() != treeNumber) {
         treeNumber = fTree->GetTreeNumber();
         selector->Notify();
      }
      if (selector->ProcessCut(entryNumber))
          selector->ProcessFill(entryNumber); //<==call user analysis function
   }
   selector->Terminate();  //<==call user termination function

   return fSelectedRows;
}

//______________________________________________________________________________
Int_t TTreePlayer::Scan(const char *varexp, const char *selection, Option_t *,
                       Int_t nentries, Int_t firstentry)
{
   // Loop on Tree and print entries passing selection. If varexp is 0 (or "")
   // then print only first 8 columns. If varexp = "*" print all columns.
   // Otherwise a columns selection can be made using "var1:var2:var3".

   TTreeFormula **var;
   TString *cnames;
   TString onerow;
   Int_t entry,entryNumber,i,nch;
   Int_t *index = 0;
   Int_t ncols = 8;   // by default first 8 columns are printed only
   ofstream out;
   Int_t lenfile = 0;
   char * fname = 0;
   if (fScanRedirect) {
      fTree->SetScanField(0);  // no page break if Scan is redirected
      fname = (char *) fScanFileName;
      if (!fname) fname = (char*)"";
      lenfile = strlen(fname);
      if (!lenfile) {
         Int_t nch = strlen(fTree->GetName());
         fname = new char[nch+10];
         strcpy(fname, fTree->GetName());
         strcat(fname, "-scan.dat");
      }
      out.open(fname, ios::out);
      if (!out.good ()) {
         if (!lenfile) delete [] fname;
         Error("Scan","Can not open file for redirection");
         return 0;
      }
   }
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = (Int_t)fTree->GetEntriesFriend() -1;
      nentries   = lastentry - firstentry + 1;
   }

//*-*- Compile selection expression if there is one
   fSelect = 0;
   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      if (!fSelect) return -1;
      if (!fSelect->GetNdim()) { delete fSelect; fSelect = 0; return -1; }
   }
//*-*- if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (!strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      cnames = new TString[ncols];
      Int_t ncs = ncols;
      ncols = 0;
      for (i=0;i<ncs;i++) {
         TLeaf *lf = (TLeaf*)leaves->At(i);
         if (lf->GetBranch()->GetListOfBranches()->GetEntries() > 0) continue;
         cnames[ncols] = lf->GetName();
         ncols++;
      }
//*-*- otherwise select only the specified columns
   } else {
      ncols = 1;
      onerow = varexp;
      for (i=0;i<onerow.Length();i++)  if (onerow[i] == ':') ncols++;
      cnames = new TString[ncols];
      index  = new Int_t[ncols+1];
      MakeIndex(onerow,index);
      for (i=0;i<ncols;i++) {
         cnames[i] = GetNameByIndex(onerow,index,i);
      }
   }
   var = new TTreeFormula* [ncols];

//*-*- Create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
   }
//*-*- Print header
   onerow = "***********";
   for (i=0;i<ncols;i++) {
      onerow += Form("*%11.11s",var[i]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   onerow = "*    Row   ";
   for (i=0;i<ncols;i++) {
      onerow += Form("* %9.9s ",var[i]->PrintValue(-1));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   onerow = "***********";
   for (i=0;i<ncols;i++) {
      onerow += Form("*%11.11s",var[i]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
//*-*- loop on all selected entries
   fSelectedRows = 0;
   Int_t tnumber = -1;
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Int_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         for (i=0;i<ncols;i++) var[i]->UpdateFormulaLeaves();
      }
      if (fSelect) {
         fSelect->GetNdata();
         if (fSelect->EvalInstance(0) == 0) continue;
      }
      onerow = Form("* %8d ",entryNumber);
      for (i=0;i<ncols;i++) {
         onerow += Form("* %9.9s ",var[i]->PrintValue(0));
      }
      fSelectedRows++;
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
      if (fTree->GetScanField() > 0 && fSelectedRows > 0) {
         if (fSelectedRows%fTree->GetScanField() == 0) {
            fprintf(stderr,"Type <CR> to continue or q to quit ==> ");
            int answer, readch;
            readch = getchar();
            answer = readch;
            while (readch != '\n' && readch != EOF) readch = getchar();
            if (answer == 'q' || answer == 'Q') break;
         }
      }
   }
   onerow = "***********";
   for (i=0;i<ncols;i++) {
      onerow += Form("*%11.11s",var[i]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   if (fSelect) Printf("==> %d selected %s", fSelectedRows,
                      fSelectedRows == 1 ? "entry" : "entries");
   if (fScanRedirect) printf("File <%s> created\n", fname);

//*-*- delete temporary objects
   if (!lenfile) delete [] fname;
   delete fSelect; fSelect = 0;
   for (i=0;i<ncols;i++) {
      delete var[i];
   }
   delete [] var;
   delete [] cnames;
   delete [] index;
   return fSelectedRows;
}

//______________________________________________________________________________
TSQLResult *TTreePlayer::Query(const char *varexp, const char *selection,
                               Option_t *, Int_t nentries, Int_t firstentry)
{
   // Loop on Tree and return TSQLResult object containing entries passing
   // selection. If varexp is 0 (or "") then print only first 8 columns.
   // If varexp = "*" print all columns. Otherwise a columns selection can
   // be made using "var1:var2:var3". In case of error 0 is returned otherwise
   // a TSQLResult object which must be deleted by the user.

   TTreeFormula **var;
   TString *cnames;
   TString onerow;
   Int_t entry,entryNumber,i,nch;
   Int_t *index = 0;
   Int_t ncols = 8;   // by default first 8 columns are printed only
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;
   Int_t lastentry = firstentry + nentries -1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = (Int_t)fTree->GetEntriesFriend() -1;
      nentries   = lastentry - firstentry + 1;
   }

   // compile selection expression if there is one
   fSelect = 0;
   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      if (!fSelect) return 0;
      if (!fSelect->GetNdim()) { delete fSelect; fSelect = 0; return 0; }
   }

   // if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (!strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      cnames = new TString[ncols];
      for (i=0;i<ncols;i++) {
         cnames[i] = ((TLeaf*)leaves->At(i))->GetName();
      }
   } else {
      // otherwise select only the specified columns
      ncols = 1;
      onerow = varexp;
      for (i=0;i<onerow.Length();i++)  if (onerow[i] == ':') ncols++;
      cnames = new TString[ncols];
      index  = new Int_t[ncols+1];
      MakeIndex(onerow,index);
      for (i=0;i<ncols;i++) {
         cnames[i] = GetNameByIndex(onerow,index,i);
      }
   }
   var = new TTreeFormula* [ncols];

   // create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
   }

   // fill header info into result object
   TTreeResult *res = new TTreeResult(ncols);
   for (i = 0; i < ncols; i++) {
      res->AddField(i, var[i]->PrintValue(-1));
   }

   // loop on all selected entries
   const char *aresult;
   Int_t len;
   char *arow = new char[ncols*50];
   fSelectedRows = 0;
   Int_t tnumber = -1;
   Int_t *fields = new Int_t[ncols];
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Int_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         for (i=0;i<ncols;i++) var[i]->UpdateFormulaLeaves();
      }
      if (fSelect) {
         fSelect->GetNdata();
         if (fSelect->EvalInstance(0) == 0) continue;
      }

      for (i=0;i<ncols;i++) {
         aresult = var[i]->PrintValue(0);
         len = strlen(aresult)+1;
         if (i == 0) {
            memcpy(arow,aresult,len);
            fields[i] = len;
         } else {
            memcpy(arow+fields[i-1],aresult,len);
            fields[i] = fields[i-1] + len;
         }
      }
      res->AddRow(new TTreeRow(ncols,fields,arow));
      fSelectedRows++;
   }

   // delete temporary objects
   delete fSelect; fSelect = 0;
   for (i=0;i<ncols;i++) {
      delete var[i];
   }
   delete [] fields;
   delete [] arow;
   delete [] var;
   delete [] cnames;
   delete [] index;

   return res;
}

//_______________________________________________________________________
void TTreePlayer::SetEstimate(Int_t )
{
//*-*-*-*-*-*-*-*-*Set number of entries to estimate variable limits*-*-*-*
//*-*              ================================================
//
   delete [] fV1;  fV1 = 0;
   delete [] fV2;  fV2 = 0;
   delete [] fV3;  fV3 = 0;
   delete [] fW;   fW  = 0;
}

//_______________________________________________________________________
void TTreePlayer::SetPacketSize(Int_t size)
{
//*-*-*-*-*-*-*-*-*Set number of entries per packet for parallel root*-*-*-*-*
//*-*              =================================================

   fPacketSize = size;
}

//_______________________________________________________________________
void TTreePlayer::StartViewer(Int_t ww, Int_t wh)
{
//*-*-*-*-*-*-*-*-*Start the TTreeViewer on this TTree*-*-*-*-*-*-*-*-*-*
//*-*              ===================================
//
//  ww is the width of the canvas in pixels
//  wh is the height of the canvas in pixels

   if (gROOT->IsBatch()) {
      Warning("StartViewer", "viewer cannot run in batch mode");
      return;
   }

   gROOT->LoadClass("TTreeViewer","TreeViewer");
#ifdef R__WIN32
#ifndef GDK_WIN32
   gROOT->ProcessLine(Form("new TTreeViewer(\"%s\",\"TreeViewer\",%d,%d);",fTree->GetName(),ww,wh));
#else
   if (ww || wh) { }   // use unused variables
   gROOT->ProcessLine(Form("new TTreeViewer(\"%s\");",fTree->GetName()));
#endif
#else
   if (ww || wh) { }   // use unused variables
   gROOT->ProcessLine(Form("new TTreeViewer(\"%s\");",fTree->GetName()));
#endif
}

//______________________________________________________________________________
void TTreePlayer::TakeAction(Int_t nfill, Int_t &npoints, Int_t &action, TObject *obj, Option_t *option)
{
//*-*-*-*-*-*Execute action for object obj nfill times*-*-*-*-*-*-*-*-*-*
//*-*        =========================================

  Int_t i;
//__________________________1-D histogram_______________________
  if      (action ==  1) ((TH1*)obj)->FillN(nfill,fV1,fW);
//__________________________2-D histogram_______________________
  else if (action ==  2) {
     TH2 *h2 = (TH2*)obj;
     for(i=0;i<nfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
  }
//__________________________Profile histogram_______________________
  else if (action ==  4) ((TProfile*)obj)->FillN(nfill,fV2,fV1,fW);
//__________________________Event List
  else if (action ==  5) {
     TEventList *elist = (TEventList*)obj;
     Int_t enumb = fTree->GetChainOffset() + fTree->GetReadEntry();
     if (elist->GetIndex(enumb) < 0) elist->Enter(enumb);
  }
//__________________________2D scatter plot_______________________
  else if (action == 12) {
     TPolyMarker *pm = new TPolyMarker(nfill);
     pm->SetMarkerStyle(fTree->GetMarkerStyle());
     pm->SetMarkerColor(fTree->GetMarkerColor());
     pm->SetMarkerSize(fTree->GetMarkerSize());
     Double_t u, v;
     Double_t umin = gPad->GetUxmin();
     Double_t umax = gPad->GetUxmax();
     Double_t vmin = gPad->GetUymin();
     Double_t vmax = gPad->GetUymax();

     for (i=0;i<nfill;i++) {
        u = gPad->XtoPad(fV2[i]);
        v = gPad->YtoPad(fV1[i]);
        if (u < umin) u = umin;
        if (u > umax) u = umax;
        if (v < vmin) v = vmin;
        if (v > vmax) v = vmax;
        pm->SetPoint(i,u,v);
     }

     pm->Draw();
     TH2 *h2 = (TH2*)obj;
     for(i=0;i<nfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
  }
//__________________________3D scatter plot_______________________
  else if (action ==  3) {
     TH3 *h3 =(TH3*)obj;
     for(i=0;i<nfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
  }
  else if (action == 13) {
     TPolyMarker3D *pm3d = new TPolyMarker3D(nfill);
     pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
     pm3d->SetMarkerColor(fTree->GetMarkerColor());
     pm3d->SetMarkerSize(fTree->GetMarkerSize());
     for (i=0;i<nfill;i++) { pm3d->SetPoint(i,fV3[i],fV2[i],fV1[i]);}
     pm3d->Draw();
     TH3 *h3 =(TH3*)obj;
     for(i=0;i<nfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
  }
//__________________________2D Profile Histogram__________________
  else if (action == 23) {
     TProfile2D *hp2 =(TProfile2D*)obj;
     for(i=0;i<nfill;i++) hp2->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
  }
  else if (action < 0) {
     action = -action;
     TakeEstimate(nfill,npoints,action,obj,option);
  }

//*-* Do we need to update screen?
  npoints += nfill;
  if (!fTree->GetUpdate()) return;
  if (npoints > fDraw+fTree->GetUpdate()) {
     if (fDraw) gPad->Modified();
     else       obj->Draw(option);
     gPad->Update();
     fDraw = npoints;
  }
}


//______________________________________________________________________________
void TTreePlayer::TakeEstimate(Int_t nfill, Int_t &, Int_t action, TObject *obj, Option_t *option)
{
//*-*-*-*-*-*Estimate limits for 1-D, 2-D or 3-D objects*-*-*-*-*-*-*-*-*-*
//*-*        ===========================================

  Int_t i;
  Double_t rmin[3],rmax[3];
  fVmin[0] = fVmin[1] = fVmin[2] = FLT_MAX; //in float.h
  fVmax[0] = fVmax[1] = fVmax[2] = -fVmin[0];
//__________________________1-D histogram_______________________
  if      (action ==  1) {
     TH1 *h1 = (TH1*)obj;
     fHistogram = h1;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h1,fVmin[0],fVmax[0]);
     }     
     h1->FillN(nfill, fV1, fW);
//__________________________2-D histogram_______________________
  } else if (action ==  2) {
     TH2 *h2 = (TH2*)obj;
     fHistogram = h2;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
           if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
           if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
     }  
     for(i=0;i<nfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
//__________________________Profile histogram_______________________
  } else if (action ==  4) {
     TProfile *hp = (TProfile*)obj;
     fHistogram = hp;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
           if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
           if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hp,fVmin[1],fVmax[1]);
     }
     hp->FillN(nfill, fV2, fV1, fW);
//__________________________2D scatter plot_______________________
  } else if (action == 12) {
     TH2 *h2 = (TH2*)obj;
     fHistogram = h2;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
           if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
           if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
     }
     
     if (!strstr(option,"same") && !strstr(option,"goff")) {
        UInt_t statbit = h2->TestBit(TH1::kNoStats);
        h2->SetBit(TH1::kNoStats);
        h2->DrawCopy(option);
        if (statbit) h2->SetBit(TH1::kNoStats);
        gPad->Update();
     }
     TPolyMarker *pm = new TPolyMarker(nfill);
     pm->SetMarkerStyle(fTree->GetMarkerStyle());
     pm->SetMarkerColor(fTree->GetMarkerColor());
     pm->SetMarkerSize(fTree->GetMarkerSize());
     Double_t u, v;
     Double_t umin = gPad->GetUxmin();
     Double_t umax = gPad->GetUxmax();
     Double_t vmin = gPad->GetUymin();
     Double_t vmax = gPad->GetUymax();

     for (i=0;i<nfill;i++) {
        u = gPad->XtoPad(fV2[i]);
        v = gPad->YtoPad(fV1[i]);
        if (u < umin) u = umin;
        if (u > umax) u = umax;
        if (v < vmin) v = vmin;
        if (v > vmax) v = vmax;
        pm->SetPoint(i,u,v);
     }
     if (!fDraw && !strstr(option,"goff")) pm->Draw();
     if (!h2->TestBit(kCanDelete)) {
        for (i=0;i<nfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
     }
//__________________________3D scatter plot_______________________
  } else if (action == 3 || action == 13) {
     TH3 *h3 = (TH3*)obj;
     fHistogram = h3;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
           if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
           if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
           if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
           if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h3,fVmin[2],fVmax[2],fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
     }
     
     if (action == 3 || !h3->TestBit(kCanDelete)) {
        for (i=0;i<nfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
     }
     if (action == 3) return;
     if (!strstr(option,"same") && !strstr(option,"goff")) {
        h3->DrawCopy(option);
        gPad->Update();
     } else {
        rmin[0] = fVmin[2]; rmin[1] = fVmin[1]; rmin[2] = fVmin[0];
        rmax[0] = fVmax[2]; rmax[1] = fVmax[1]; rmax[2] = fVmax[0];
        gPad->Clear();
        gPad->Range(-1,-1,1,1);
        new TView(rmin,rmax,1);
     }
     TPolyMarker3D *pm3d = new TPolyMarker3D(nfill);
     pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
     pm3d->SetMarkerColor(fTree->GetMarkerColor());
     pm3d->SetMarkerSize(fTree->GetMarkerSize());
     for (i=0;i<nfill;i++) { pm3d->SetPoint(i,fV3[i],fV2[i],fV1[i]);}
     if (!fDraw && !strstr(option,"goff")) pm3d->Draw();
//__________________________2D Profile Histogram__________________
  } else if (action == 23) {
     TProfile2D *hp = (TProfile2D*)obj;
     fHistogram = hp;
     if (fHistogram->TestBit(TH1::kCanRebin)) {
        for (i=0;i<nfill;i++) {
           if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
           if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
           if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
           if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
           if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
           if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
        }
        THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hp,fVmin[2],fVmax[2],fVmin[1],fVmax[1]);
     }
     for (i=0;i<nfill;i++) hp->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
  }
}

//______________________________________________________________________________
void TreeUnbinnedFitLikelihood(Int_t &npar, Double_t *gin, Double_t &r, Double_t *par, Int_t flag)
{
// The fit function used by the unbinned likelihood fit.

  TF1 *fitfunc = (TF1*)tFitter->GetObjectFit();
  Int_t n = gTree->GetSelectedRows();
  Double_t  *data1 = gTree->GetV1();
  Double_t  *data2 = gTree->GetV2();
  Double_t  *data3 = gTree->GetV3();
  Double_t *weight = gTree->GetW();
  Double_t logEpsilon = -230;   // protect against negative probabilities
  Double_t logL = 0.0, prob;
  Double_t sum = fitfunc->GetChisquare();

  Double_t x[3];
  for(Int_t i = 0; i < n; i++) {
    x[0] = data1[i];
    if (data2) x[1] = data2[i];
    if (data3) x[2] = data3[i];
    prob = fitfunc->EvalPar(x,par) * weight[i]/sum;
    if(prob > 0) logL += TMath::Log(prob);
    else         logL += logEpsilon;
  }

  r = -logL;
}


//______________________________________________________________________________
Int_t TTreePlayer::UnbinnedFit(const char *funcname ,const char *varexp, const char *selection,Option_t *option ,Int_t nentries, Int_t firstentry)
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

  Int_t i, npar,nvpar,nparx;
  Double_t par, we, al, bl;
  Double_t eplus,eminus,eparab,globcc,amin,edm,errdef,werr;
  Double_t arglist[10];

  // Set the global fit function so that TreeUnbinnedFitLikelihood can find it.
  TF1* fitfunc = (TF1*)gROOT->GetFunction(funcname);
  if (!fitfunc) { Error("UnbinnedFit", "Unknown function: %s",funcname); return 0; }
  npar = fitfunc->GetNpar();
  if (npar <=0) { Error("UnbinnedFit", "Illegal number of parameters = %d",npar); return 0; }

  // Spin through the data to select out the events of interest
  // Make sure that the arrays V1,etc are created large enough to accomodate
  // all entries
  Int_t oldEstimate = fTree->GetEstimate();
  Int_t nent = Int_t(fTree->GetEntriesFriend());
  fTree->SetEstimate(TMath::Min(nent,nentries));

  Int_t nsel = DrawSelect(varexp, selection, "goff", nentries, firstentry);

  //if no selected entries return
  Int_t nrows = GetSelectedRows();
  if (nrows <= 0) {
     Error("UnbinnedFit", "Cannot fit: no entries selected");
     return 0;
  }

  // Check that function has same dimension as number of variables
  Int_t ndim = GetDimension();
  if (ndim != fitfunc->GetNdim()) {
     Error("UnbinnedFit", "Function dimension=%d not equal to expression dimension=%d",fitfunc->GetNdim(),ndim);
     return 0;
  }

  //Compute total sum of weights to set the normalization factor
  Double_t sum = 0;
  Double_t *w = GetW();
  for (i=0;i<nrows;i++) {
     sum += w[i];
  }
  fitfunc->SetChisquare(sum); //this info can be used in fitfunc

  // Create and set up the fitter
  gTree = fTree;
  tFitter = TVirtualFitter::Fitter(fTree);
  tFitter->Clear();
  tFitter->SetFCN(TreeUnbinnedFitLikelihood);

  tFitter->SetObjectFit(fitfunc);

  TString opt = option;
  opt.ToLower();
  // Some initialisations
   if (!opt.Contains("v")) {
      arglist[0] = -1;
      tFitter->ExecuteCommand("SET PRINT", arglist,1);
      arglist[0] = 0;
      tFitter->ExecuteCommand("SET NOW",   arglist,0);
   }

  // Setup the parameters (#, name, start, step, min, max)
  Double_t min, max;
  for(i = 0; i < npar; i++) {
    fitfunc->GetParLimits(i, min, max);
    if(min < max) {
      tFitter->SetParameter(i, fitfunc->GetParName(i),
                               fitfunc->GetParameter(i),
                               fitfunc->GetParameter(i)/100.0, min, max);
    } else {
      tFitter->SetParameter(i, fitfunc->GetParName(i),
                               fitfunc->GetParameter(i),
                               fitfunc->GetParameter(i)/100.0, 0, 0);
    }


    // Check for a fixed parameter
    if(max <= min && min > 0.0) {
       tFitter->FixParameter(i);
    }
  }  // end for loop through parameters

   // Reset Print level
   if (opt.Contains("v")) {
      arglist[0] = 0;
      tFitter->ExecuteCommand("SET PRINT", arglist,1);
   }

  // Now ready for minimization step
  arglist[0] = TVirtualFitter::GetMaxIterations();
  arglist[1] = 1;
  tFitter->ExecuteCommand("MIGRAD", arglist, 2);
  if (opt.Contains("m")) {
     tFitter->ExecuteCommand("IMPROVE",arglist,0);
  }
  if (opt.Contains("e")) {
     tFitter->ExecuteCommand("HESSE",arglist,0);
     tFitter->ExecuteCommand("MINOS",arglist,0);
  }
  fitfunc->SetChisquare(0); //to not confuse user with the stored sum of w**2
  fitfunc->SetNDF(fitfunc->GetNumberFitPoints()-npar);

   // Get return status into function
   char parName[50];
   for (i=0;i<npar;i++) {
      tFitter->GetParameter(i,parName, par,we,al,bl);
      if (opt.Contains("e")) werr = we;
      else {
         tFitter->GetErrors(i,eplus,eminus,eparab,globcc);
         if (eplus > 0 && eminus < 0) werr = 0.5*(eplus-eminus);
         else                         werr = we;
      }
      fitfunc->SetParameter(i,par);
      fitfunc->SetParError(i,werr);
   }
   tFitter->GetStats(amin,edm,errdef,nvpar,nparx);

   // Print final values of parameters.
   if (!opt.Contains("q")) {
      amin = 0;
      tFitter->PrintResults(1, amin);
   }

   //reset estimate
   fTree->SetEstimate(oldEstimate);

   return nsel;
}


//______________________________________________________________________________
void TTreePlayer::UpdateFormulaLeaves()
{
   // this function is called by TChain::LoadTree when a new Tree is loaded.
   // Because Trees in a TChain may have a different list of leaves, one
   // must update the leaves numbers in the TTreeFormula used by the TreePlayer.

   if (fVar1) fVar1->UpdateFormulaLeaves();
   if (fVar2) fVar2->UpdateFormulaLeaves();
   if (fVar3) fVar3->UpdateFormulaLeaves();
   if (fVar4) fVar4->UpdateFormulaLeaves();
   if (fSelect) fSelect->UpdateFormulaLeaves();
   if (fMultiplicity) fMultiplicity->UpdateFormulaLeaves();
}
