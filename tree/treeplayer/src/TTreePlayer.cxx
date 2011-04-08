// @(#)root/treeplayer:$Id$
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

#include "Riostream.h"
#include "TTreePlayer.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TEventList.h"
#include "TEntryList.h"
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
#include "TPolyMarker.h"
#include "TPolyMarker3D.h"
#include "TDirectory.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TVirtualPad.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TStyle.h"
#include "Foption.h"
#include "TTreeResult.h"
#include "TTreeRow.h"
#include "TPrincipal.h"
#include "TChain.h"
#include "TChainElement.h"
#include "TF1.h"
#include "TH1.h"
#include "TVirtualFitter.h"
#include "TEnv.h"
#include "THLimitsFinder.h"
#include "TSelectorDraw.h"
#include "TSelectorEntries.h"
#include "TPluginManager.h"
#include "TObjString.h"
#include "TTreeProxyGenerator.h"
#include "TTreeIndex.h"
#include "TChainIndex.h"
#include "TRefProxy.h"
#include "TRefArrayProxy.h"
#include "TVirtualMonitoring.h"
#include "TTreeCache.h"
#include "TStyle.h"

#include "HFitInterface.h"
#include "Foption.h"
#include "Fit/UnBinData.h"
#include "Math/MinimizerOptions.h"

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
   fScanFileName   = 0;
   fScanRedirect   = kFALSE;
   fSelectedRows   = 0;
   fDimension      = 0;
   fHistogram      = 0;
   fFormulaList    = new TList();
   fFormulaList->SetOwner(kTRUE);
   fSelector         = new TSelectorDraw();
   fSelectorFromFile = 0;
   fSelectorClass    = 0;
   fSelectorUpdate   = 0;
   fInput            = new TList();
   fInput->Add(new TNamed("varexp",""));
   fInput->Add(new TNamed("selection",""));
   fSelector->SetInputList(fInput);
   gROOT->GetListOfCleanups()->Add(this);
   TClass::GetClass("TRef")->AdoptReferenceProxy(new TRefProxy());
   TClass::GetClass("TRefArray")->AdoptReferenceProxy(new TRefArrayProxy());
}

//______________________________________________________________________________
TTreePlayer::~TTreePlayer()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   delete fFormulaList;
   delete fSelector;
   DeleteSelectorFromFile();
   fInput->Delete();
   delete fInput;
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
TVirtualIndex *TTreePlayer::BuildIndex(const TTree *T, const char *majorname, const char *minorname)
{
   // Build the index for the tree (see TTree::BuildIndex)

   TVirtualIndex *index;
   if (dynamic_cast<const TChain*>(T)) {
      index = new TChainIndex(T, majorname, minorname);
      if (index->IsZombie()) {
         delete index;
         Error("BuildIndex", "Creating a TChainIndex unsuccessfull - switching to TTreeIndex");
      }
      else
         return index;
   }
   return new TTreeIndex(T,majorname,minorname);
}

//______________________________________________________________________________
TTree *TTreePlayer::CopyTree(const char *selection, Option_t *, Long64_t nentries,
                             Long64_t firstentry)
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
   // IMPORTANT: The copied tree stays connected with this tree until this tree
   //            is deleted.  In particular, any changes in branch addresses
   //            in this tree are forwarded to the clone trees.  Any changes
   //            made to the branch addresses of the copied trees are over-ridden
   //            anytime this tree changes its branch addresses.
   //            Once this tree is deleted, all the addresses of the copied tree
   //            are reset to their default values.
   //
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
   TTree *tree = fTree->CloneTree(0);
   if (tree == 0) return 0;

   // The clone should not delete any shared i/o buffers.
   TObjArray* branches = tree->GetListOfBranches();
   Int_t nb = branches->GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i) {
      TBranch* br = (TBranch*) branches->UncheckedAt(i);
      if (br->InheritsFrom(TBranchElement::Class())) {
         ((TBranchElement*) br)->ResetDeleteObject();
      }
   }

   Long64_t entry,entryNumber;
   nentries = GetEntriesToProcess(firstentry, nentries);

   // Compile selection expression if there is one
   TTreeFormula *select = 0; // no need to interfere with fSelect since we
                             // handle the loop explicitly below and can call
                             // UpdateFormulaLeaves ourselves.
   if (strlen(selection)) {
      select = new TTreeFormula("Selection",selection,fTree);
      if (!select || !select->GetNdim()) {
         delete select;
         delete tree;
         return 0;
      }
      fFormulaList->Add(select);
   }

   //loop on the specified entries
   Int_t tnumber = -1;
   for (entry=firstentry;entry<firstentry+nentries;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if (select) select->UpdateFormulaLeaves();
      }
      if (select) {
         Int_t ndata = select->GetNdata();
         Bool_t keep = kFALSE;
         for(Int_t current = 0; current<ndata && !keep; current++) {
            keep |= (select->EvalInstance(current) != 0);
         }
         if (!keep) continue;
      }
      fTree->GetEntry(entryNumber);
      tree->Fill();
   }
   fFormulaList->Clear();
   return tree;
}

//______________________________________________________________________________
void TTreePlayer::DeleteSelectorFromFile()
{
// Delete any selector created by this object.
// The selector has been created using TSelector::GetSelector(file)

   if (fSelectorFromFile && fSelectorClass) {
      if (fSelectorClass->IsLoaded()) {
         delete fSelectorFromFile;
      }
   }
   fSelectorFromFile = 0;
   fSelectorClass = 0;
}

//______________________________________________________________________________
Long64_t TTreePlayer::DrawScript(const char* wrapperPrefix,
                                 const char *macrofilename, const char *cutfilename,
                                 Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Draw the result of a C++ script.
   //
   // The macrofilename and optionally cutfilename are assumed to contain
   // at least a method with the same name as the file.  The method
   // should return a value that can be automatically cast to
   // respectively a double and a boolean.
   //
   // Both methods will be executed in a context such that the
   // branch names can be used as C++ variables. This is
   // accomplished by generating a TTreeProxy (see MakeProxy)
   // and including the files in the proper location.
   //
   // If the branch name can not be used a proper C++ symbol name,
   // it will be modified as follow:
   //    - white spaces are removed
   //    - if the leading character is not a letter, an underscore is inserted
   //    - < and > are replace by underscores
   //    - * is replaced by st
   //    - & is replaced by rf
   //
   // If a cutfilename is specified, for each entry, we execute
   //   if (cutfilename()) htemp->Fill(macrofilename());
   // If no cutfilename is specified, for each entry we execute
   //   htemp(macrofilename());
   //
   // The default for the histogram are the same as for
   // TTreePlayer::DrawSelect

   if (!macrofilename || strlen(macrofilename)==0) return 0;

   TString aclicMode;
   TString arguments;
   TString io;
   TString realcutname;
   if (cutfilename && strlen(cutfilename))
      realcutname =  gSystem->SplitAclicMode(cutfilename, aclicMode, arguments, io);

   // we ignore the aclicMode for the cutfilename!
   TString realname = gSystem->SplitAclicMode(macrofilename, aclicMode, arguments, io);

   TString selname = wrapperPrefix;

   TTreeProxyGenerator gp(fTree,realname,realcutname,selname,option,3);

   selname = gp.GetFileName();
   if (aclicMode.Length()==0) {
      Warning("DrawScript","TTreeProxy does not work in interpreted mode yet. The script will be compiled.");
      aclicMode = "+";
   }
   selname.Append(aclicMode);

   Info("DrawScript","%s",Form("Will process tree/chain using %s",selname.Data()));
   Long64_t result = fTree->Process(selname,option,nentries,firstentry);
   fTree->SetNotify(0);

   // could delete the file selname+".h"
   // However this would remove the optimization of avoiding a useless
   // recompilation if the user ask for the same thing twice!

   return result;
}

//______________________________________________________________________________
Long64_t TTreePlayer::DrawSelect(const char *varexp0, const char *selection, Option_t *option,Long64_t nentries, Long64_t firstentry)
{
// Draw expression varexp for specified entries
// Returns -1 in case of error or number of selected events in case of success.
//
//  varexp is an expression of the general form
//   - "e1"           produces a 1-d histogram of expression "e1"
//   - "e1:e2"        produces a 2-d histogram (or profile) of "e1" versus "e2"
//   - "e1:e2:e3"     produces a 3-d scatter-plot of "e1" versus "e2" versus "e3"
//   - "e1:e2:e3:e4"  produces a 3-d scatter-plot of "e1" versus "e2" versus "e3"
//                    and "e4" mapped on the color number.
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
//  The expressions can use all the operations and build-in functions
//  supported by TFormula (See TFormula::Analyze), including free
//  standing function taking numerical arguments (TMath::Bessel).
//  In addition, you can call member functions taking numerical
//  arguments. For example:
//      - "TMath::BreitWigner(fPx,3,2)"
//      - "event.GetHistogram().GetXaxis().GetXmax()"
//  Note: You can only pass expression that depend on the TTree's data
//  to static functions and you can only call non-static member function
//  with 'fixed' parameters.
//
//  The selection is an expression with a combination of the columns.
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
//  selection1 returns a weight = 0 or 1
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
// Let assume, a leaf fMatrix, on the branch fEvent, which is a 3 by 3 array,
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
// TTree::Draw now also properly handles operations involving 2 or more arrays.
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
//  One can retrieve a pointer to this histogram with:
//    TH1F *htemp = (TH1F*)gPad->GetPrimitive("htemp");
//
//  If varexp0 contains >>hnew (following the variable(s) name(s),
//  the new histogram created is called hnew and it is kept in the current
//  directory (and also the current pad).
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
//     Retrieving  information about the tree itself.
//     ============================================
//
//  You can refer to the tree (or chain) containing the data by using the
//  string 'This'.
//  You can then could any TTree methods.  For example:
//     tree->Draw("This->GetReadEntry()");
//  will display the local entry numbers be read.
//     tree->Draw("This->GetUserInfo()->At(0)->GetName()");
//  will display the name of the first 'user info' object.
//
//     Special functions and variables
//     ===============================
//
//  Entry$:  A TTree::Draw formula can use the special variable Entry$
//  to access the entry number being read.  For example to draw every
//  other entry use:
//    tree.Draw("myvar","Entry$%2==0");
//
//  Entry$      : return the current entry number (== TTree::GetReadEntry())
//  LocalEntry$ : return the current entry number in the current tree of a chain (== GetTree()->GetReadEntry())
//  Entries$    : return the total number of entries (== TTree::GetEntries())
//  Length$     : return the total number of element of this formula for this
//              entry (==TTreeFormula::GetNdata())
//  Iteration$:   return the current iteration over this formula for this
//                 entry (i.e. varies from 0 to Length$).
//
//  Length$(formula): return the total number of element of the formula given as a
//                    parameter.
//  Sum$(formula): return the sum of the value of the elements of the formula given
//                    as a parameter.  For example the mean for all the elements in
//                    one entry can be calculated with:
//                Sum$(formula)/Length$(formula)
//  Min$(formula): return the minimun (within one TTree entry) of the value of the
//                    elements of the formula given as a parameter.
//  Max$(formula): return the maximum (within one TTree entry) of the value of the
//                    elements of the formula given as a parameter.
//  MinIf$(formula,condition)
//  MaxIf$(formula,condition): return the minimum (maximum) (within one TTree entry)
//                    of the value of the elements of the formula given as a parameter
//                    if they match the condition. If not element match the condition, the result is zero.  To avoid the
//                    the result is zero.  To avoid the consequent peak a zero, use the
//                    pattern:
//    tree->Draw("MinIf$(formula,condition)","condition");
//                    which will avoid calculation MinIf$ for the entries that have no match
//                    for the condition.
//
//  Alt$(primary,alternate) : return the value of "primary" if it is available
//                 for the current iteration otherwise return the value of "alternate".
//                 For example, with arr1[3] and arr2[2]
//    tree->Draw("arr1+Alt$(arr2,0)");
//                 will draw arr[0]+arr2[0] ; arr[1]+arr2[1] and arr[1]+0
//                 Or with a variable size array arr3
//    tree->Draw("Alt$(arr3[0],0)+Alt$(arr3[1],0)+Alt$(arr3[2],0)");
//                 will draw the sum arr3 for the index 0 to min(2,actual_size_of_arr3-1)
//                 As a comparison
//    tree->Draw("arr3[0]+arr3[1]+arr3[2]");
//                 will draw the sum arr3 for the index 0 to 2 only if the
//                 actual_size_of_arr3 is greater or equal to 3.
//                 Note that the array in 'primary' is flattened/linearized thus using
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
//  instead of a TH2F histogram by specifying option=prof or option=profs.
//  The option=prof is automatically selected in case of y:x>>pf
//  where pf is an existing TProfile histogram.
//
//     Making a 2D Profile histogram
//     ==========================
//  In case of a 3-Dim expression, one can generate a TProfile2D histogram
//  instead of a TH3F histogram by specifying option=prof or option=profs.
//  The option=prof is automatically selected in case of z:y:x>>pf
//  where pf is an existing TProfile2D histogram.
//
//     Making a 5-D plot with GL
//     =========================
//  When the option "gl5d" is specified and the dimension of the query is 5
//  a 5-d plot is created using GL, eg
//      T->Draw("x:y:z:u:w","","gl5d")
//
//     Making a parallel coordinates plot.
//     ===========================
//  In case of a 2-Dim or more expression with the option=para, one can generate
//  a parallel coordinates plot. With that option, the number of dimensions is
//  arbitrary. Giving more than 4 variables without the option=para or
//  option=candle or option=goff will produce an error.
//
//     Making a candle sticks chart.
//     ===========================
//  In case of a 2-Dim or more expression with the option=candle, one can generate
//  a candle sticks chart. With that option, the number of dimensions is
//  arbitrary. Giving more than 4 variables without the option=para or
//  option=candle or option=goff will produce an error.
//
//     Normalizing the ouput histogram to 1
//     ====================================
//  When option contains "norm" the output histogram is normalized to 1.
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
//    -GetSelectedRows()    //return the number of values accepted by the
//                          //selection expression. In case where no selection
//                          //was specified, returns the number of values processed.
//    -GetV1()              //returns a pointer to the double array of V1
//    -GetV2()              //returns a pointer to the double array of V2
//    -GetV3()              //returns a pointer to the double array of V3
//    -GetV4()              //returns a pointer to the double array of V4
//    -GetW()               //returns a pointer to the double array of Weights
//                          //where weight equal the result of the selection expression.
//   where V1,V2,V3 correspond to the expressions in
//   TTree::Draw("V1:V2:V3:V4",selection);
//   If the expression has more than 4 component use GetVal(index)
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
//    Important note: By default TTree::Draw creates the arrays obtained
//    with GetW, GetV1, GetV2, GetV3, GetV4, GetVal with a length corresponding 
//    to the parameter fEstimate.  The content will be the last
//            GetSelectedRows() % GetEstimate()
//    values calculated.
//    By default fEstimate=10000 and can be modified
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

   // Let's see if we have a filename as arguments instead of
   // a TTreeFormula expression.

   TString possibleFilename = varexp0;
   Ssiz_t dot_pos = possibleFilename.Last('.');
   if ( dot_pos != kNPOS
       && possibleFilename.Index("Alt$")<0 && possibleFilename.Index("Entries$")<0
       && possibleFilename.Index("Length$")<0  && possibleFilename.Index("Entry$")<0
       && possibleFilename.Index("LocalEntry$")<0
       && possibleFilename.Index("Min$")<0 && possibleFilename.Index("Max$")<0
       && possibleFilename.Index("MinIf$")<0 && possibleFilename.Index("MaxIf$")<0
       && possibleFilename.Index("Iteration$")<0 && possibleFilename.Index("Sum$")<0
       && possibleFilename.Index(">")<0 && possibleFilename.Index("<")<0
       && gSystem->IsFileInIncludePath(possibleFilename.Data())) {

      if (selection && strlen(selection) && !gSystem->IsFileInIncludePath(selection)) {
         Error("DrawSelect",
               "Drawing using a C++ file currently requires that both the expression and the selection are files\n\t\"%s\" is not a file",
               selection);
         return 0;
      }
      return DrawScript("generatedSel",varexp0,selection,option,nentries,firstentry);

   } else {
      possibleFilename = selection;
      if (possibleFilename.Index("Alt$")<0 && possibleFilename.Index("Entries$")<0
          && possibleFilename.Index("Length$")<0  && possibleFilename.Index("Entry$")<0
          && possibleFilename.Index("LocalEntry$")<0
          && possibleFilename.Index("Min$")<0 && possibleFilename.Index("Max$")<0
          && possibleFilename.Index("MinIf$")<0 && possibleFilename.Index("MaxIf$")<0
          && possibleFilename.Index("Iteration$")<0 && possibleFilename.Index("Sum$")<0
          && possibleFilename.Index(">")<0 && possibleFilename.Index("<")<0
          && gSystem->IsFileInIncludePath(possibleFilename.Data())) {

         Error("DrawSelect",
               "Drawing using a C++ file currently requires that both the expression and the selection are files\n\t\"%s\" is not a file",
               varexp0);
         return 0;
      }
   }

   Long64_t oldEstimate  = fTree->GetEstimate();
   TEventList *evlist  = fTree->GetEventList();
   TEntryList *elist = fTree->GetEntryList();
   if (evlist && elist){
      elist->SetBit(kCanDelete, kTRUE);
   }
   TNamed *cvarexp    = (TNamed*)fInput->FindObject("varexp");
   TNamed *cselection = (TNamed*)fInput->FindObject("selection");
   if (cvarexp) cvarexp->SetTitle(varexp0);
   if (cselection) cselection->SetTitle(selection);

   TString opt = option;
   opt.ToLower();
   Bool_t optpara   = kFALSE;
   Bool_t optcandle = kFALSE;
   Bool_t optgl5d   = kFALSE;
   Bool_t optnorm   = kFALSE;
   if (opt.Contains("norm")) {optnorm = kTRUE; opt.ReplaceAll("norm",""); opt.ReplaceAll(" ","");}
   if (opt.Contains("para")) optpara = kTRUE;
   if (opt.Contains("candle")) optcandle = kTRUE;
   if (opt.Contains("gl5d")) optgl5d = kTRUE;
   Bool_t pgl = gStyle->GetCanvasPreferGL();
   if (optgl5d) {
      fTree->SetEstimate(fTree->GetEntries());
      if (!gPad) {
         if (pgl == kFALSE) gStyle->SetCanvasPreferGL(kTRUE);
         gROOT->ProcessLineFast("new TCanvas();");
      }
   }


   // Do not process more than fMaxEntryLoop entries
   if (nentries > fTree->GetMaxEntryLoop()) nentries = fTree->GetMaxEntryLoop();

   // invoke the selector
   Long64_t nrows = Process(fSelector,option,nentries,firstentry);
   fSelectedRows = nrows;
   fDimension = fSelector->GetDimension();

   //*-* an Event List
   if (fDimension <= 0) {
      fTree->SetEstimate(oldEstimate);
      if (fSelector->GetCleanElist()) {
         // We are in the case where the input list was reset!
         fTree->SetEntryList(elist);
         delete fSelector->GetObject();
      }
      return nrows;
   }

   // Draw generated histogram
   Long64_t drawflag = fSelector->GetDrawFlag();
   Int_t action   = fSelector->GetAction();
   Bool_t draw = kFALSE;
   if (!drawflag && !opt.Contains("goff")) draw = kTRUE;
   if (!optcandle && !optpara) fHistogram = (TH1*)fSelector->GetObject();
   if (optnorm) {
      Double_t sumh= fHistogram->GetSumOfWeights();
      if (sumh != 0) fHistogram->Scale(1./sumh);
   }
   
   //if (!nrows && draw && drawflag && !opt.Contains("same")) {
   //   if (gPad) gPad->Clear();
   //   return 0;
   //}

   //*-*- 1-D distribution
   if (fDimension == 1) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (draw) fHistogram->Draw(opt.Data());

   //*-*- 2-D distribution
   } else if (fDimension == 2 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (action == 4) {
         if (draw) fHistogram->Draw(opt.Data());
      } else {
         Bool_t graph = kFALSE;
         Int_t l = opt.Length();
         if (l == 0 || opt == "same") graph = kTRUE;
         if (opt.Contains("p")     || opt.Contains("*")    || opt.Contains("l"))    graph = kTRUE;
         if (opt.Contains("surf")  || opt.Contains("lego") || opt.Contains("cont")) graph = kFALSE;
         if (opt.Contains("col")   || opt.Contains("hist") || opt.Contains("scat")) graph = kFALSE;
         if (!graph) {
            if (draw) fHistogram->Draw(opt.Data());
         } else {
            if (fSelector->GetOldHistogram() && draw) fHistogram->Draw(opt.Data());
         }
      }
   //*-*- 3-D distribution
   } else if (fDimension == 3 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Z");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar3()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (action == 23) {
         if (draw) fHistogram->Draw(opt.Data());
      } else {
         Int_t noscat = opt.Length();
         if (opt.Contains("same")) noscat -= 4;
         if (noscat) {
            if (draw) fHistogram->Draw(opt.Data());
         } else {
            if (fSelector->GetOldHistogram() && draw) fHistogram->Draw(opt.Data());
         }
      }
   //*-*- 4-D distribution
   } else if (fDimension == 4 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Z");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar3()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (draw) fHistogram->Draw(opt.Data());
      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *pms = (TObjArray*)fHistogram->GetListOfFunctions()->FindObject("polymarkers");
      for (Int_t col=0;col<ncolors;col++) {
         if (!pms) continue;
         TPolyMarker3D *pm3d = (TPolyMarker3D*)pms->UncheckedAt(col);
         if (draw) pm3d->Draw();
      }
   //*-*- Parallel Coordinates or Candle chart.
   } else if (optpara || optcandle) {
      if (draw) {
         TObject* para = fSelector->GetObject();
         TObject *enlist = gDirectory->FindObject("enlist");
         fTree->Draw(">>enlist",selection,"entrylist",nentries,firstentry);
         gROOT->ProcessLineFast(Form("TParallelCoord::SetEntryList((TParallelCoord*)0x%lx,(TEntryList*)0x%lx)",
                                     (ULong_t)para, (ULong_t)enlist));
      }
   //*-*- 5d with gl
   } else if (optgl5d) {
      gROOT->ProcessLineFast(Form("(new TGL5DDataSet((TTree *)0x%lx))->Draw(\"%s\");", (ULong_t)fTree, opt.Data()));
      gStyle->SetCanvasPreferGL(pgl);
   }

   if (fHistogram) fHistogram->ResetBit(TH1::kCanRebin);
   return fSelectedRows;
}

//______________________________________________________________________________
Int_t TTreePlayer::Fit(const char *formula ,const char *varexp, const char *selection,Option_t *option ,Option_t *goption,Long64_t nentries, Long64_t firstentry)
{
// Fit  a projected item(s) from a Tree.
// Returns -1 in case of error or number of selected events in case of success.
//
//  The formula is a TF1 expression.
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
//   Return status
//   =============
// The function returns the status of the histogram fit (see TH1::Fit)
// If no entries were selected, the function returns -1;
//   (i.e. fitResult is null if the fit is OK)

   Int_t nch = option ? strlen(option) + 10 : 10;
   char *opt = new char[nch];
   if (option) strlcpy(opt,option,nch-1);
   else        strlcpy(opt,"goff",5);
   
   Long64_t nsel = DrawSelect(varexp,selection,opt,nentries,firstentry);

   delete [] opt;
   Int_t fitResult = -1;

   if (fHistogram && nsel > 0) {
      fitResult = fHistogram->Fit(formula,option,goption);
   }
   return fitResult;
}

//______________________________________________________________________________
Long64_t TTreePlayer::GetEntries(const char *selection)
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

   TSelectorEntries s(selection);
   fTree->Process(&s);
   fTree->SetNotify(0);
   return s.GetSelectedRows();
}

//______________________________________________________________________________
Long64_t TTreePlayer::GetEntriesToProcess(Long64_t firstentry, Long64_t nentries) const
{
   // return the number of entries to be processed
   // this function checks that nentries is not bigger than the number
   // of entries in the Tree or in the associated TEventlist

   Long64_t lastentry = firstentry + nentries - 1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = fTree->GetEntriesFriend() - 1;
      nentries   = lastentry - firstentry + 1;
   }
   //TEventList *elist = fTree->GetEventList();
   //if (elist && elist->GetN() < nentries) nentries = elist->GetN();
   TEntryList *elist = fTree->GetEntryList();
   if (elist && elist->GetN() < nentries) nentries = elist->GetN();
   return nentries;
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
static TString R__GetBranchPointerName(TLeaf *leaf)
{
   // Return the name of the branch pointer needed by MakeClass/MakeSelector

   TLeaf *leafcount = leaf->GetLeafCount();
   TBranch *branch = leaf->GetBranch();

   TString branchname( branch->GetName() );

   if ( branch->GetNleaves() <= 1 ) {
       if (branch->IsA() != TBranchObject::Class()) {
         if (!leafcount) {
            TBranch *mother = branch->GetMother();
            const char* ltitle = leaf->GetTitle();
            if (mother && mother!=branch) {
               branchname = mother->GetName();
               if (branchname[branchname.Length()-1]!='.') {
                  branchname += ".";
               }
               if (strncmp(branchname.Data(),ltitle,branchname.Length())==0) {
                  branchname = "";
               }
            } else {
               branchname = "";
            }
            branchname += ltitle;
         }
      }
   }
   char *bname = (char*)branchname.Data();
   char *twodim = (char*)strstr(bname,"[");
   if (twodim) *twodim = 0;
   while (*bname) {
      if (*bname == '.') *bname='_';
      if (*bname == ',') *bname='_';
      if (*bname == ':') *bname='_';
      if (*bname == '<') *bname='_';
      if (*bname == '>') *bname='_';
      bname++;
   }
   return branchname;
}

//______________________________________________________________________________
Int_t TTreePlayer::MakeClass(const char *classname, const char *option)
{
// Generate skeleton analysis class for this Tree.
//
// The following files are produced: classname.h and classname.C
// If classname is 0, classname will be called "nameoftree.
//
// The generated code in classname.h includes the following:
//    - Identification of the original Tree and Input file name
//    - Definition of analysis class (data and functions)
//    - the following class functions:
//       - constructor (connecting by default the Tree file)
//       - GetEntry(Long64_t entry)
//       - Init(TTree *tree) to initialize a new TTree
//       - Show(Long64_t entry) to read and Dump entry
//
// The generated code in classname.C includes only the main
// analysis function Loop.
//
// To use this function:
//    - connect your Tree file (eg: TFile f("myfile.root");)
//    - T->MakeClass("MyClass");
// where T is the name of the Tree in file myfile.root
// and MyClass.h, MyClass.C the name of the files created by this function.
// In a ROOT session, you can do:
//    root > .L MyClass.C
//    root > MyClass t
//    root > t.GetEntry(12); // Fill t data members with entry number 12
//    root > t.Show();       // Show values of entry 12
//    root > t.Show(16);     // Read and show values of entry 16
//    root > t.Loop();       // Loop on all entries
//
//  NOTE: Do not use the code generated for one Tree in case of a TChain.
//        Maximum dimensions calculated on the basis of one TTree only
//        might be too small when processing all the TTrees in one TChain.
//        Instead of myTree.MakeClass(..,  use myChain.MakeClass(..

   TString opt = option;
   opt.ToLower();

   // Connect output files
   if (!classname) classname = fTree->GetName();

   TString thead;
   thead.Form("%s.h", classname);
   FILE *fp = fopen(thead, "w");
   if (!fp) {
      Error("MakeClass","cannot open output file %s", thead.Data());
      return 3;
   }
   TString tcimp;
   tcimp.Form("%s.C", classname);
   FILE *fpc = fopen(tcimp, "w");
   if (!fpc) {
      Error("MakeClass","cannot open output file %s", tcimp.Data());
      fclose(fp);
      return 3;
   }
   TString treefile;
   if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile()) {
      treefile = fTree->GetDirectory()->GetFile()->GetName();
   } else {
      treefile = "Memory Directory";
   }
   // In the case of a chain, the GetDirectory information usually does
   // pertain to the Chain itself but to the currently loaded tree.
   // So we can not rely on it.
   Bool_t ischain = fTree->InheritsFrom(TChain::Class());
   Bool_t isHbook = fTree->InheritsFrom("THbookTree");
   if (isHbook)
      treefile = fTree->GetTitle();

//======================Generate classname.h=====================
   // Print header
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;
   TDatime td;
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"// This class has been automatically generated on\n");
   fprintf(fp,"// %s by ROOT version %s\n",td.AsString(),gROOT->GetVersion());
   if (!ischain) {
      fprintf(fp,"// from TTree %s/%s\n",fTree->GetName(),fTree->GetTitle());
      fprintf(fp,"// found on file: %s\n",treefile.Data());
   } else {
      fprintf(fp,"// from TChain %s/%s\n",fTree->GetName(),fTree->GetTitle());
   }
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"\n");
   fprintf(fp,"#ifndef %s_h\n",classname);
   fprintf(fp,"#define %s_h\n",classname);
   fprintf(fp,"\n");
   fprintf(fp,"#include <TROOT.h>\n");
   fprintf(fp,"#include <TChain.h>\n");
   fprintf(fp,"#include <TFile.h>\n");
   if (isHbook) fprintf(fp,"#include <THbookFile.h>\n");
   if (opt.Contains("selector")) fprintf(fp,"#include <TSelector.h>\n");

// First loop on all leaves to generate dimension declarations
   Int_t len, lenb,l;
   char blen[1024];
   char *bname;
   Int_t *leaflen = new Int_t[nleaves];
   TObjArray *leafs = new TObjArray(nleaves);
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      leafs->AddAt(new TObjString(leaf->GetName()),l);
      leaflen[l] = leaf->GetMaximum();
   }
   if (ischain) {
      // In case of a chain, one must find the maximum dimension of each leaf
      // One must be careful and not assume that all Trees in the chain
      // have the same leaves and in the same order!
      TChain *chain = (TChain*)fTree;
      Int_t ntrees = chain->GetNtrees();
      for (Int_t file=0;file<ntrees;file++) {
         Long64_t first = chain->GetTreeOffset()[file];
         chain->LoadTree(first);
         for (l=0;l<nleaves;l++) {
            TObjString *obj = (TObjString*)leafs->At(l);
            TLeaf *leaf = chain->GetLeaf(obj->GetName());
            if (leaf) {
               leaflen[l] = TMath::Max(leaflen[l],leaf->GetMaximum());
            }
         }
      }
      chain->LoadTree(0);
   }

   leaves = fTree->GetListOfLeaves();
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      strlcpy(blen,leaf->GetName(),sizeof(blen)); 
      bname = &blen[0];
      while (*bname) {
         if (*bname == '.') *bname='_';
         if (*bname == ',') *bname='_';
         if (*bname == ':') *bname='_';
         if (*bname == '<') *bname='_';
         if (*bname == '>') *bname='_';
         bname++;
      }
      lenb = strlen(blen);
      if (blen[lenb-1] == '_') {
         blen[lenb-1] = 0;
         len = leaflen[l];
         if (len <= 0) len = 1;
         fprintf(fp,"   const Int_t kMax%s = %d;\n",blen,len);
      }
   }
   delete [] leaflen;
   leafs->Delete();
   delete leafs;

// second loop on all leaves to generate type declarations
   fprintf(fp,"\n");
   if (opt.Contains("selector")) {
      fprintf(fp,"class %s : public TSelector {\n",classname);
      fprintf(fp,"public :\n");
      fprintf(fp,"   TTree          *fChain;   //!pointer to the analyzed TTree or TChain\n");
   } else {
      fprintf(fp,"class %s {\n",classname);
      fprintf(fp,"public :\n");
      fprintf(fp,"   TTree          *fChain;   //!pointer to the analyzed TTree or TChain\n");
      fprintf(fp,"   Int_t           fCurrent; //!current Tree number in a TChain\n");
   }
   fprintf(fp,"\n   // Declaration of leaf types\n");
   TLeaf *leafcount;
   TLeafObject *leafobj;
   TBranchElement *bre=0;
   const char *headOK  = "   ";
   const char *headcom = " //";
   const char *head;
   char branchname[1024];
   char aprefix[1024];
   TObjArray branches(100);
   TObjArray mustInit(100);
   TObjArray mustInitArr(100);
   mustInitArr.SetOwner(kFALSE);
   Int_t *leafStatus = new Int_t[nleaves];
   for (l=0;l<nleaves;l++) {
      Int_t kmax = 0;
      head = headOK;
      leafStatus[l] = 0;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen(); if (len<=0) len = 1;
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      branchname[0] = 0;
      strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
      strlcpy(aprefix,branch->GetName(),sizeof(aprefix)); 
      if (!branches.FindObject(branch)) branches.Add(branch);
      else leafStatus[l] = 1;
      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strlcat(branchname,".",sizeof(branchname));
         strlcat(branchname,leaf->GetTitle(),sizeof(branchname));
         if (leafcount) {
            // remove any dimension in title
            char *dim =  (char*)strstr(branchname,"["); if (dim) dim[0] = 0;
         }
      } else {
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
      }
      char *twodim = (char*)strstr(leaf->GetTitle(),"][");
      bname = branchname;
      while (*bname) {
         if (*bname == '.') *bname='_';
         if (*bname == ',') *bname='_';
         if (*bname == ':') *bname='_';
         if (*bname == '<') *bname='_';
         if (*bname == '>') *bname='_';
         bname++;
      }
      if (branch->IsA() == TBranchObject::Class()) {
         if (branch->GetListOfBranches()->GetEntriesFast()) {leafStatus[l] = 1; continue;}
         leafobj = (TLeafObject*)leaf;
         if (!leafobj->GetClass()) {leafStatus[l] = 1; head = headcom;}
         fprintf(fp,"%s%-15s *%s;\n",head,leafobj->GetTypeName(), leafobj->GetName());
         if (leafStatus[l] == 0) mustInit.Add(leafobj);
         continue;
      }
      if (leafcount) {
         len = leafcount->GetMaximum();
         if (len<=0) len = 1;
         strlcpy(blen,leafcount->GetName(),sizeof(blen)); 
         bname = &blen[0];
         while (*bname) {
            if (*bname == '.') *bname='_';
            if (*bname == ',') *bname='_';
            if (*bname == ':') *bname='_';
            if (*bname == '<') *bname='_';
            if (*bname == '>') *bname='_';
            bname++;
         }
         lenb = strlen(blen);
         if (blen[lenb-1] == '_') {blen[lenb-1] = 0; kmax = 1;}
         else                     snprintf(blen,sizeof(blen),"%d",len);
      }
      if (branch->IsA() == TBranchElement::Class()) {
         bre = (TBranchElement*)branch;
         if (bre->GetType() != 3 && bre->GetType() != 4
             && bre->GetStreamerType() <= 0 && bre->GetListOfBranches()->GetEntriesFast()) {
            leafStatus[l] = 0;
         }
         if (bre->GetType() == 3 || bre->GetType() == 4) {
            fprintf(fp,"   %-15s %s_;\n","Int_t", branchname);
            continue;
         }
         if (bre->IsBranchFolder()) {
            fprintf(fp,"   %-15s *%s;\n",bre->GetClassName(), branchname);
            mustInit.Add(bre);
            continue;
         } else {
            if (branch->GetListOfBranches()->GetEntriesFast()) {leafStatus[l] = 1;}
         }
         if (bre->GetStreamerType() < 0) {
            if (branch->GetListOfBranches()->GetEntriesFast()) {
               fprintf(fp,"%s%-15s *%s;\n",headcom,bre->GetClassName(), branchname);
            } else {
               fprintf(fp,"%s%-15s *%s;\n",head,bre->GetClassName(), branchname);
               mustInit.Add(bre);
            }
            continue;
         }
         if (bre->GetStreamerType() == 0) {
            if (!TClass::GetClass(bre->GetClassName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
            fprintf(fp,"%s%-15s *%s;\n",head,bre->GetClassName(), branchname);
            if (leafStatus[l] == 0) mustInit.Add(bre);
            continue;
         }
         if (bre->GetStreamerType() > 60) {
            TClass *cle = TClass::GetClass(bre->GetClassName());
            if (!cle) {leafStatus[l] = 1; continue;}
            if (bre->GetStreamerType() == 66) leafStatus[l] = 0;
            char brename[256];
            strlcpy(brename,bre->GetName(),255); 
            char *bren = brename;
            char *adot = strrchr(bren,'.');
            if (adot) bren = adot+1;
            char *brack = strchr(bren,'[');
            if (brack) *brack = 0;
            TStreamerElement *elem = (TStreamerElement*)cle->GetStreamerInfo()->GetElements()->FindObject(bren);
            if (elem) {
               if (elem->IsA() == TStreamerBase::Class()) {leafStatus[l] = 1; continue;}
               if (!TClass::GetClass(elem->GetTypeName())) {leafStatus[l] = 1; continue;}
               if (!TClass::GetClass(elem->GetTypeName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
               if (leafcount) fprintf(fp,"%s%-15s %s[kMax%s];\n",head,elem->GetTypeName(), branchname,blen);
               else           fprintf(fp,"%s%-15s %s;\n",head,elem->GetTypeName(), branchname);
            } else {
               if (!TClass::GetClass(bre->GetClassName())->GetClassInfo()) {leafStatus[l] = 1; head = headcom;}
               fprintf(fp,"%s%-15s %s;\n",head,bre->GetClassName(), branchname);
            }
            continue;
         }
      }
      if (strlen(leaf->GetTypeName()) == 0) {leafStatus[l] = 1; continue;}
      if (leafcount) {
         //len = leafcount->GetMaximum();
         //strlcpy(blen,leafcount->GetName(),sizeof(blen));
         //bname = &blen[0];
         //while (*bname) {if (*bname == '.') *bname='_'; bname++;}
         //lenb = strlen(blen);
         //Int_t kmax = 0;
         //if (blen[lenb-1] == '_') {blen[lenb-1] = 0; kmax = 1;}
         //else                     sprintf(blen,"%d",len);

         const char *stars = " ";
         if (bre && bre->GetBranchCount2()) {
            stars = "*";
         }
         // Dimensions can be in the branchname for a split Object with a fix length C array.
         // Theses dimensions HAVE TO be placed after the dimension explicited by leafcount
         TString dimensions;
         char *dimInName = (char*) strstr(branchname,"[");
         if ( twodim || dimInName ) {
            if (dimInName) {
               dimensions = dimInName; 
               dimInName[0] = 0; // terminate branchname before the array dimensions.
            }
            if (twodim) dimensions += (char*)(twodim+1);
         }
         const char* leafcountName = leafcount->GetName();
         char b2len[1024];
         if (bre && bre->GetBranchCount2()) {
            TLeaf * l2 = (TLeaf*)bre->GetBranchCount2()->GetListOfLeaves()->At(0);
            strlcpy(b2len,l2->GetName(),sizeof(b2len)); 
            bname = &b2len[0];
            while (*bname) {
               if (*bname == '.') *bname='_';
               if (*bname == ',') *bname='_';
               if (*bname == ':') *bname='_';
               if (*bname == '<') *bname='_';
               if (*bname == '>') *bname='_';
               bname++;
            }
            leafcountName = b2len;
         }
         if (dimensions.Length()) {
            if (kmax) fprintf(fp,"   %-14s %s%s[kMax%s]%s;   //[%s]\n",leaf->GetTypeName(), stars,
                              branchname,blen,dimensions.Data(),leafcountName);
            else      fprintf(fp,"   %-14s %s%s[%d]%s;   //[%s]\n",leaf->GetTypeName(), stars,
                              branchname,len,dimensions.Data(),leafcountName);
         } else {
            if (kmax) fprintf(fp,"   %-14s %s%s[kMax%s];   //[%s]\n",leaf->GetTypeName(), stars, branchname,blen,leafcountName);
            else      fprintf(fp,"   %-14s %s%s[%d];   //[%s]\n",leaf->GetTypeName(), stars, branchname,len,leafcountName);
         }
         if (stars[0]=='*') {
            TNamed *n;
            if (kmax) n = new TNamed(branchname, Form("kMax%s",blen));
            else n = new TNamed(branchname, Form("%d",len));
            mustInitArr.Add(n);
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
   fprintf(fp,"   // List of branches\n");
   for (l=0;l<nleaves;l++) {
      if (leafStatus[l]) continue;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      fprintf(fp,"   TBranch        *b_%s;   //!\n",R__GetBranchPointerName(leaf).Data());
   }

// generate class member functions prototypes
   if (opt.Contains("selector")) {
      fprintf(fp,"\n");
      fprintf(fp,"   %s(TTree * /*tree*/ =0) { }\n",classname) ;
      fprintf(fp,"   virtual ~%s() { }\n",classname);
      fprintf(fp,"   virtual Int_t   Version() const { return 2; }\n");
      fprintf(fp,"   virtual void    Begin(TTree *tree);\n");
      fprintf(fp,"   virtual void    SlaveBegin(TTree *tree);\n");
      fprintf(fp,"   virtual void    Init(TTree *tree);\n");
      fprintf(fp,"   virtual Bool_t  Notify();\n");
      fprintf(fp,"   virtual Bool_t  Process(Long64_t entry);\n");
      fprintf(fp,"   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }\n");
      fprintf(fp,"   virtual void    SetOption(const char *option) { fOption = option; }\n");
      fprintf(fp,"   virtual void    SetObject(TObject *obj) { fObject = obj; }\n");
      fprintf(fp,"   virtual void    SetInputList(TList *input) { fInput = input; }\n");
      fprintf(fp,"   virtual TList  *GetOutputList() const { return fOutput; }\n");
      fprintf(fp,"   virtual void    SlaveTerminate();\n");
      fprintf(fp,"   virtual void    Terminate();\n\n");
      fprintf(fp,"   ClassDef(%s,0);\n",classname);
      fprintf(fp,"};\n");
      fprintf(fp,"\n");
      fprintf(fp,"#endif\n");
      fprintf(fp,"\n");
   } else {
      fprintf(fp,"\n");
      fprintf(fp,"   %s(TTree *tree=0);\n",classname);
      fprintf(fp,"   virtual ~%s();\n",classname);
      fprintf(fp,"   virtual Int_t    Cut(Long64_t entry);\n");
      fprintf(fp,"   virtual Int_t    GetEntry(Long64_t entry);\n");
      fprintf(fp,"   virtual Long64_t LoadTree(Long64_t entry);\n");
      fprintf(fp,"   virtual void     Init(TTree *tree);\n");
      fprintf(fp,"   virtual void     Loop();\n");
      fprintf(fp,"   virtual Bool_t   Notify();\n");
      fprintf(fp,"   virtual void     Show(Long64_t entry = -1);\n");
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
      if (isHbook) {
         fprintf(fp,"      THbookFile *f = (THbookFile*)gROOT->GetListOfBrowsables()->FindObject(\"%s\");\n",
                    treefile.Data());
         fprintf(fp,"      if (!f) {\n");
         fprintf(fp,"         f = new THbookFile(\"%s\");\n",treefile.Data());
         fprintf(fp,"      }\n");
         Int_t hid;
         sscanf(fTree->GetName(),"h%d",&hid);
         fprintf(fp,"      tree = (TTree*)f->Get(%d);\n\n",hid);
      } else {
         fprintf(fp,"      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(\"%s\");\n",treefile.Data());
         fprintf(fp,"      if (!f || !f->IsOpen()) {\n");
         fprintf(fp,"         f = new TFile(\"%s\");\n",treefile.Data());
         fprintf(fp,"      }\n");
         if (fTree->GetDirectory() != fTree->GetCurrentFile()) {
            fprintf(fp,"      TDirectory * dir = (TDirectory*)f->Get(\"%s\");\n",fTree->GetDirectory()->GetPath());
            fprintf(fp,"      dir->GetObject(\"%s\",tree);\n\n",fTree->GetName());
         } else {
            fprintf(fp,"      f->GetObject(\"%s\",tree);\n\n",fTree->GetName());
         }
      }
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
      if (isHbook) {
         //fprintf(fp,"   delete fChain->GetCurrentFile();\n");
      } else {
         fprintf(fp,"   delete fChain->GetCurrentFile();\n");
      }
      fprintf(fp,"}\n");
      fprintf(fp,"\n");
   }
// generate code for class member function GetEntry()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Int_t %s::GetEntry(Long64_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Read contents of entry.\n");

      fprintf(fp,"   if (!fChain) return 0;\n");
      fprintf(fp,"   return fChain->GetEntry(entry);\n");
      fprintf(fp,"}\n");
   }
// generate code for class member function LoadTree()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Long64_t %s::LoadTree(Long64_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Set the environment to read one entry\n");
      fprintf(fp,"   if (!fChain) return -5;\n");
      fprintf(fp,"   Long64_t centry = fChain->LoadTree(entry);\n");
      fprintf(fp,"   if (centry < 0) return centry;\n");
      fprintf(fp,"   if (fChain->GetTreeNumber() != fCurrent) {\n");
      fprintf(fp,"      fCurrent = fChain->GetTreeNumber();\n");
      fprintf(fp,"      Notify();\n");
      fprintf(fp,"   }\n");
      fprintf(fp,"   return centry;\n");
      fprintf(fp,"}\n");
      fprintf(fp,"\n");
   }

// generate code for class member function Init(), first pass = get branch pointer
   fprintf(fp,"void %s::Init(TTree *tree)\n",classname);
   fprintf(fp,"{\n");
   fprintf(fp,"   // The Init() function is called when the selector needs to initialize\n"
              "   // a new tree or chain. Typically here the branch addresses and branch\n"
              "   // pointers of the tree will be set.\n"
              "   // It is normally not necessary to make changes to the generated\n"
              "   // code, but the routine can be extended by the user if needed.\n"
              "   // Init() will be called many times when running on PROOF\n"
              "   // (once per file to be processed).\n\n");
   if (mustInit.Last()) {
      TIter next(&mustInit);
      TObject *obj;
      fprintf(fp,"   // Set object pointer\n");
      while( (obj = next()) ) {
         if (obj->InheritsFrom(TBranch::Class())) {
            strlcpy(branchname,((TBranch*)obj)->GetName(),sizeof(branchname));
         } else if (obj->InheritsFrom(TLeaf::Class())) {
            strlcpy(branchname,((TLeaf*)obj)->GetName(),sizeof(branchname)); 
         }
         branchname[1023]=0;
         bname = branchname;
         while (*bname) {
            if (*bname == '.') *bname='_';
            if (*bname == ',') *bname='_';
            if (*bname == ':') *bname='_';
            if (*bname == '<') *bname='_';
            if (*bname == '>') *bname='_';
            bname++;
         }
         fprintf(fp,"   %s = 0;\n",branchname );
      }
   }
   if (mustInitArr.Last()) {
      TIter next(&mustInitArr);
      TNamed *info;
      fprintf(fp,"   // Set array pointer\n");
      while( (info = (TNamed*)next()) ) {
         fprintf(fp,"   for(int i=0; i<%s; ++i) %s[i] = 0;\n",info->GetTitle(),info->GetName());
      }
      fprintf(fp,"\n");
   }
   fprintf(fp,"   // Set branch addresses and branch pointers\n");
   fprintf(fp,"   if (!tree) return;\n");
   fprintf(fp,"   fChain = tree;\n");
   if (!opt.Contains("selector")) fprintf(fp,"   fCurrent = -1;\n");
   fprintf(fp,"   fChain->SetMakeClass(1);\n");
   fprintf(fp,"\n");
   for (l=0;l<nleaves;l++) {
      if (leafStatus[l]) continue;
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      strlcpy(aprefix,branch->GetName(),sizeof(aprefix)); 

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
         strlcat(branchname,".",sizeof(branchname));
         strlcat(branchname,leaf->GetTitle(),sizeof(branchname));
         if (leafcount) {
            // remove any dimension in title
            char *dim =  (char*)strstr(branchname,"["); if (dim) dim[0] = 0;
         }
      } else {
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
         if (branch->IsA() == TBranchElement::Class()) {
            bre = (TBranchElement*)branch;
            if (bre->GetType() == 3 || bre->GetType()==4) strlcat(branchname,"_",sizeof(branchname));
         }
      }
      bname = branchname;
      char *brak = strstr(branchname,"[");     if (brak) *brak = 0;
      char *twodim = (char*)strstr(bname,"["); if (twodim) *twodim = 0;
      while (*bname) {
         if (*bname == '.') *bname='_';
         if (*bname == ',') *bname='_';
         if (*bname == ':') *bname='_';
         if (*bname == '<') *bname='_';
         if (*bname == '>') *bname='_';
         bname++;
      }
      if (branch->IsA() == TBranchObject::Class()) {
         if (branch->GetListOfBranches()->GetEntriesFast()) {
            fprintf(fp,"   fChain->SetBranchAddress(\"%s\",(void*)-1,&b_%s);\n",branch->GetName(),R__GetBranchPointerName(leaf).Data());
            continue;
         }
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
      }
      if (branch->IsA() == TBranchElement::Class()) {
         if (((TBranchElement*)branch)->GetType() == 3) len =1;
         if (((TBranchElement*)branch)->GetType() == 4) len =1;
      }
      if (leafcount) len = leafcount->GetMaximum()+1;
      if (len > 1) fprintf(fp,"   fChain->SetBranchAddress(\"%s\", %s, &b_%s);\n",
                           branch->GetName(), branchname, R__GetBranchPointerName(leaf).Data());
      else         fprintf(fp,"   fChain->SetBranchAddress(\"%s\", &%s, &b_%s);\n",
                           branch->GetName(), branchname, R__GetBranchPointerName(leaf).Data());
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
   fprintf(fp,"   // The Notify() function is called when a new file is opened. This\n"
              "   // can be either for a new TTree in a TChain or when when a new TTree\n"
              "   // is started when using PROOF. It is normally not necessary to make changes\n"
              "   // to the generated code, but the routine can be extended by the\n"
              "   // user if needed. The return value is currently not used.\n\n");
   fprintf(fp,"   return kTRUE;\n");
   fprintf(fp,"}\n");
   fprintf(fp,"\n");

// generate code for class member function Show()
   if (!opt.Contains("selector")) {
      fprintf(fp,"void %s::Show(Long64_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// Print contents of entry.\n");
      fprintf(fp,"// If entry is not specified, print current entry\n");

      fprintf(fp,"   if (!fChain) return;\n");
      fprintf(fp,"   fChain->Show(entry);\n");
      fprintf(fp,"}\n");
   }
// generate code for class member function Cut()
   if (!opt.Contains("selector")) {
      fprintf(fp,"Int_t %s::Cut(Long64_t entry)\n",classname);
      fprintf(fp,"{\n");
      fprintf(fp,"// This function may be called from Loop.\n");
      fprintf(fp,"// returns  1 if entry is accepted.\n");
      fprintf(fp,"// returns -1 otherwise.\n");

      fprintf(fp,"   return 1;\n");
      fprintf(fp,"}\n");
   }
   fprintf(fp,"#endif // #ifdef %s_cxx\n",classname);

//======================Generate classname.C=====================
   if (!opt.Contains("selector")) {
      // generate code for class member function Loop()
      fprintf(fpc,"#define %s_cxx\n",classname);
      fprintf(fpc,"#include \"%s\"\n",thead.Data());
      fprintf(fpc,"#include <TH2.h>\n");
      fprintf(fpc,"#include <TStyle.h>\n");
      fprintf(fpc,"#include <TCanvas.h>\n");
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
      fprintf(fpc,"\n//     This is the loop skeleton where:\n");
      fprintf(fpc,"//    jentry is the global entry number in the chain\n");
      fprintf(fpc,"//    ientry is the entry number in the current Tree\n");
      fprintf(fpc,"//  Note that the argument to GetEntry must be:\n");
      fprintf(fpc,"//    jentry for TChain::GetEntry\n");
      fprintf(fpc,"//    ientry for TTree::GetEntry and TBranch::GetEntry\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"//       To read only selected branches, Insert statements like:\n");
      fprintf(fpc,"// METHOD1:\n");
      fprintf(fpc,"//    fChain->SetBranchStatus(\"*\",0);  // disable all branches\n");
      fprintf(fpc,"//    fChain->SetBranchStatus(\"branchname\",1);  // activate branchname\n");
      fprintf(fpc,"// METHOD2: replace line\n");
      fprintf(fpc,"//    fChain->GetEntry(jentry);       //read all branches\n");
      fprintf(fpc,"//by  b_branchname->GetEntry(ientry); //read only this branch\n");
      fprintf(fpc,"   if (fChain == 0) return;\n");
      fprintf(fpc,"\n   Long64_t nentries = fChain->GetEntriesFast();\n");
      fprintf(fpc,"\n   Long64_t nbytes = 0, nb = 0;\n");
      fprintf(fpc,"   for (Long64_t jentry=0; jentry<nentries;jentry++) {\n");
      fprintf(fpc,"      Long64_t ientry = LoadTree(jentry);\n");
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
      fprintf(fpc,"// by the ROOT utility TTree::MakeSelector(). This class is derived\n");
      fprintf(fpc,"// from the ROOT class TSelector. For more information on the TSelector\n"
                  "// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.\n\n");
      fprintf(fpc,"// The following methods are defined in this file:\n");
      fprintf(fpc,"//    Begin():        called every time a loop on the tree starts,\n");
      fprintf(fpc,"//                    a convenient place to create your histograms.\n");
      fprintf(fpc,"//    SlaveBegin():   called after Begin(), when on PROOF called only on the\n"
                  "//                    slave servers.\n");
      fprintf(fpc,"//    Process():      called for each event, in this function you decide what\n");
      fprintf(fpc,"//                    to read and fill your histograms.\n");
      fprintf(fpc,"//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF\n"
                  "//                    called only on the slave servers.\n");
      fprintf(fpc,"//    Terminate():    called at the end of the loop on the tree,\n");
      fprintf(fpc,"//                    a convenient place to draw/fit your histograms.\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// To use this file, try the following session on your Tree T:\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// Root > T->Process(\"%s.C\")\n",classname);
      fprintf(fpc,"// Root > T->Process(\"%s.C\",\"some options\")\n",classname);
      fprintf(fpc,"// Root > T->Process(\"%s.C+\")\n",classname);
      fprintf(fpc,"//\n\n");
      fprintf(fpc,"#include \"%s\"\n",thead.Data());
      fprintf(fpc,"#include <TH2.h>\n");
      fprintf(fpc,"#include <TStyle.h>\n");
      fprintf(fpc,"\n");
      // generate code for class member function Begin
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Begin(TTree * /*tree*/)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Begin() function is called at the start of the query.\n");
      fprintf(fpc,"   // When running with PROOF Begin() is only called on the client.\n");
      fprintf(fpc,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   TString option = GetOption();\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function SlaveBegin
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::SlaveBegin(TTree * /*tree*/)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The SlaveBegin() function is called after the Begin() function.\n");
      fprintf(fpc,"   // When running with PROOF SlaveBegin() is called on each slave server.\n");
      fprintf(fpc,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   TString option = GetOption();\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function Process
      fprintf(fpc,"\n");
      fprintf(fpc,"Bool_t %s::Process(Long64_t entry)\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Process() function is called for each entry in the tree (or possibly\n"
                  "   // keyed object in the case of PROOF) to be processed. The entry argument\n"
                  "   // specifies which entry in the currently loaded tree is to be processed.\n"
                  "   // It can be passed to either %s::GetEntry() or TBranch::GetEntry()\n"
                  "   // to read either all or the required parts of the data. When processing\n"
                  "   // keyed objects with PROOF, the object is already loaded and is available\n"
                  "   // via the fObject pointer.\n"
                  "   //\n"
                  "   // This function should contain the \"body\" of the analysis. It can contain\n"
                  "   // simple or elaborate selection criteria, run algorithms on the data\n"
                  "   // of the event and typically fill histograms.\n"
                  "   //\n"
                  "   // The processing can be stopped by calling Abort().\n"
                  "   //\n"
                  "   // Use fStatus to set the return value of TTree::Process().\n"
                  "   //\n"
                  "   // The return value is currently not used.\n\n", classname);
      fprintf(fpc,"\n");
      fprintf(fpc,"   return kTRUE;\n");
      fprintf(fpc,"}\n");
      // generate code for class member function SlaveTerminate
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::SlaveTerminate()\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The SlaveTerminate() function is called after all entries or objects\n"
                  "   // have been processed. When running with PROOF SlaveTerminate() is called\n"
                  "   // on each slave server.");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function Terminate
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Terminate()\n",classname);
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Terminate() function is the last function to be called during\n"
                  "   // a query. It always runs on the client, it can be used to present\n"
                  "   // the results graphically or save the results to file.");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
   }
   Info("MakeClass","Files: %s and %s generated from TTree: %s",thead.Data(),tcimp.Data(),fTree->GetName());
   delete [] leafStatus;
   fclose(fp);
   fclose(fpc);

   return 0;
}


//______________________________________________________________________________
Int_t TTreePlayer::MakeCode(const char *filename)
{
// Generate skeleton function for this Tree
//
// The function code is written on filename.
// If filename is 0, filename will be called nameoftree.C
//
// The generated code includes the following:
//    - Identification of the original Tree and Input file name
//    - Connection of the Tree file
//    - Declaration of Tree variables
//    - Setting of branches addresses
//    - A skeleton for the entry loop
//
// To use this function:
//    - connect your Tree file (eg: TFile f("myfile.root");)
//    - T->MakeCode("anal.C");
// where T is the name of the Tree in file myfile.root
// and anal.C the name of the file created by this function.
//
// NOTE: Since the implementation of this function, a new and better
//       function TTree::MakeClass() has been developed.

// Connect output file
   TString tfile;
   if (filename)
      tfile = filename;
   else
      tfile.Form("%s.C", fTree->GetName());
   FILE *fp = fopen(tfile, "w");
   if (!fp) {
      Error("MakeCode","cannot open output file %s", tfile.Data());
      return 3;
   }
   TString treefile;
   if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile()) {
      treefile = fTree->GetDirectory()->GetFile()->GetName();
   } else {
      treefile = "Memory Directory";
   }
   // In the case of a chain, the GetDirectory information usually does
   // pertain to the Chain itself but to the currently loaded tree.
   // So we can not rely on it.
   Bool_t ischain = fTree->InheritsFrom(TChain::Class());

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
      fprintf(fp,"//   found on file: %s\n",treefile.Data());
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
   fprintf(fp,"   TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject(\"%s\");\n",treefile.Data());
   fprintf(fp,"   if (!f) {\n");
   fprintf(fp,"      f = new TFile(\"%s\");\n",treefile.Data());
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
   char branchname[1024];
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();
      if (branch->GetListOfBranches()->GetEntriesFast() > 0) continue;

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
         strlcat(branchname,".",sizeof(branchname));
         strlcat(branchname,leaf->GetTitle(),sizeof(branchname));
         if (leafcount) {
            // remove any dimension in title
            char *dim =  (char*)strstr(branchname,"[");
            dim[0] = 0;
         }
      } else {
         if (leafcount) strlcpy(branchname,branch->GetName(),sizeof(branchname));
         else           strlcpy(branchname,leaf->GetTitle(),sizeof(branchname));
      }
      char *twodim = (char*)strstr(leaf->GetTitle(),"][");
      bname = branchname;
      while (*bname) {
         if (*bname == '.') *bname='_';
         if (*bname == ',') *bname='_';
         if (*bname == ':') *bname='_';
         if (*bname == '<') *bname='_';
         if (*bname == '>') *bname='_';
         bname++;
      }
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
         TString dimensions;
         if ( twodim || dimInName ) {
            if (dimInName) {
               dimensions = dimInName; 
               dimInName[0] = 0; // terminate branchname before the array dimensions.
            }
            if (twodim) dimensions += (char*)(twodim+1);
         }
         if (dimensions.Length()) {
            fprintf(fp,"   %-15s %s[%d]%s;\n",leaf->GetTypeName(), branchname,len,dimensions.Data());
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
   fprintf(fp,"\n   // Set branch addresses.\n");
   for (l=0;l<nleaves;l++) {
      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
      len = leaf->GetLen();
      leafcount =leaf->GetLeafCount();
      TBranch *branch = leaf->GetBranch();

      if ( branch->GetNleaves() > 1) {
         // More than one leaf for the branch we need to distinguish them
         strlcpy(branchname,branch->GetName(),sizeof(branchname)); 
         strlcat(branchname,".",sizeof(branchname));
         strlcat(branchname,leaf->GetTitle(),sizeof(branchname));
         if (leafcount) {
            // remove any dimension in title
            char *dim =  (char*)strstr(branchname,"[");
            dim[0] = 0;
         }
      } else {
         if (leafcount) strlcpy(branchname,branch->GetName(),sizeof(branchname));
         else           strlcpy(branchname,leaf->GetTitle(),sizeof(branchname));
      }
      bname = branchname;
      while (*bname) {
         if (*bname == '.') *bname='_';
         if (*bname == ',') *bname='_';
         if (*bname == ':') *bname='_';
         if (*bname == '<') *bname='_';
         if (*bname == '>') *bname='_';
         bname++;
      }
      char *brak = strstr(branchname,"[");
      if (brak) *brak = 0;
      head = headOK;
      if (branch->IsA() == TBranchObject::Class()) {
         strlcpy(branchname,branch->GetName(),sizeof(branchname));
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
   fprintf(fp,"\n   Long64_t nentries = %s->GetEntries();\n",fTree->GetName());
   fprintf(fp,"\n   Long64_t nbytes = 0;\n");
   fprintf(fp,"//   for (Long64_t i=0; i<nentries;i++) {\n");
   fprintf(fp,"//      nbytes += %s->GetEntry(i);\n",fTree->GetName());
   fprintf(fp,"//   }\n");
   fprintf(fp,"}\n");

   printf("Macro: %s generated from Tree: %s\n",tfile.Data(), fTree->GetName());
   fclose(fp);

   return 0;
}

//______________________________________________________________________________
Int_t TTreePlayer::MakeProxy(const char *proxyClassname,
                             const char *macrofilename, const char *cutfilename,
                             const char *option, Int_t maxUnrolling)
{
   // Generate a skeleton analysis class for this Tree using TBranchProxy.
   // TBranchProxy is the base of a class hierarchy implementing an
   // indirect access to the content of the branches of a TTree.
   //
   // "proxyClassname" is expected to be of the form:
   //    [path/]fileprefix
   // The skeleton will then be generated in the file:
   //    fileprefix.h
   // located in the current directory or in 'path/' if it is specified.
   // The class generated will be named 'fileprefix'.
   // If the fileprefix contains a period, the right side of the period
   // will be used as the extension (instead of 'h') and the left side
   // will be used as the classname.
   //
   // "macrofilename" and optionally "cutfilename" are expected to point
   // to source file which will be included in by the generated skeletong.
   // Method of the same name as the file(minus the extension and path)
   // will be called by the generated skeleton's Process method as follow:
   //    [if (cutfilename())] htemp->Fill(macrofilename());
   //
   // "option" can be used select some of the optional features during
   // the code generation.  The possible options are:
   //    nohist : indicates that the generated ProcessFill should not
   //             fill the histogram.
   //
   // 'maxUnrolling' controls how deep in the class hierarchy does the
   // system 'unroll' class that are not split.  'unrolling' a class
   // will allow direct access to its data members a class (this
   // emulates the behavior of TTreeFormula).
   //
   // The main features of this skeleton are:
   //
   //    * on-demand loading of branches
   //    * ability to use the 'branchname' as if it was a data member
   //    * protection against array out-of-bound
   //    * ability to use the branch data as object (when the user code is available)
   //
   // For example with Event.root, if
   //    Double_t somepx = fTracks.fPx[2];
   // is executed by one of the method of the skeleton,
   // somepx will be updated with the current value of fPx of the 3rd track.
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
   // This 2 functions are run in a context such that the branch names are
   // available as local variables of the correct (read-only) type.
   //
   // Note that if you use the same 'variable' twice, it is more efficient
   // to 'cache' the value. For example
   //   Int_t n = fEventNumber; // Read fEventNumber
   //   if (n<10 || n>10) { ... }
   // is more efficient than
   //   if (fEventNumber<10 || fEventNumber>10)
   //
   // Access to TClonesArray.
   //
   // If a branch (or member) is a TClonesArray (let's say fTracks), you
   // can access the TClonesArray itself by using ->:
   //    fTracks->GetLast();
   // However this will load the full TClonesArray object and its content.
   // To quickly read the size of the TClonesArray use (note the dot):
   //    fTracks.GetEntries();
   // This will read only the size from disk if the TClonesArray has been
   // split.
   // To access the content of the TClonesArray, use the [] operator:
   //    float px = fTracks[i].fPx; // fPx of the i-th track
   //
   // Warning:
   //    The variable actually use for access are 'wrapper' around the
   // real data type (to add autoload for example) and hence getting to
   // the data involves the implicit call to a C++ conversion operator.
   // This conversion is automatic in most case.  However it is not invoked
   // in a few cases, in particular in variadic function (like printf).
   // So when using printf you should either explicitly cast the value or
   // use any intermediary variable:
   //      fprintf(stdout,"trs[%d].a = %d\n",i,(int)trs.a[i]);
   //
   // Also, optionally, the generated selector will also call methods named
   // macrofilename_methodname in each of 6 main selector methods if the method
   // macrofilename_methodname exist (Where macrofilename is stripped of its
   // extension).
   //
   // Concretely, with the script named h1analysisProxy.C,
   //
   // The method         calls the method (if it exist)
   // Begin           -> void h1analysisProxy_Begin(TTree*);
   // SlaveBegin      -> void h1analysisProxy_SlaveBegin(TTree*);
   // Notify          -> Bool_t h1analysisProxy_Notify();
   // Process         -> Bool_t h1analysisProxy_Process(Long64_t);
   // SlaveTerminate  -> void h1analysisProxy_SlaveTerminate();
   // Terminate       -> void h1analysisProxy_Terminate();
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

   if (macrofilename==0 || strlen(macrofilename)==0 ) {
      // We currently require a file name for the script
      Error("MakeProxy","A file name for the user script is required");
      return 0;
   }

   TTreeProxyGenerator gp(fTree,macrofilename,cutfilename,proxyClassname,option,maxUnrolling);

   return 0;
}

//______________________________________________________________________________
TPrincipal *TTreePlayer::Principal(const char *varexp, const char *selection, Option_t *option, Long64_t nentries, Long64_t firstentry)
{
// Interface to the Principal Components Analysis class.
//
//   Create an instance of TPrincipal
//   Fill it with the selected variables
//   if option "n" is specified, the TPrincipal object is filled with
//                 normalized variables.
//   If option "p" is specified, compute the principal components
//   If option "p" and "d" print results of analysis
//   If option "p" and "h" generate standard histograms
//   If option "p" and "c" generate code of conversion functions
//   return a pointer to the TPrincipal object. It is the user responsibility
//   to delete this object.
//   The option default value is "np"
//
//   See TTreePlayer::DrawSelect for explanation of the other parameters.

   TTreeFormula **var;
   std::vector<TString> cnames;
   TString opt = option;
   opt.ToLower();
   TPrincipal *principal = 0;
   Long64_t entry,entryNumber;
   Int_t i,nch;
   Int_t ncols = 8;   // by default first 8 columns are printed only
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;

   nentries = GetEntriesToProcess(firstentry, nentries);

//*-*- Compile selection expression if there is one
   TTreeFormula *select = 0;
   if (strlen(selection)) {
      select = new TTreeFormula("Selection",selection,fTree);
      if (!select) return principal;
      if (!select->GetNdim()) { delete select; return principal; }
      fFormulaList->Add(select);
   }
//*-*- if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (varexp && !strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      for (i=0;i<ncols;i++) {
         cnames.push_back( ((TLeaf*)leaves->At(i))->GetName() );
      }
//*-*- otherwise select only the specified columns
   } else {
      ncols = fSelector->SplitNames(varexp,cnames);
   }
   var = new TTreeFormula* [ncols];
   Double_t *xvars = new Double_t[ncols];

//*-*- Create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
      fFormulaList->Add(var[i]);
   }

//*-*- Create a TreeFormulaManager to coordinate the formulas
   TTreeFormulaManager *manager=0;
   if (fFormulaList->LastIndex()>=0) {
      manager = new TTreeFormulaManager;
      for(i=0;i<=fFormulaList->LastIndex();i++) {
         manager->Add((TTreeFormula*)fFormulaList->At(i));
      }
      manager->Sync();
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
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if (manager) manager->UpdateFormulaLeaves();
      }
      int ndata = 1;
      if (manager && manager->GetMultiplicity()) {
         ndata = manager->GetNdata();
      }

      for(int inst=0;inst<ndata;inst++) {
         Bool_t loaded = kFALSE;
         if (select) {
            if (select->EvalInstance(inst) == 0) {
               continue;
            }
         }

         if (inst==0) loaded = kTRUE;
         else if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (i=0;i<ncols;i++) {
               var[i]->EvalInstance(0);
            }
            loaded = kTRUE;
         }

         for (i=0;i<ncols;i++) {
            xvars[i] = var[i]->EvalInstance(inst);
         }
         principal->AddRow(xvars);
      }
   }

   //*-* some actions with principal ?
   if (opt.Contains("p")) {
      principal->MakePrincipals(); // Do the actual analysis
      if (opt.Contains("d")) principal->Print();
      if (opt.Contains("h")) principal->MakeHistograms();
      if (opt.Contains("c")) principal->MakeCode();
   }

//*-*- delete temporary objects
   fFormulaList->Clear();
   delete [] var;
   delete [] xvars;

   return principal;
}

//______________________________________________________________________________
Long64_t TTreePlayer::Process(const char *filename,Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Process this tree executing the TSelector code in the specified filename.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   //
   // The code in filename is loaded (interpreted or compiled, see below),
   // filename must contain a valid class implementation derived from TSelector,
   // where TSelector has the following member functions:
   //
   //    Begin():        called every time a loop on the tree starts,
   //                    a convenient place to create your histograms.
   //    SlaveBegin():   called after Begin(), when on PROOF called only on the
   //                    slave servers.
   //    Process():      called for each event, in this function you decide what
   //                    to read and fill your histograms.
   //    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
   //                    called only on the slave servers.
   //    Terminate():    called at the end of the loop on the tree,
   //                    a convenient place to draw/fit your histograms.
   //
   // If filename is of the form file.C, the file will be interpreted.
   // If filename is of the form file.C++, the file file.C will be compiled
   // and dynamically loaded.
   // If filename is of the form file.C+, the file file.C will be compiled
   // and dynamically loaded. At next call, if file.C is older than file.o
   // and file.so, the file.C is not compiled, only file.so is loaded.
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

   DeleteSelectorFromFile(); //delete previous selector if any

   // This might reloads the script and delete your option
   // string! so let copy it first:
   TString opt(option);
   TString file(filename);
   TSelector *selector = TSelector::GetSelector(file);
   if (!selector) return -1;

   fSelectorFromFile = selector;
   fSelectorClass    = selector->IsA();

   Long64_t nsel = Process(selector,opt,nentries,firstentry);
   return nsel;
}

//______________________________________________________________________________
Long64_t TTreePlayer::Process(TSelector *selector,Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Process this tree executing the code in the specified selector.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   //
   //   The TSelector class has the following member functions:
   //
   //    Begin():        called every time a loop on the tree starts,
   //                    a convenient place to create your histograms.
   //    SlaveBegin():   called after Begin(), when on PROOF called only on the
   //                    slave servers.
   //    Process():      called for each event, in this function you decide what
   //                    to read and fill your histograms.
   //    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
   //                    called only on the slave servers.
   //    Terminate():    called at the end of the loop on the tree,
   //                    a convenient place to draw/fit your histograms.
   //
   //  If the Tree (Chain) has an associated EventList, the loop is on the nentries
   //  of the EventList, starting at firstentry, otherwise the loop is on the
   //  specified Tree entries.

   nentries = GetEntriesToProcess(firstentry, nentries);

   TDirectory::TContext ctxt(0);

   fTree->SetNotify(selector);

   selector->SetOption(option);

   selector->Begin(fTree);       //<===call user initialization function
   selector->SlaveBegin(fTree);  //<===call user initialization function
   if (selector->Version() >= 2)
      selector->Init(fTree);
   selector->Notify();

   if (gMonitoringWriter)
      gMonitoringWriter->SendProcessingStatus("STARTED",kTRUE);

   if (selector->GetAbort() != TSelector::kAbortProcess
       && (selector->Version() != 0 || selector->GetStatus() != -1)) {

      Long64_t readbytesatstart = 0;
      readbytesatstart = TFile::GetFileBytesRead();

      //set the file cache
      TTreeCache *tpf = 0;
      TFile *curfile = fTree->GetCurrentFile();
      if (curfile && fTree->GetCacheSize() > 0) {
         tpf = (TTreeCache*)curfile->GetCacheRead();
         if (tpf)
            tpf->SetEntryRange(firstentry,firstentry+nentries);
         else {
            fTree->SetCacheSize(fTree->GetCacheSize());
            tpf = (TTreeCache*)curfile->GetCacheRead();
            if (tpf) tpf->SetEntryRange(firstentry,firstentry+nentries);
         }
      }

      //Create a timer to get control in the entry loop(s)
      TProcessEventTimer *timer = 0;
      Int_t interval = fTree->GetTimerInterval();
      if (!gROOT->IsBatch() && interval)
         timer = new TProcessEventTimer(interval);

      //loop on entries (elist or all entries)
      Long64_t entry, entryNumber, localEntry;

      Bool_t useCutFill = selector->Version() == 0;

      // force the first monitoring info
      if (gMonitoringWriter)
         gMonitoringWriter->SendProcessingProgress(0,0,kTRUE);

      //trying to set the first tree, because in the Draw function
      //the tree corresponding to firstentry has already been loaded,
      //so it is not set in the entry list
      fSelectorUpdate = selector;
      UpdateFormulaLeaves();

      for (entry=firstentry;entry<firstentry+nentries;entry++) {
         entryNumber = fTree->GetEntryNumber(entry);
         if (entryNumber < 0) break;
         if (timer && timer->ProcessEvents()) break;
         if (gROOT->IsInterrupted()) break;
         localEntry = fTree->LoadTree(entryNumber);
         if (localEntry < 0) break;
         if(useCutFill) {
            if (selector->ProcessCut(localEntry))
               selector->ProcessFill(localEntry); //<==call user analysis function
         } else {
            selector->Process(localEntry);        //<==call user analysis function
         }
         if (gMonitoringWriter)
            gMonitoringWriter->SendProcessingProgress((entry-firstentry),TFile::GetFileBytesRead()-readbytesatstart,kTRUE);
         if (selector->GetAbort() == TSelector::kAbortProcess) break;
      }
      delete timer;
      //we must reset the cache
      {
         TFile *curfile2 = fTree->GetCurrentFile();
         if (curfile2 && fTree->GetCacheSize() > 0) {
            tpf = (TTreeCache*)curfile2->GetCacheRead();
            if (tpf) tpf->SetEntryRange(0,0);
         }
      }
   }

   if (selector->Version() != 0 || selector->GetStatus() != -1) {
      selector->SlaveTerminate();   //<==call user termination function
      selector->Terminate();        //<==call user termination function
   }
   fSelectorUpdate = 0;
   if (gMonitoringWriter)
      gMonitoringWriter->SendProcessingStatus("DONE");

   return selector->GetStatus();
}

//______________________________________________________________________________
void TTreePlayer::RecursiveRemove(TObject *obj)
{
// cleanup pointers in the player pointing to obj

   if (fHistogram == obj) fHistogram = 0;
}

//______________________________________________________________________________
Long64_t TTreePlayer::Scan(const char *varexp, const char *selection,
                           Option_t * option,
                           Long64_t nentries, Long64_t firstentry)
{
   // Loop on Tree and print entries passing selection. If varexp is 0 (or "")
   // then print only first 8 columns. If varexp = "*" print all columns.
   // Otherwise a columns selection can be made using "var1:var2:var3".
   // The function returns the number of entries passing the selection.
   //
   // By default 50 rows are shown and you are asked for <CR>
   // to see the next 50 rows.
   // You can change the default number of rows to be shown before <CR>
   // via  mytree->SetScanField(maxrows) where maxrows is 50 by default.
   // if maxrows is set to 0 all rows of the Tree are shown.
   // This option is interesting when dumping the contents of a Tree to
   // an ascii file, eg from the command line
   //   tree->SetScanField(0);
   //   tree->Scan("*"); >tree.log
   //  will create a file tree.log
   //
   // Arrays (within an entry) are printed in their linear forms.
   // If several arrays with multiple dimensions are printed together,
   // they will NOT be synchronized.  For example print
   //   arr1[4][2] and arr2[2][3] will results in a printing similar to:
   // ***********************************************
   // *    Row   * Instance *      arr1 *      arr2 *
   // ***********************************************
   // *        x *        0 * arr1[0][0]* arr2[0][0]*
   // *        x *        1 * arr1[0][1]* arr2[0][1]*
   // *        x *        2 * arr1[1][0]* arr2[0][2]*
   // *        x *        3 * arr1[1][1]* arr2[1][0]*
   // *        x *        4 * arr1[2][0]* arr2[1][1]*
   // *        x *        5 * arr1[2][1]* arr2[1][2]*
   // *        x *        6 * arr1[3][0]*           *
   // *        x *        7 * arr1[3][1]*           *
   //
   // However, if there is a selection criterion which is an array, then
   // all the formulas will be synchronized with the selection criterion
   // (see TTreePlayer::DrawSelect for more information).
   //
   // The options string can contains the following parameters:
   //    lenmax=dd
   //       Where 'dd' is the maximum number of elements per array that should
   //       be printed.  If 'dd' is 0, all elements are printed (this is the
   //       default)
   //    colsize=ss
   //       Where 'ss' will be used as the default size for all the column
   //       If this options is not specified, the default column size is 9
   //    precision=pp
   //       Where 'pp' will be used as the default 'precision' for the
   //       printing format.
   //    col=xxx
   //       Where 'xxx' is colon (:) delimited list of printing format for
   //       each column. The format string should follow the printf format
   //       specification.  The value given will be prefixed by % and, if no
   //       conversion specifier is given, will be suffixed by the letter g.
   //       before being passed to fprintf.  If no format is specified for a
   //       column, the default is used  (aka ${colsize}.${precision}g )
   // For example:
   //   tree->Scan("a:b:c","","colsize=30 precision=3 col=::20.10:#x:5ld");
   // Will print 3 columns, the first 2 columns will be 30 characters long,
   // the third columns will be 20 characters long.  The printing format used
   // for the columns (assuming they are numbers) will be respectively:
   //   %30.3g %30.3g %20.10g %#x %5ld


   TString opt = option;
   opt.ToLower();
   UInt_t ui;
   UInt_t lenmax = 0;
   UInt_t colDefaultSize = 9;
   UInt_t colPrecision = 9;
   vector<TString> colFormats;
   vector<Int_t> colSizes;

   if (opt.Contains("lenmax=")) {
      int start = opt.Index("lenmax=");
      int numpos = start + strlen("lenmax=");
      int numlen = 0;
      int len = opt.Length();
      while( (numpos+numlen<len) && isdigit(opt[numpos+numlen]) ) numlen++;
      TString num = opt(numpos,numlen);
      opt.Remove(start,strlen("lenmax")+numlen);

      lenmax = atoi(num.Data());
   }
   if (opt.Contains("colsize=")) {
      int start = opt.Index("colsize=");
      int numpos = start + strlen("colsize=");
      int numlen = 0;
      int len = opt.Length();
      while( (numpos+numlen<len) && isdigit(opt[numpos+numlen]) ) numlen++;
      TString num = opt(numpos,numlen);
      opt.Remove(start,strlen("size")+numlen);

      colDefaultSize = atoi(num.Data());
      colPrecision = colDefaultSize;
      if (colPrecision>18) colPrecision = 18;
   }
   if (opt.Contains("precision=")) {
      int start = opt.Index("precision=");
      int numpos = start + strlen("precision=");
      int numlen = 0;
      int len = opt.Length();
      while( (numpos+numlen<len) && isdigit(opt[numpos+numlen]) ) numlen++;
      TString num = opt(numpos,numlen);
      opt.Remove(start,strlen("precision")+numlen);

      colPrecision = atoi(num.Data());
   }
   TString defFormat = Form("%d.%d",colDefaultSize,colPrecision);
   if (opt.Contains("col=")) {
      int start = opt.Index("col=");
      int numpos = start + strlen("col=");
      int numlen = 0;
      int len = opt.Length();
      while( (numpos+numlen<len) &&
             (isdigit(opt[numpos+numlen])
              || opt[numpos+numlen] == 'c'
              || opt[numpos+numlen] == 'd'
              || opt[numpos+numlen] == 'i'
              || opt[numpos+numlen] == 'o'
              || opt[numpos+numlen] == 'x'
              || opt[numpos+numlen] == 'X'
              || opt[numpos+numlen] == 'u'
              || opt[numpos+numlen] == 'f'
              || opt[numpos+numlen] == 'e'
              || opt[numpos+numlen] == 'E'
              || opt[numpos+numlen] == 'g'
              || opt[numpos+numlen] == 'G'
              || opt[numpos+numlen] == 'l'
              || opt[numpos+numlen] == 'L'
              || opt[numpos+numlen] == 'h'
              || opt[numpos+numlen] == '#'
              || opt[numpos+numlen]=='.'
              || opt[numpos+numlen]==':')) numlen++;
      TString flist = opt(numpos,numlen);
      opt.Remove(start,strlen("col")+numlen);

      int i = 0;
      while(i<flist.Length() && flist[i]==':') {
         colFormats.push_back(defFormat);
         colSizes.push_back(colDefaultSize);
         ++i;
      }
      for(; i<flist.Length(); ++i) {
         int next = flist.Index(":",i);
         if (next==i) {
            colFormats.push_back(defFormat);
         } else if (next==kNPOS) {
            colFormats.push_back(flist(i,flist.Length()-i));
            i = flist.Length();
         } else {
            colFormats.push_back(flist(i,next-i));
            i = next;
         }
         UInt_t siz = atoi(colFormats[colFormats.size()-1].Data());
         colSizes.push_back( siz ? siz : colDefaultSize );
      }
   }

   TTreeFormula **var;
   std::vector<TString> cnames;
   TString onerow;
   Long64_t entry,entryNumber;
   Int_t i,nch;
   UInt_t ncols = 8;   // by default first 8 columns are printed only
   ofstream out;
   Int_t lenfile = 0;
   char * fname = 0;
   if (fScanRedirect) {
      fTree->SetScanField(0);  // no page break if Scan is redirected
      fname = (char *) fScanFileName;
      if (!fname) fname = (char*)"";
      lenfile = strlen(fname);
      if (!lenfile) {
         Int_t nch2 = strlen(fTree->GetName());
         fname = new char[nch2+10];
         strlcpy(fname, fTree->GetName(),nch2+10);
         strlcat(fname, "-scan.dat",nch2+10);
      }
      out.open(fname, ios::out);
      if (!out.good ()) {
         if (!lenfile) delete [] fname;
         Error("Scan","Can not open file for redirection");
         return 0;
      }
   }
   TObjArray *leaves = fTree->GetListOfLeaves();
   if (leaves==0) return 0;
   UInt_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;

   nentries = GetEntriesToProcess(firstentry, nentries);

//*-*- Compile selection expression if there is one
   TTreeFormula        *select  = 0;
   if (selection && strlen(selection)) {
      select = new TTreeFormula("Selection",selection,fTree);
      if (!select) return -1;
      if (!select->GetNdim()) { delete select; return -1; }
      fFormulaList->Add(select);
   }
//*-*- if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (varexp && !strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      UInt_t ncs = ncols;
      ncols = 0;
      for (ui=0;ui<ncs;++ui) {
         TLeaf *lf = (TLeaf*)leaves->At(ui);
         if (lf->GetBranch()->GetListOfBranches()->GetEntries() > 0) continue;
         cnames.push_back( lf->GetBranch()->GetMother()->GetName() );
         if (cnames[ncols] == lf->GetName() ) {
            // Already complete, let move on.
         } else if (cnames[ncols][cnames[ncols].Length()-1]=='.') {
            cnames[ncols] = lf->GetBranch()->GetName(); // name of branch already include mother's name
         } else {
            if (lf->GetBranch()->GetMother()->IsA()->InheritsFrom(TBranchElement::Class())) {
               TBranchElement *mother = (TBranchElement*)lf->GetBranch()->GetMother();
               if (mother->GetType() == 3 || mother->GetType() == 4) {
                  // The name of the mother branch is embedded in the sub-branch names.
                  cnames[ncols] = lf->GetBranch()->GetName();
                  ++ncols;
                  continue;
               }
            }
            if (!strchr(lf->GetBranch()->GetName() ,'[') ) {
               cnames[ncols].Append('.');
               cnames[ncols].Append( lf->GetBranch()->GetName() );
            }
         }
         if (strcmp( lf->GetBranch()->GetName(), lf->GetName() ) != 0 ) {
            cnames[ncols].Append('.');
            cnames[ncols].Append( lf->GetName() );
         }
         ++ncols;
      }
//*-*- otherwise select only the specified columns
   } else {

      ncols = fSelector->SplitNames(varexp, cnames);

   }
   var = new TTreeFormula* [ncols];

   for(ui=colFormats.size();ui<ncols;++ui) {
      colFormats.push_back(defFormat);
      colSizes.push_back(colDefaultSize);
   }

//*-*- Create the TreeFormula objects corresponding to each column
   for (ui=0;ui<ncols;ui++) {
      var[ui] = new TTreeFormula("Var1",cnames[ui].Data(),fTree);
      fFormulaList->Add(var[ui]);
   }

//*-*- Create a TreeFormulaManager to coordinate the formulas
   TTreeFormulaManager *manager=0;
   Bool_t hasArray = kFALSE;
   Bool_t forceDim = kFALSE;
   if (fFormulaList->LastIndex()>=0) {
      if (select) {
         if (select->GetManager()->GetMultiplicity() > 0 ) {
            manager = new TTreeFormulaManager;
            for(i=0;i<=fFormulaList->LastIndex();i++) {
               manager->Add((TTreeFormula*)fFormulaList->At(i));
            }
            manager->Sync();
         }
      }
      for(i=0;i<=fFormulaList->LastIndex();i++) {
         TTreeFormula *form = ((TTreeFormula*)fFormulaList->At(i));
         switch( form->GetManager()->GetMultiplicity() ) {
            case  1:
            case  2:
               hasArray = kTRUE;
               forceDim = kTRUE;
               break;
            case -1:
               forceDim = kTRUE;
               break;
            case  0:
               break;
         }

      }
   }

//*-*- Print header
   onerow = "***********";
   if (hasArray) onerow += "***********";

   for (ui=0;ui<ncols;ui++) {
      TString starFormat = Form("*%%%d.%ds",colSizes[ui]+2,colSizes[ui]+2);
      onerow += Form(starFormat.Data(),var[ui]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   onerow = "*    Row   ";
   if (hasArray) onerow += "* Instance ";
   for (ui=0;ui<ncols;ui++) {
      TString numbFormat = Form("* %%%d.%ds ",colSizes[ui],colSizes[ui]);
      onerow += Form(numbFormat.Data(),var[ui]->PrintValue(-1));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   onerow = "***********";
   if (hasArray) onerow += "***********";
   for (ui=0;ui<ncols;ui++) {
      TString starFormat = Form("*%%%d.%ds",colSizes[ui]+2,colSizes[ui]+2);
      onerow += Form(starFormat.Data(),var[ui]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
//*-*- loop on all selected entries
   fSelectedRows = 0;
   Int_t tnumber = -1;
   Bool_t exitloop = kFALSE;
   for (entry=firstentry;
        entry<(firstentry+nentries) && !exitloop;
        entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if (manager) manager->UpdateFormulaLeaves();
         else {
            for(i=0;i<=fFormulaList->LastIndex();i++) {
               ((TTreeFormula*)fFormulaList->At(i))->UpdateFormulaLeaves();
            }
         }
      }

      int ndata = 1;
      if (forceDim) {

         if (manager) {

            ndata = manager->GetNdata(kTRUE);

         } else {

            // let's print the max number of column
            for (ui=0;ui<ncols;ui++) {
               if (ndata < var[ui]->GetNdata() ) {
                  ndata = var[ui]->GetNdata();
               }
            }
            if (select && select->GetNdata()==0) ndata = 0;
         }

      }

      if (lenmax && ndata>(int)lenmax) ndata = lenmax;
      Bool_t loaded = kFALSE;
      for(int inst=0;inst<ndata;inst++) {
         if (select) {
            if (select->EvalInstance(inst) == 0) {
               continue;
            }
         }
         if (inst==0) loaded = kTRUE;
         else if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (ui=0;ui<ncols;ui++) {
               var[ui]->EvalInstance(0);
            }
            loaded = kTRUE;
         }
         onerow = Form("* %8lld ",entryNumber);
         if (hasArray) {
            onerow += Form("* %8d ",inst);
         }
         for (ui=0;ui<ncols;++ui) {
            TString numbFormat = Form("* %%%d.%ds ",colSizes[ui],colSizes[ui]);
            if (var[ui]->GetNdim()) onerow += Form(numbFormat.Data(),var[ui]->PrintValue(0,inst,colFormats[ui].Data()));
            else {
               TString emptyForm = Form("* %%%dc ",colSizes[ui]);
               onerow += Form(emptyForm.Data(),' ');
            }
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
               if (answer == 'q' || answer == 'Q') {
                  exitloop = kTRUE;
                  break;
               }
            }
         }
      }
   }
   onerow = "***********";
   if (hasArray) onerow += "***********";
   for (ui=0;ui<ncols;ui++) {
      TString starFormat = Form("*%%%d.%ds",colSizes[ui]+2,colSizes[ui]+2);
      onerow += Form(starFormat.Data(),var[ui]->PrintValue(-2));
   }
   if (fScanRedirect)
      out<<onerow.Data()<<"*"<<endl;
   else
      printf("%s*\n",onerow.Data());
   if (select) Printf("==> %lld selected %s", fSelectedRows,
                      fSelectedRows == 1 ? "entry" : "entries");
   if (fScanRedirect) printf("File <%s> created\n", fname);

//*-*- delete temporary objects
   fFormulaList->Clear();
   // The TTreeFormulaManager is deleted by the last TTreeFormula.
   delete [] var;
   return fSelectedRows;
}

//______________________________________________________________________________
TSQLResult *TTreePlayer::Query(const char *varexp, const char *selection,
                               Option_t *, Long64_t nentries, Long64_t firstentry)
{
   // Loop on Tree and return TSQLResult object containing entries passing
   // selection. If varexp is 0 (or "") then print only first 8 columns.
   // If varexp = "*" print all columns. Otherwise a columns selection can
   // be made using "var1:var2:var3". In case of error 0 is returned otherwise
   // a TSQLResult object which must be deleted by the user.

   TTreeFormula **var;
   std::vector<TString> cnames;
   TString onerow;
   Long64_t entry,entryNumber;
   Int_t i,nch;
   Int_t ncols = 8;   // by default first 8 columns are printed only
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntriesFast();
   if (nleaves < ncols) ncols = nleaves;
   nch = varexp ? strlen(varexp) : 0;

   nentries = GetEntriesToProcess(firstentry, nentries);

   // compile selection expression if there is one
   TTreeFormula *select = 0;
   if (strlen(selection)) {
      select = new TTreeFormula("Selection",selection,fTree);
      if (!select) return 0;
      if (!select->GetNdim()) { delete select; return 0; }
      fFormulaList->Add(select);
   }

   // if varexp is empty, take first 8 columns by default
   int allvar = 0;
   if (varexp && !strcmp(varexp, "*")) { ncols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      for (i=0;i<ncols;i++) {
         cnames.push_back( ((TLeaf*)leaves->At(i))->GetName() );
      }
   } else {
      // otherwise select only the specified columns
      ncols = fSelector->SplitNames(varexp,cnames);
   }
   var = new TTreeFormula* [ncols];

   // create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
      fFormulaList->Add(var[i]);
   }

   // fill header info into result object
   TTreeResult *res = new TTreeResult(ncols);
   for (i = 0; i < ncols; i++) {
      res->AddField(i, var[i]->PrintValue(-1));
   }

   //*-*- Create a TreeFormulaManager to coordinate the formulas
   TTreeFormulaManager *manager=0;
   if (fFormulaList->LastIndex()>=0) {
      manager = new TTreeFormulaManager;
      for(i=0;i<=fFormulaList->LastIndex();i++) {
         manager->Add((TTreeFormula*)fFormulaList->At(i));
      }
      manager->Sync();
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
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         for (i=0;i<ncols;i++) var[i]->UpdateFormulaLeaves();
      }

      Int_t ndata = 1;
      if (manager && manager->GetMultiplicity()) {
         ndata = manager->GetNdata();
      }

      if (select) {
         select->GetNdata();
         if (select->EvalInstance(0) == 0) continue;
      }

      Bool_t loaded = kFALSE;
      for(int inst=0;inst<ndata;inst++) {
         if (select) {
            if (select->EvalInstance(inst) == 0) {
               continue;
            }
         }

         if (inst==0) loaded = kTRUE;
         else if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (i=0;i<ncols;i++) {
               var[i]->EvalInstance(0);
            }
            loaded = kTRUE;
         }
         for (i=0;i<ncols;i++) {
            aresult = var[i]->PrintValue(0,inst);
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
   }

   // delete temporary objects
   fFormulaList->Clear();
   // The TTreeFormulaManager is deleted by the last TTreeFormula.
   delete [] fields;
   delete [] arow;
   delete [] var;

   return res;
}

//_______________________________________________________________________
void TTreePlayer::SetEstimate(Long64_t n)
{
//*-*-*-*-*-*-*-*-*Set number of entries to estimate variable limits*-*-*-*
//*-*              ================================================
//
   fSelector->SetEstimate(n);
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

   if (ww || wh) { }   // use unused variables
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualTreeViewer"))) {
      if (h->LoadPlugin() == -1)
         return;
      h->ExecPlugin(1,fTree);
   }
}

//______________________________________________________________________________
void TreeUnbinnedFitLikelihood(Int_t & /*npar*/, Double_t * /*gin*/,
                               Double_t &r, Double_t *par, Int_t /*flag*/)
{
   // The fit function used by the unbinned likelihood fit.

   Double_t x[3];
   TF1 *fitfunc = (TF1*)tFitter->GetObjectFit();
   fitfunc->InitArgs(x,par);

   Long64_t n = gTree->GetSelectedRows();
   Double_t  *data1 = gTree->GetV1();
   Double_t  *data2 = gTree->GetV2();
   Double_t  *data3 = gTree->GetV3();
   Double_t *weight = gTree->GetW();
   Double_t logEpsilon = -230;   // protect against negative probabilities
   Double_t logL = 0.0, prob;
   //printf("n=%lld, data1=%x, weight=%x\n",n,data1,weight);

   for(Long64_t i = 0; i < n; i++) {
      if (weight[i] <= 0) continue;
      x[0] = data1[i];
      if (data2) x[1] = data2[i];
      if (data3) x[2] = data3[i];
      prob = fitfunc->EvalPar(x,par);
      //printf("i=%lld, x=%g, w=%g, prob=%g, logL=%g\n",i,x[0],weight[i],prob,logL);
      if(prob > 0) logL += TMath::Log(prob) * weight[i];
      else         logL += logEpsilon * weight[i];
   }

   r = -2*logL;
}


//______________________________________________________________________________
Int_t TTreePlayer::UnbinnedFit(const char *funcname ,const char *varexp, const char *selection,Option_t *option ,Long64_t nentries, Long64_t firstentry)
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
//             = "D" Draw the projected histogram with the fitted function
//                   normalized to the number of selected rows
//                   and multiplied by the bin width
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
//
//    Return status
//    =============
//   The function return the status of the fit in the following form
//     fitResult = migradResult + 10*minosResult + 100*hesseResult + 1000*improveResult
//   The fitResult is 0 is the fit is OK.
//   The fitResult is negative in case of an error not connected with the fit.
//   The number of entries used in the fit can be obtained via
//     mytree.GetSelectedRows();
//   If the number of selected entries is null the function returns -1


// new implementation using new Fitter classes

   gTree = fTree; // not sure if this is still needed

   // function is given by name, find it in gROOT
   TF1* fitfunc = (TF1*)gROOT->GetFunction(funcname);
   if (!fitfunc) { Error("UnbinnedFit", "Unknown function: %s",funcname); return 0; }

   Int_t npar = fitfunc->GetNpar();
   if (npar <=0) { Error("UnbinnedFit", "Illegal number of parameters = %d",npar); return 0; }

   // Spin through the data to select out the events of interest
   // Make sure that the arrays V1,etc are created large enough to accommodate
   // all entries
   Long64_t oldEstimate = fTree->GetEstimate();
   Long64_t nent = fTree->GetEntriesFriend();
   fTree->SetEstimate(TMath::Min(nent,nentries));

   // build FitOptions
   TString opt = option;
   opt.ToUpper();
   Foption_t fitOption;
   if (opt.Contains("Q")) fitOption.Quiet   = 1;
   if (opt.Contains("V")){fitOption.Verbose = 1; fitOption.Quiet   = 0;}
   if (opt.Contains("E")) fitOption.Errors  = 1;
   if (opt.Contains("M")) fitOption.More    = 1;
   if (!opt.Contains("D")) fitOption.Nograph    = 1;  // what about 0
   // could add range and automatic normalization of functions and gradient

   TString drawOpt = "goff para";
   if (!fitOption.Nograph) drawOpt = "";
   Long64_t nsel = DrawSelect(varexp, selection,drawOpt, nentries, firstentry);

   if (!fitOption.Nograph  && GetSelectedRows() <= 0 && GetDimension() > 4) {
      Info("UnbinnedFit","Ignore option D with more than 4 variables");
      nsel = DrawSelect(varexp, selection,"goff para", nentries, firstentry);
   }

   //if no selected entries return
   Long64_t nrows = GetSelectedRows();

   if (nrows <= 0) {
      Error("UnbinnedFit", "Cannot fit: no entries selected");
      return -1;
   }

   // Check that function has same dimension as number of variables
   Int_t ndim = GetDimension();
   // do not check with TF1::GetNdim() since it returns 1 for TF1 classes created with
   // a C function with larger dimension


   // use pointer stored in the tree (not copy the data in)
   std::vector<double *> vlist(ndim);
   for (int i = 0; i < ndim; ++i)
      vlist[i] = fSelector->GetVal(i);

   // fill the data
   ROOT::Fit::UnBinData * fitdata = new ROOT::Fit::UnBinData(nrows, ndim, vlist.begin());



   ROOT::Math::MinimizerOptions minOption;
   TFitResultPtr ret = ROOT::Fit::UnBinFit(fitdata,fitfunc, fitOption, minOption);

   //reset estimate
   fTree->SetEstimate(oldEstimate);

   //if option "D" is specified, draw the projected histogram
   //with the fitted function normalized to the number of selected rows
   //and multiplied by the bin width
   if (!fitOption.Nograph && fHistogram) {
      if (fHistogram->GetDimension() < 2) {
         TH1 *hf = (TH1*)fHistogram->Clone("unbinnedFit");
         hf->SetLineWidth(3);
         hf->Reset();
         Int_t nbins = fHistogram->GetXaxis()->GetNbins();
         Double_t norm = ((Double_t)nsel)*fHistogram->GetXaxis()->GetBinWidth(1);
         for (Int_t bin=1;bin<=nbins;bin++) {
            Double_t func = norm*fitfunc->Eval(hf->GetBinCenter(bin));
            hf->SetBinContent(bin,func);
         }
         fHistogram->GetListOfFunctions()->Add(hf,"lsame");
      }
      fHistogram->Draw();
   }


   return int(ret);

}

//______________________________________________________________________________
void TTreePlayer::UpdateFormulaLeaves()
{
   // this function is called by TChain::LoadTree when a new Tree is loaded.
   // Because Trees in a TChain may have a different list of leaves, one
   // must update the leaves numbers in the TTreeFormula used by the TreePlayer.

   if (fSelector)  fSelector->Notify();
   if (fSelectorUpdate){
      //If the selector is writing into a TEntryList, the entry list's
      //sublists need to be changed according to the loaded tree
      if (fSelector==fSelectorUpdate) {
         //FIXME: should be more consistent with selector from file
         TObject *obj = fSelector->GetObject();
         if (obj){
            if (fSelector->GetObject()->InheritsFrom(TEntryList::Class())){
               ((TEntryList*)fSelector->GetObject())->SetTree(fTree->GetTree());
            }
         }
      }
      if (fSelectorFromFile==fSelectorUpdate) {
         TIter next(fSelectorFromFile->GetOutputList());
         TEntryList *elist=0;
         while ((elist=(TEntryList*)next())){
            if (elist->InheritsFrom(TEntryList::Class())){
               elist->SetTree(fTree->GetTree());
            }
         }
      }
   }

   if (fFormulaList->GetSize()) {
      TObjLink *lnk = fFormulaList->FirstLink();
      while (lnk) {
         lnk->GetObject()->Notify();
         lnk = lnk->Next();
      }
   }
}
