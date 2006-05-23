// @(#)root/tree:$Name: v5-11-02 $:$Id: TSelector.cxx,v 1.21 2006/03/20 21:43:44 pcanal Exp $
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// A TSelector object is used by the TTree::Draw, TTree::Scan,            //
//  TTree::Loop, TTree::Process to navigate in a TTree and make           //
//  selections.                                                           //
//                                                                        //
//  The following members functions are called by the TTree functions.    //
//    Init:        Attach a new TTree during the loop                     //
//    Begin:       called everytime a loop on the tree(s) starts.         //
//                 a convenient place to create your histograms.          //
//                                                                        //
//    Notify():    This function is called at the first entry of a new    //
//                 tree in a chain.                                       //
//    ProcessCut:  called at the beginning of each entry to return a flag //
//                 true if the entry must be analyzed.                    //
//    ProcessFill: called in the entry loop for all entries accepted      //
//                 by Select.                                             //
//    Terminate:   called at the end of a loop on a TTree.                //
//                 a convenient place to draw/fit your histograms.        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "TError.h"
#include "TSelectorCint.h"
#include "Api.h"
#include "TClass.h"

ClassImp(TSelector)

//______________________________________________________________________________
TSelector::TSelector() : TObject()
{
   // Default selector ctor.

   fStatus = 0;
   fObject = 0;
   fInput  = 0;
   fOutput = new TSelectorList;
   fOutput->SetOwner();
}

//______________________________________________________________________________
TSelector::TSelector(const TSelector& sel) :
  TObject(sel),
  fStatus(sel.fStatus),
  fOption(sel.fOption),
  fObject(sel.fObject),
  fInput(sel.fInput),
  fOutput(sel.fOutput)
{ }
   
//______________________________________________________________________________
TSelector& TSelector::operator=(const TSelector& sel)
{
  if(this!=&sel) {
    TObject::operator=(sel);
    fStatus=sel.fStatus;
    fOption=sel.fOption;
    fObject=sel.fObject;
    fInput=sel.fInput;
    fOutput=sel.fOutput;
  } return *this;
}
   
//______________________________________________________________________________
TSelector::~TSelector()
{
   // Selector destructor.

   delete fOutput;
}

//______________________________________________________________________________
TSelector *TSelector::GetSelector(const char *filename)
{
//   The code in filename is loaded (interpreted or compiled, see below)
//   filename must contain a valid class implementation derived from TSelector,
//   where TSelector has the following member functions:
//
//     void TSelector::Init(TTree *t). Called every time a new TTree is attached.
//     void TSelector::Begin(). This function is called before looping on the
//          events in the Tree. The user can create his histograms in this function.
//
//     Bool_t TSelector::Notify(). This function is called at the first entry
//          of a new file in a chain.
//
//     Bool_t TSelector::Process(Long64_t entry). This function is called
//          to process an event. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          Once the entry is in memory one can apply a selection and if the
//          event is selected histograms can be filled. Processing stops
//          when this function returns kFALSE. This function combines the
//          next two functions in one, avoiding to have to maintain state
//          in the class to communicate between these two funtions.
//          See WARNING below about entry.
//          This method is used by PROOF.
//     Bool_t TSelector::ProcessCut(Long64_t entry). This function is called
//          before processing entry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise. See WARNING below about entry.
//     void TSelector::ProcessFill(Long64_t entry). This function is called for
//          all selected events. User fills histograms in this function.
//           See WARNING below about entry.
//     void TSelector::Terminate(). This function is called at the end of
//          the loop on all events.
//
//   WARNING when a selector is used with a TChain:
//    in the Process, ProcessCut, ProcessFill function, you must use
//    the pointer to the current Tree to call GetEntry(entry).
//    entry is always the local entry number in the current tree.
//    Assuming that fChain is the pointer to the TChain being processed,
//    use fChain->GetTree()->GetEntry(entry);
//
//   If filename is of the form file.C, the file will be interpreted.
//   If filename is of the form file.C++, the file file.C will be compiled
//      and dynamically loaded. The corresponding binary file and shared
//      library will be deleted at the end of the function.
//   If filename is of the form file.C+, the file file.C will be compiled
//      and dynamically loaded. At next call, if file.C is older than file.o
//      and file.so, the file.C is not compiled, only file.so is loaded.
//
//   The static function returns a pointer to a TSelector object

   // If the filename does not contain "." assume class is compiled in
   char localname[4096];
   Bool_t fromFile = kFALSE;
   if ( strchr(filename, '.') != 0 ) {

      //Interpret/compile filename via CINT
      sprintf(localname,".L %s",filename);
      gROOT->ProcessLine(localname);
      fromFile = kTRUE;
   }

   //loop on all classes known to CINT to find the class on filename
   //that derives from TSelector
   const char *basename = gSystem->BaseName(filename);
   if (basename==0) {
      ::Error("TSelector::GetSelector","Unable to determine the classname for file %s",filename);
      return 0;
   }
   strcpy(localname,basename);
   Bool_t  isCompiled = !fromFile || strchr(localname,'+') != 0 ;
   char *dot        = strchr(localname,'.');
   if (dot) dot[0] = 0;

   G__ClassInfo cl;
   Bool_t ok = kFALSE;
   while (cl.Next()) {
      if (strcmp(cl.Name(),localname)) continue;
      if (cl.IsBase("TSelector")) ok = kTRUE;
      break;
   }
   if (!ok) {
      if ( fromFile ) {
         ::Error("TSelector::GetSelector",
         "file %s does not have a valid class deriving from TSelector",filename);
      } else {
         ::Error("TSelector::GetSelector",
         "class %s does not exist or does not derive from TSelector",filename);
      }
      return 0;
   }

   // we can now create an instance of the class
   TSelector *selector = (TSelector*)cl.New();
   if (!selector || isCompiled) return selector;
   //interpreted selector: cannot be used as such
   //create a fake selector
   TSelectorCint *select = new TSelectorCint();
   select->Build(selector,&cl);

   return select;
}

//______________________________________________________________________________
Bool_t TSelector::IsStandardDraw(const char *selec)
{
   // Find out if this is a standard selection used for Draw actions
   // (either TSelectorDraw, TProofDraw or deriving from them).

   // Make sure we have a name
   if (!selec) {
      ::Info("TSelector::IsStandardDraw",
             "selector name undefined - do nothing");
      return kFALSE;
   }

   Bool_t stdselec = kFALSE;
   if (!strchr(selec, '.')) {
      if (strstr(selec, "TSelectorDraw")) {
         stdselec = kTRUE;
      } else {
         TClass *cl = TClass::GetClass(selec);
         if (cl && (cl->InheritsFrom("TProofDraw") ||
                    cl->InheritsFrom("TSelectorDraw")))
            stdselec = kTRUE;
         SafeDelete(cl);
      }
   }

   // We are done
   return stdselec;
}
