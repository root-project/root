// @(#)root/treeplayer:$Name:  $:$Id: TSelector.cxx,v 1.6 2001/03/03 08:49:35 brun Exp $
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
//    Begin:       called everytime a loop on the tree starts.            //
//                 a convenient place to create your histograms.          //
//                                                                        //
//    Notify():    This function is called at the first entry of a new    //
//                 in a chain.                                            //
//    ProcessCut:  called at the beginning of each entry to return a flag //
//                 true if the entry must be analyzed.                    //
//    ProcessFill: called in the entry loop for all entries accepted      //
//                 by Select.                                             //
//    Terminate:   called at the end of a loop on a TTree.                //
//                 a convenient place to draw/fit your histograms.        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TTree.h"
#include "THashList.h"
#include "TError.h"
#include "TSelectorCint.h"
#include "Api.h"

ClassImp(TSelector)

//______________________________________________________________________________
TSelector::TSelector() : TObject()
{
   // Default selector ctor.

   fObject = 0;
   fInput  = 0;
   fOutput = new THashList;
   fOutput->IsOwner();;
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
//     Bool_t TSelector::ProcessCut(Int_t entry). This function is called
//          before processing entry. It is the user's responsability to read
//          the corresponding entry in memory (may be just a partial read).
//          The function returns kTRUE if the entry must be processed,
//          kFALSE otherwise.
//     void TSelector::ProcessFill(Int_t entry). This function is called for
//          all selected events. User fills histograms in this function.
//     void TSelector::Terminate(). This function is called at the end of
//          the loop on all events.
//
//   if filename is of the form file.C, the file will be interpreted.
//   if filename is of the form file.C++, the file file.C will be compiled
//      and dynamically loaded. The corresponding binary file and shared library
//      will be deleted at the end of the function.
//   if filename is of the form file.C+, the file file.C will be compiled
//      and dynamically loaded. At next call, if file.C is older than file.o
//      and file.so, the file.C is not compiled, only file.so is loaded.
//
//   The static function returns a pointer to a TSelector object

   //Interpret/compile filename via CINT
   char localname[256];
   sprintf(localname,".L %s",filename);
   gROOT->ProcessLine(localname);

   //loop on all classes known to CINT to find the class on filename
   //that derives from TSelector
   strcpy(localname,filename);
   char *IsCompiled = strchr(localname,'+');
   char *dot        = strchr(localname,'.');
   if (dot) dot[0] = 0;

   G__ClassInfo cl;
   Bool_t OK = kFALSE;
   while (cl.Next()) {
      if (strcmp(cl.Name(),localname)) continue;
      if (cl.IsBase("TSelector")) OK = kTRUE;
      break;
   }
   if (!OK) {
      ::Error("TSelector::GetSelector","file %s does not have a valid class deriving from TSelector",filename);
      return 0;
   }

   // we can now create an instance of the class
   TSelector *selector = (TSelector*)cl.New();
   if (!selector || IsCompiled) return selector;
   //interpreted selector: cannot be used as such
   //create a fake selector
   TSelectorCint *select = new TSelectorCint();
   select->Build(selector,&cl);

   return select;
}

