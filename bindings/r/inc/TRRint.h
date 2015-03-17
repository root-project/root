// @(#)root/r:$Id$
// Author: Omar Zapata   30/05/2014


/*************************************************************************
 * Copyright (C)  2014, Omar Andres Zapata Mesa                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRRINT
#define ROOT_R_TRRINT

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef ROOT_TRint
#include<TRint.h>
#endif

//________________________________________________________________________________________________________
/**
   This is TRint class for R


   @ingroup R
*/
//NOTE:the arguments for TRint like Int_t*argc and char** is not supported for Rcpp,
//then the class was rewritten
namespace ROOT {
   namespace R {

      class TRRint: public TRint {
      public:
         TRRint(): TRint("ROOTR", 0, 0, 0, 0, kTRUE){}
         TRRint(TString name): TRint(name.Data(), 0, 0, 0, 0, kTRUE){}
         Long_t ProcessLine(TString line){return TRint::ProcessLine(line.Data());}
         TString WorkingDirectory(){return TRint::WorkingDirectory();}
      };
   }
}


ROOTR_MODULE(ROOTR_TRRint)
{

   ROOT::R::class_<ROOT::R::TRRint>("TRRint", "TRint class to create a ROOT application.")
   .constructor()
   .constructor<TString>()
   .method("ProcessLine", (Long_t (ROOT::R::TRRint::*)(TString))&ROOT::R::TRRint::ProcessLine)
   .method("Run", (void (ROOT::R::TRRint::*)(Bool_t))&ROOT::R::TRRint::Run)
   .method("Terminate", (void (ROOT::R::TRRint::*)(Int_t))&ROOT::R::TRRint::Terminate)
   .method("WorkingDirectory", (TString (ROOT::R::TRRint::*)())(&ROOT::R::TRRint::WorkingDirectory))
   ;
}

#endif
