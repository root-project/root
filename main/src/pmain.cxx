// @(#)root/main:$Name:  $:$Id: pmain.cxx,v 1.5 2006/04/19 10:57:44 rdm Exp $
// Author: Fons Rademakers   15/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// PMain                                                                //
//                                                                      //
// Main program used to create PROOF server application.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif
#include "TApplication.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

// Special type for the hook to the TXProofServ constructor, needed to avoid
// using the plugin manager
typedef TApplication *(*TProofServ_t)(Int_t *argc, char **argv);

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // PROOF server main program.

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   gROOT->SetBatch();
   TApplication *theApp = 0;

   // Enable autoloading
   gInterpreter->EnableAutoLoading();

   TString getter("GetTProofServ");
#ifdef ROOTLIBDIR
   TString prooflib = TString(ROOTLIBDIR) + "/libProof";
#else
   TString prooflib = TString(gRootDir) + "/lib/libProof";
#endif
   if (argc > 2) {
      // XPD: additionally load the appropriate library
      prooflib.ReplaceAll("/libProof", "/libProofx");
      getter.ReplaceAll("GetTProofServ", "GetTXProofServ");
   }
   char *p = 0;
   if ((p = gSystem->DynamicPathName(prooflib, kTRUE))) {
      delete[] p;
      if (gSystem->Load(prooflib) == -1) {
         Printf("%s:%s: can't load %s", argv[0], argv[1], prooflib.Data());
         return 0;
      }
   } else {
      Printf("%s:%s: can't locate %s", argv[0], argv[1], prooflib.Data());
      return 0;
   }

   // Locate constructor
   Func_t f = gSystem->DynFindSymbol(prooflib, getter);
   if (f) {
      theApp = (TApplication *) (*((TProofServ_t)f))(&argc, argv);
   } else {
      Printf("%s:%s: can't find %s", argv[0], argv[1], getter.Data());
      return 0;
   }

   // Ready to run
   theApp->Run();

   // When we return here we are done
   delete theApp;

   return 0;
}
