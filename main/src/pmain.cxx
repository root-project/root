// @(#)root/main:$Name:  $:$Id: pmain.cxx,v 1.3 2001/04/20 17:56:50 rdm Exp $
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

#include "TROOT.h"
#include "TProofServ.h"
#include "TPluginManager.h"

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
   TProofServ *theApp = 0;

   // The third argument is the plugin identifier
   if (argc > 2) {
     TPluginHandler *h = 0;
     if ((h = gROOT->GetPluginManager()->FindHandler("TProofServ",argv[2])) &&
        h->LoadPlugin() == 0) {
        theApp = (TProofServ *) h->ExecPlugin(2, &argc, argv);
     }
   }

   // Starndard server as default
   if (!theApp) {
      theApp = new TProofServ(&argc, argv);
   }

   // Actual server creation
   theApp->CreateServer();

   // Ready to run
   theApp->Run();

   delete theApp;

   return 0;
}
