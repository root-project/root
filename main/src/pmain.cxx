// @(#)root/main:$Name$:$Id$
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

TROOT root("Proofserv","The PROOF Server");

//______________________________________________________________________________
int main(int argc, char **argv)
{
   gROOT->SetBatch();
   TProofServ *theApp = new TProofServ(&argc, argv);

   theApp->Run();

   delete theApp;

   return(0);
}
