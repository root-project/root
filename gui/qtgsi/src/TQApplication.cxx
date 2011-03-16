// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQApplication                                                        //
//                                                                      //
// This class create the ROOT native GUI version of the ROOT            //
// application environment. This in contrast to the Win32 version.      //
// Once the native widgets work on Win32 this class can be folded into  //
// the TApplication class (since all graphic will go via TVirtualX).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TQApplication.h"
#include "TQRootGuiFactory.h"

ClassImp(TQApplication)

//______________________________________________________________________________
TQApplication::TQApplication()
   :TApplication()
{
   // Used by Dictionary()

   fCustomized=kFALSE;
}

//______________________________________________________________________________
TQApplication::TQApplication(const char *appClassName,
                             Int_t *argc, char **argv, void *options, Int_t numOptions)
   : TApplication(appClassName,argc,argv,options,numOptions)
{
   // Create the root application and load the graphic libraries

   fCustomized=kFALSE;
   LoadGraphicsLibs();
}

//______________________________________________________________________________
TQApplication::~TQApplication()
{
   // Delete ROOT application environment.

   if (gApplication)  gApplication->Terminate(0);
}

//______________________________________________________________________________
void TQApplication::LoadGraphicsLibs()
{
   // Here we overload the LoadGraphicsLibs() function.
   // This function now just instantiates a QRootGuiFactory
   // object and redirect the global pointer gGuiFactory to point
   // to it.

   if (gROOT->IsBatch()) return;
   gROOT->LoadClass("TCanvas", "Gpad");
   gGuiFactory =  new TQRootGuiFactory();

}

//______________________________________________________________________________
void TQApplication::SetCustomized()
{
   // Set the custom flag

   fCustomized = kTRUE;
   if (fCustomized) ((TQRootGuiFactory*) gGuiFactory)->SetCustomFlag(kTRUE);
}
