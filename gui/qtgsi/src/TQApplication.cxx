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

////////////////////////////////////////////////////////////////////////////////
/// Used by Dictionary()

TQApplication::TQApplication()
   :TApplication()
{
   fCustomized=kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the root application and load the graphic libraries

TQApplication::TQApplication(const char *appClassName,
                             Int_t *argc, char **argv, void *options, Int_t numOptions)
   : TApplication(appClassName,argc,argv,options,numOptions)
{
   fCustomized=kFALSE;
   LoadGraphicsLibs();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete ROOT application environment.

TQApplication::~TQApplication()
{
   if (gApplication)  gApplication->Terminate(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Here we overload the LoadGraphicsLibs() function.
/// This function now just instantiates a QRootGuiFactory
/// object and redirect the global pointer gGuiFactory to point
/// to it.

void TQApplication::LoadGraphicsLibs()
{
   if (gROOT->IsBatch()) return;
   gROOT->LoadClass("TCanvas", "Gpad");
   gGuiFactory =  new TQRootGuiFactory();

}

////////////////////////////////////////////////////////////////////////////////
/// Set the custom flag

void TQApplication::SetCustomized()
{
   fCustomized = kTRUE;
   if (fCustomized) ((TQRootGuiFactory*) gGuiFactory)->SetCustomFlag(kTRUE);
}
