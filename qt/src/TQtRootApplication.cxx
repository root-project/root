// @(#)root/gui:$Name:  $:$Id: TQtRootApplication.cxx,v 1.1 2003/05/21 18:40:06 fine Exp $
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtRootApplication                                                     //
//                                                                      //
// This class create the ROOT native GUI version of the ROOT            //
// application environment. This in contrast to the Win32 version.      //
// Once the native widgets work on Win32 this class can be folded into  //
// the TApplication class (since all graphic will go via TVirtualX).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQtRootApplication.h"
#include "TSystem.h"
#include "TString.h"


// ClassImp(TQtRootApplication)

//______________________________________________________________________________
TQtRootApplication::TQtRootApplication(const char *appClassName,
                                   Int_t *argc, char **argv)
{
   fApplicationName = appClassName;
   fDisplay         = 0;

   GetOptions(argc, argv);

   if (!fDisplay)
      // Set DISPLAY based on utmp (only if DISPLAY is not yet set).
      gSystem->SetDisplay();
}

//______________________________________________________________________________
TQtRootApplication::~TQtRootApplication()
{
   // Delete ROOT application environment.

   delete fDisplay;
}

//______________________________________________________________________________
void TQtRootApplication::GetOptions(Int_t *argc, char **argv)
{
   // Handle command line arguments. Arguments handled are removed from the
   // argument array. Currently only option "-display xserver" is considered.

   if (!argc) return;

   int i, j;
   for (i = 0; i < *argc; i++) {
      if (!strcmp(argv[i], "-display")) {
         if (argv[i+1] && strlen(argv[i+1]) && argv[i+1][0] != '-') {
            fDisplay = StrDup(argv[i+1]);
            argv[i]   = 0;
            argv[i+1] = 0;
            i++;
         }
      }
   }

   j = 0;
   for (i = 0; i < *argc; i++) {
      if (argv[i]) {
         argv[j] = argv[i];
         j++;
      }
   }

   *argc = j;
}

