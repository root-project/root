// @(#)root/gui:$Id$
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TRootApplication
    \ingroup guiwidgets

This class create the ROOT native GUI version of the ROOT
application environment. This in contrast to the Win32 version.
Once the native widgets work on Win32 this class can be folded into
the TApplication class (since all graphic will go via TVirtualX).

*/


#include "TRootApplication.h"
#include "TSystem.h"
#include "TString.h"
#include "TGClient.h"
#include "TVirtualX.h"

ClassImp(TRootApplication);

////////////////////////////////////////////////////////////////////////////////
/// Create ROOT application environment.

TRootApplication::TRootApplication(const char *appClassName,
                                   Int_t *argc, char **argv)
{
   fApplicationName = appClassName;
   fDisplay         = 0;

   GetOptions(argc, argv);

   if (!fDisplay)
      // Set DISPLAY based on utmp (only if DISPLAY is not yet set).
      gSystem->SetDisplay();

   fClient = new TGClient(fDisplay);

   if (fClient->IsZombie()) {
      delete fClient;
      fClient = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete ROOT application environment.

TRootApplication::~TRootApplication()
{
   delete [] fDisplay;
   delete fClient;
}

////////////////////////////////////////////////////////////////////////////////
/// By default (for UNIX) ROOT is a single thread application
/// For win32gdk returns kTRUE if it's called from inside of server/cmd thread

Bool_t TRootApplication::IsCmdThread()
{
   return gVirtualX ? gVirtualX->IsCmdThread() : kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle command line arguments. Arguments handled are removed from the
/// argument array. Currently only option "-display xserver" is considered.

void TRootApplication::GetOptions(Int_t *argc, char **argv)
{
   if (!argc) return;

   int i, j;
   for (i = 0; i < *argc; i++) {
      if (argv[i] && !strcmp(argv[i], "-display")) {
         if (argv[i+1] && strlen(argv[i+1]) && argv[i+1][0] != '-') {
            fDisplay  = StrDup(argv[i+1]);
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

