// @(#)root/gui:$Name:$:$Id:$
// Author: Guy Barrand   30/05/2001

/*************************************************************************
 * Copyright (C) 2001, Guy Barrand.                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGApplication                                                        //
//                                                                      //
// This class initialize the ROOT GUI toolkit.                          //
// This class must be instantiated exactly once in any given            //
// application.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TGApplication.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TGClient.h"

#include "TError.h"
#include "TEnv.h"
#include "TVirtualX.h"
#include "TStyle.h"
#include "TInterpreter.h"

ClassImp(TGApplication)

//_____________________________________________________________________________
TGApplication::TGApplication(const char *appClassName,
                             int *argc, char **argv, void*, int)
   : TApplication(), fDisplay(0), fClient(0)
{
   // Create a GUI application environment. Use this class if you only
   // want to use the ROOT GUI and no other services. In all other cases
   // use either TApplication or TRint.

   if (gApplication) {
      Error("TGApplication", "only one instance of TGApplication allowed");
      return;
   }

   if (!gROOT)
      ::Fatal("TGApplication::TGApplication", "ROOT system not initialized");

   if (!gSystem)
      ::Fatal("TGApplication::TGApplication", "gSystem not initialized");

   gApplication = this;
   gROOT->SetApplication(this);
   gROOT->SetName(appClassName);

   GetOptions(argc, argv);

   LoadGraphicsLibs();

   if (!fDisplay) gSystem->SetDisplay();
   fClient = new TGClient(fDisplay);

#if !defined(WIN32)
   if (strcmp(appClassName, "proofserv")) {
      const char *ttpath = gEnv->GetValue("Root.TTFontPath",
# ifdef TTFFONTDIR
                                          TTFFONTDIR);
# else
                                          "$(HOME)/ttf/fonts");
# endif
      char *ttfont = gSystem->Which(ttpath, "arialbd.ttf", kReadPermission);

      if (!gROOT->IsBatch() && ttfont && gEnv->GetValue("Root.UseTTFonts", 1))
         gROOT->LoadClass("TGX11TTF", "GX11TTF");

      delete [] ttfont;
   }
#endif

   // Create the canvas colors early so they are allocated before
   // any color table expensive bitmaps get allocated in GUI routines (like
   // creation of XPM bitmaps).
   InitializeColors();

   if (argv && argv[0])
      gSystem->SetProgname(argv[0]);

   // Set default screen factor (if not disabled in rc file)
   if (gEnv->GetValue("Canvas.UseScreenFactor", 1)) {
      Int_t  x, y;
      UInt_t w, h;
      if (gVirtualX) {
         gVirtualX->GetGeometry(-1, x, y, w, h);
         if (h > 0 && h < 1000) gStyle->SetScreenFactor(0.0011*h);
      }
   }

   // Make sure all registered dictionaries have been initialized
   gInterpreter->InitializeDictionaries();

   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   gROOT->SetLineHasBeenProcessed(); // to allow user to interact with TCanvas's under WIN32

}

//_____________________________________________________________________________
TGApplication::~TGApplication()
{
   // TGApplication dtor.

   delete fDisplay;
   delete fClient;
}

//_____________________________________________________________________________
void TGApplication::LoadGraphicsLibs()
{
  // Load shared libs neccesary for GUI.

#ifndef WIN32
   gROOT->LoadClass("TGX11", "GX11");  // implicitely loads X11 and Xpm
   gROOT->ProcessLineFast("new TGX11(\"X11\", \"ROOT interface to X11\");");
#else
   gROOT->LoadClass("TGWin32", "Win32");
   gVirtualX = (TVirtualX *) gROOT->ProcessLineFast("new TGWin32(\"Win32\", \"ROOT interface to Win32\");");
#endif
}

//______________________________________________________________________________
void TGApplication::GetOptions(Int_t *argc, char **argv)
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

