// @(#)root/gui:$Id$
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

#include "RConfigure.h"

#include "TGApplication.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TGClient.h"
#include "TPluginManager.h"
#include "TError.h"
#include "TEnv.h"
#include "TVirtualX.h"
#include "TStyle.h"
#include "TInterpreter.h"
#include "TColor.h"

ClassImp(TGApplication);

////////////////////////////////////////////////////////////////////////////////
/// Create a GUI application environment. Use this class if you only
/// want to use the ROOT GUI and no other services. In all other cases
/// use either TApplication or TRint.

TGApplication::TGApplication(const char *appClassName,
                             int *argc, char **argv, void*, int)
   : TApplication(), fDisplay(0), fClient(0)
{
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
   if (argv && argv[0])
      gSystem->SetProgname(argv[0]);

   // Tell TSystem the TApplication has been created
   gSystem->NotifyApplicationCreated();

   LoadGraphicsLibs();

   if (!fDisplay) gSystem->SetDisplay();
   fClient = new TGClient(fDisplay);

   if (fClient->IsZombie()) {
      Error("TGApplication", "cannot switch to batch mode, exiting...");
      gSystem->Exit(1);
   }

   // a GUI application is never run in batch mode
   gROOT->SetBatch(kFALSE);

   if (strcmp(appClassName, "proofserv")) {
      const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                          TROOT::GetTTFFontDir());
      char *ttfont = gSystem->Which(ttpath, "arialbd.ttf", kReadPermission);
      // Added by cholm for use of DFSG - fonts - based on fix by Kevin
      if (!ttfont)
         ttfont = gSystem->Which(ttpath, "FreeSansBold.ttf", kReadPermission);
      if (ttfont && gEnv->GetValue("Root.UseTTFonts", 1)) {
         TPluginHandler *h;
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualX", "x11ttf")))
            if (h->LoadPlugin() == -1)
               Info("TGApplication", "no TTF support");
      }

      delete [] ttfont;
   }

   // Create the canvas colors early so they are allocated before
   // any color table expensive bitmaps get allocated in GUI routines (like
   // creation of XPM bitmaps).
   TColor::InitializeColors();

   // Set default screen factor (if not disabled in rc file)
   if (gEnv->GetValue("Canvas.UseScreenFactor", 1)) {
      Int_t  x, y;
      UInt_t w, h;
      if (gVirtualX) {
         gVirtualX->GetGeometry(-1, x, y, w, h);
         if (h > 0 && h < 1000) gStyle->SetScreenFactor(0.0011*h);
      }
   }

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // to allow user to interact with TCanvas's under WIN32
   gROOT->SetLineHasBeenProcessed();
}

////////////////////////////////////////////////////////////////////////////////
/// TGApplication dtor.

TGApplication::~TGApplication()
{
   delete fDisplay;
   delete fClient;
}

////////////////////////////////////////////////////////////////////////////////
/// Load shared libs necessary for GUI.

void TGApplication::LoadGraphicsLibs()
{
   TString name;
   TString title1 = "ROOT interface to ";
   TString nativex, title;
#ifndef R__WIN32
   nativex = "x11";
   name    = "X11";
   title   = title1 + "X11";
#else
   nativex = "win32gdk";
   name    = "Win32gdk";
   title   = title1 + "Win32gdk";
#endif

   TString guiBackend(gEnv->GetValue("Gui.Backend", "native"));
   guiBackend.ToLower();
   if (guiBackend == "native") {
      guiBackend = nativex;
   } else {
      name   = guiBackend;
      title  = title1 + guiBackend;
   }

   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualX", guiBackend))) {
      if (h->LoadPlugin() == -1)
         return;
      gVirtualX = (TVirtualX *) h->ExecPlugin(2, name.Data(), title.Data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle command line arguments. Arguments handled are removed from the
/// argument array. Currently only option "-display xserver" is considered.

void TGApplication::GetOptions(Int_t *argc, char **argv)
{
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

