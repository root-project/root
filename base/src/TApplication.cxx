// @(#)root/base:$Name:  $:$Id: TApplication.cxx,v 1.79 2007/02/13 14:23:15 rdm Exp $
// Author: Fons Rademakers   22/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TApplication                                                         //
//                                                                      //
// This class creates the ROOT Application Environment that interfaces  //
// to the windowing system eventloop and eventhandlers.                 //
// This class must be instantiated exactly once in any given            //
// application. Normally the specific application class inherits from   //
// TApplication (see TRint).                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
#endif

#include "Riostream.h"
#include "TApplication.h"
#include "TGuiFactory.h"
#include "TVirtualX.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TString.h"
#include "TError.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TTimer.h"
#include "TInterpreter.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TEnv.h"
#include "TColor.h"
#include "TClassTable.h"
#include "TSystemDirectory.h"
#include "TPluginManager.h"
#include "TClassTable.h"


TApplication *gApplication = 0;
Bool_t TApplication::fgGraphInit = kFALSE;

//______________________________________________________________________________
class TIdleTimer : public TTimer {
public:
   TIdleTimer(Long_t ms) : TTimer(ms, kTRUE) { }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TIdleTimer::Notify()
{
   // Notify handler.
   gApplication->HandleIdleTimer();
   Reset();
   return kFALSE;
}


ClassImp(TApplication)

//______________________________________________________________________________
TApplication::TApplication()
{
   // Default ctor. Can be used by classes deriving from TApplication.

   fArgc          = 0;
   fArgv          = 0;
   fAppImp        = 0;
   fIsRunning     = kFALSE;
   fReturnFromRun = kFALSE;
   fNoLog         = kFALSE;
   fNoLogo        = kFALSE;
   fQuit          = kFALSE;
   fFiles         = 0;
   fIdleTimer     = 0;
   fSigHandler    = 0;
}

//______________________________________________________________________________
TApplication::TApplication(const char *appClassName,
                           Int_t *argc, char **argv, void *options,
                           Int_t numOptions)
{
   // Create an application environment. The application environment
   // provides an interface to the graphics system and eventloop
   // (be it X, Windoze, MacOS or BeOS). After creating the application
   // object start the eventloop by calling its Run() method. The command
   // line options recogized by TApplication are described in the GetOptions()
   // method. The recognized options are removed from the argument array.
   // The original list of argument options can be retrieved via the Argc()
   // and Argv() methods. The appClassName "proofserv" is reserved for the
   // PROOF system. The "options" and "numOptions" arguments are not used,
   // except if you want to by-pass the argv processing by GetOptions()
   // in which case you should specify numOptions<0. All options will
   // still be available via the Argv() method for later use.

   if (gApplication) {
      Error("TApplication", "only one instance of TApplication allowed");
      return;
   }

   if (!gROOT)
      ::Fatal("TApplication::TApplication", "ROOT system not initialized");

   if (!gSystem)
      ::Fatal("TApplication::TApplication", "gSystem not initialized");

   gApplication = this;
   gROOT->SetApplication(this);
   gROOT->SetName(appClassName);

   if (options) { }  // use unused argument

   // copy command line arguments, can be later accessed via Argc() and Argv()
   if (argc && *argc > 0) {
      fArgc = *argc;
      fArgv = (char **)new char*[fArgc];
   } else {
      fArgc = 0;
      fArgv = 0;
   }

   for (int i = 0; i < fArgc; i++)
      fArgv[i] = StrDup(argv[i]);

   fNoLog         = kFALSE;
   fNoLogo        = kFALSE;
   fQuit          = kFALSE;

   if (numOptions >= 0)
      GetOptions(argc, argv);

   if (fArgv)
      gSystem->SetProgname(fArgv[0]);

   fIdleTimer     = 0;
   fSigHandler    = 0;
   fIsRunning     = kFALSE;
   fReturnFromRun = kFALSE;
   fAppImp        = gGuiFactory->CreateApplicationImp(appClassName, argc, argv);

   // Enable autoloading
   gInterpreter->EnableAutoLoading();

   // Initialize the graphics environment
   if (gClassTable->GetDict("TPad") && !fgGraphInit)
      InitializeGraphics();

   // Make sure all registered dictionaries have been initialized
   // and that all types have been loaded
   gInterpreter->InitializeDictionaries();
   gInterpreter->UpdateListOfTypes();

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // to allow user to interact with TCanvas's under WIN32
   gROOT->SetLineHasBeenProcessed();
}

//______________________________________________________________________________
TApplication::~TApplication()
{
   // TApplication dtor.

   for (int i = 0; i < fArgc; i++)
      if (fArgv[i]) delete [] fArgv[i];
   delete [] fArgv;
   SafeDelete(fAppImp);
}

//______________________________________________________________________________
void TApplication::InitializeGraphics()
{
   // Initialize the graphics environment.

   // Load the graphics related libraries
   LoadGraphicsLibs();

   // Create WM dependent application environment
   if (fAppImp)
      delete fAppImp;
   fAppImp = gGuiFactory->CreateApplicationImp(gROOT->GetName(), &fArgc, fArgv);
   if (!fAppImp) {
      MakeBatch();
      fAppImp = gGuiFactory->CreateApplicationImp(gROOT->GetName(), &fArgc, fArgv);
   }

   // Try to load TrueType font renderer. Only try to load if not in batch
   // mode and Root.UseTTFonts is true and Root.TTFontPath exists. Abort silently
   // if libttf or libGX11TTF are not found in $ROOTSYS/lib or $ROOTSYS/ttf/lib.
   if (strcmp(gROOT->GetName(), "proofserv")) {
      const char *ttpath = gEnv->GetValue("Root.TTFontPath",
#ifdef TTFFONTDIR
                                          TTFFONTDIR);
#else
                                          "$(ROOTSYS)/fonts");
#endif
      char *ttfont = gSystem->Which(ttpath, "arialbd.ttf", kReadPermission);
      // Check for use of DFSG - fonts
      if (!ttfont)
         ttfont = gSystem->Which(ttpath, "FreeSansBold.ttf", kReadPermission);

#if !defined(R__WIN32)
      if (!gROOT->IsBatch() && !strcmp(gVirtualX->GetName(), "X11") &&
          ttfont && gEnv->GetValue("Root.UseTTFonts", 1)) {
         TString plugin = "x11ttf";

         TPluginHandler *h;
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualX", plugin)))
            h->LoadPlugin();
      }
#endif
      delete [] ttfont;
   }

   // Create the canvas colors early so they are allocated before
   // any color table expensive bitmaps get allocated in GUI routines (like
   // creation of XPM bitmaps).
   InitializeColors();

   // Hook for further initializing the WM dependent application environment
   Init();

   // Set default screen factor (if not disabled in rc file)
   if (gEnv->GetValue("Canvas.UseScreenFactor", 1)) {
      Int_t  x, y;
      UInt_t w, h;
      if (gVirtualX) {
         gVirtualX->GetGeometry(-1, x, y, w, h);
         if (h > 0 && h < 1000) gStyle->SetScreenFactor(0.0011*h);
      }
   }
   fgGraphInit = kTRUE;
}

//______________________________________________________________________________
void TApplication::ClearInputFiles()
{
   // Clear list containing macro files passed as program arguments.
   // This method is called from TRint::Run() to ensure that the macro
   // files are only executed the first time Run() is called.

   if (fFiles) {
      fFiles->Delete();
      SafeDelete(fFiles);
   }
}

//______________________________________________________________________________
void TApplication::GetOptions(Int_t *argc, char **argv)
{
   // Get and handle command line options. Arguments handled are removed
   // from the argument array. The following arguments are handled:
   //    -b : run in batch mode without graphics
   //    -n : do not execute logon and logoff macros as specified in .rootrc
   //    -q : exit after processing command line macro files
   //    -l : do not show splash screen
   // The last three options are only relevant in conjunction with TRint.
   // The following help and info arguments are supported:
   //    -?      : print usage
   //    -h      : print usage
   //    --help  : print usage
   //    -config : print ./configure options

   fNoLog = kFALSE;
   fQuit  = kFALSE;
   fFiles = 0;

   if (!argc)
      return;

   int i, j;

   for (i = 1; i < *argc; i++) {
      if (!strcmp(argv[i], "-?") || !strncmp(argv[i], "-h", 2) ||
          !strncmp(argv[i], "--help", 6)) {
         fprintf(stderr, "Usage: %s [-l] [-b] [-n] [-q] [dir] [file1.C ... fileN.C]\n", argv[0]);
         fprintf(stderr, "Options:\n");
         fprintf(stderr, "  -b : run in batch mode without graphics\n");
         fprintf(stderr, "  -n : do not execute logon and logoff macros as specified in .rootrc\n");
         fprintf(stderr, "  -q : exit after processing command line macro files\n");
         fprintf(stderr, "  -l : do not show splash screen\n");
         fprintf(stderr, " dir : if dir is a valid directory cd to it before executing\n");
         fprintf(stderr, "\n");
         fprintf(stderr, "  -?      : print usage\n");
         fprintf(stderr, "  -h      : print usage\n");
         fprintf(stderr, "  --help  : print usage\n");
         fprintf(stderr, "  -config : print ./configure options\n");
         fprintf(stderr, "\n");
         Terminate(0);
      } else if (!strcmp(argv[i], "-config")) {
         fprintf(stderr, "ROOT ./configure options:\n%s\n", gROOT->GetConfigOptions());
         Terminate(0);
      } else if (!strcmp(argv[i], "-b")) {
         MakeBatch();
         argv[i] = 0;
#if 0
      } else if (!strcmp(argv[i], "-x")) {
         // remote ROOT display server not operational yet
#ifndef R__WIN32
         if (gVirtualX != gGXBatch) delete gVirtualX;
#endif
         gVirtualX = new TGXClient("Root_Client");
         if (gVirtualX->IsZombie()) {
            delete gVirtualX;
            gROOT->SetBatch();
            if (gGuiFactory != gBatchGuiFactory) delete gGuiFactory;
            gGuiFactory = gBatchGuiFactory;
            gVirtualX = gGXBatch;
         }
         argv[i] = 0;
#endif
      } else if (!strcmp(argv[i], "-n")) {
         fNoLog = kTRUE;
         argv[i] = 0;
      } else if (!strcmp(argv[i], "-q")) {
         fQuit = kTRUE;
         argv[i] = 0;
      } else if (!strcmp(argv[i], "-l")) {
         // used by front-end program to not display splash screen
         fNoLogo = kTRUE;
         argv[i] = 0;
      } else if (!strcmp(argv[i], "-splash")) {
         // used when started by front-end program to signal that
         // splash screen can be popped down (TRint::PrintLogo())
         argv[i] = 0;
      } else if (argv[i][0] != '-' && argv[i][0] != '+') {
         Long64_t size;
         Long_t id, flags, modtime;
         char *arg = strchr(argv[i], '(');
         if (arg) *arg = '\0';
         char *dir = gSystem->ExpandPathName(argv[i]);
         if (arg) *arg = '(';
         if (!gSystem->GetPathInfo(dir, &id, &size, &flags, &modtime)) {
            if ((flags & 2)) {
               // if directory make it working directory
               gSystem->ChangeDirectory(dir);
               TSystemDirectory *workdir = new TSystemDirectory("workdir", gSystem->WorkingDirectory());
               TObject *w = gROOT->GetListOfBrowsables()->FindObject("workdir");
               TObjLink *lnk = gROOT->GetListOfBrowsables()->FirstLink();
               while (lnk) {
                  if (lnk->GetObject() == w) {
                     lnk->SetObject(workdir);
                     lnk->SetOption(gSystem->WorkingDirectory());
                     break;
                  }
                  lnk = lnk->Next();
               }
               delete w;
            } else if (flags == 0 || flags == 1) {
               // if file add to list of files to be processed
               if (!fFiles) fFiles = new TObjArray;
               fFiles->Add(new TObjString(argv[i]));
            }
            argv[i] = 0;
         } else {
            char *mac, *s = strtok(dir, "+(");
            if ((mac = gSystem->Which(TROOT::GetMacroPath(), s,
                                      kReadPermission))) {
               // if file add to list of files to be processed
               if (!fFiles) fFiles = new TObjArray;
               fFiles->Add(new TObjString(argv[i]));
               argv[i] = 0;
               delete [] mac;
            } else
               // only warn if we're plain root,
               // other progs might have their own params
               if (!strcmp(GetName(),"Rint"))
                  Warning("GetOptions", "Macro %s not found.", s);
         }
         delete [] dir;
      }
      // ignore unknown options
   }

   // remove handled arguments from argument array
   j = 0;
   for (i = 0; i < *argc; i++) {
      if (argv[i]) {
         argv[j] = argv[i];
         j++;
      }
   }

   *argc = j;
}

//______________________________________________________________________________
void TApplication::HandleIdleTimer()
{
   // Handle idle timeout. When this timer expires the registered idle command
   // will be executed by this routine and a signal will be emitted.

   if (!fIdleCommand.IsNull())
      ProcessLine(GetIdleCommand());

   Emit("HandleIdleTimer()");
}

//______________________________________________________________________________
void TApplication::Help(const char *line)
{
   // Print help on interpreter.

   gInterpreter->ProcessLine(line);

   Printf("\nROOT special commands.");
   Printf("===========================================================================");
   Printf("             pwd          : show current directory, pad and style");
   Printf("             ls           : list contents of current directory");
   Printf("             which [file] : shows path of macro file");
}

//______________________________________________________________________________
void TApplication::InitializeColors()
{
   // Initialize colors used by the TCanvas based graphics (via TColor objects).
   // This method should be called before the ApplicationImp is created (which
   // initializes the GUI colors).

   if (gROOT->GetListOfColors()->First() == 0) {
      TColor *s0;
      Float_t r, g, b, h, l, s;
      Int_t   i;

      new TColor(kWhite,1,1,1,"background");
      new TColor(kBlack,0,0,0,"black");
      new TColor(kRed,1,0,0,"red");
      new TColor(kGreen,0,1,0,"green");
      new TColor(kBlue,0,0,1,"blue");
      new TColor(kYellow,1,1,0,"yellow");
      new TColor(kMagenta,1,0,1,"magenta");
      new TColor(kCyan,0,1,1,"cyan");
      new TColor(10,0.999,0.999,0.999,"white");
      new TColor(11,0.754,0.715,0.676,"editcol");

      // The color white above is defined as being nearly white.
      // Sets the associated dark color also to white.
      TColor *c110 = gROOT->GetColor(110);
      c110->SetRGB(0.999,0.999,.999);

      // Initialize Custom colors
      new TColor(20,0.8,0.78,0.67);
      new TColor(31,0.54,0.66,0.63);
      new TColor(41,0.83,0.81,0.53);
      new TColor(30,0.52,0.76,0.64);
      new TColor(32,0.51,0.62,0.55);
      new TColor(24,0.70,0.65,0.59);
      new TColor(21,0.8,0.78,0.67);
      new TColor(47,0.67,0.56,0.58);
      new TColor(35,0.46,0.54,0.57);
      new TColor(33,0.68,0.74,0.78);
      new TColor(39,0.5,0.5,0.61);
      new TColor(37,0.43,0.48,0.52);
      new TColor(38,0.49,0.6,0.82);
      new TColor(36,0.41,0.51,0.59);
      new TColor(49,0.58,0.41,0.44);
      new TColor(43,0.74,0.62,0.51);
      new TColor(22,0.76,0.75,0.66);
      new TColor(45,0.75,0.51,0.47);
      new TColor(44,0.78,0.6,0.49);
      new TColor(26,0.68,0.6,0.55);
      new TColor(28,0.53,0.4,0.34);
      new TColor(25,0.72,0.64,0.61);
      new TColor(27,0.61,0.56,0.51);
      new TColor(23,0.73,0.71,0.64);
      new TColor(42,0.87,0.73,0.53);
      new TColor(46,0.81,0.37,0.38);
      new TColor(48,0.65,0.47,0.48);
      new TColor(34,0.48,0.56,0.6);
      new TColor(40,0.67,0.65,0.75);
      new TColor(29,0.69,0.81,0.78);

      // Initialize some additional greyish non saturated colors
      new TColor(8, 0.35,0.83,0.33);
      new TColor(9, 0.35,0.33,0.85);
      new TColor(12,.3,.3,.3,"grey12");
      new TColor(13,.4,.4,.4,"grey13");
      new TColor(14,.5,.5,.5,"grey14");
      new TColor(15,.6,.6,.6,"grey15");
      new TColor(16,.7,.7,.7,"grey16");
      new TColor(17,.8,.8,.8,"grey17");
      new TColor(18,.9,.9,.9,"grey18");
      new TColor(19,.95,.95,.95,"grey19");
      new TColor(50, 0.83,0.35,0.33);

      // Initialize the Pretty Palette Spectrum Violet->Red
      //   The color model used here is based on the HLS model which
      //   is much more suitable for creating palettes than RGB.
      //   Fixing the saturation and lightness we can scan through the
      //   spectrum of visible light by using "hue" alone.
      //   In Root hue takes values from 0 to 360.
      Float_t  saturation = 1;
      Float_t  lightness = 0.5;
      Float_t  maxHue = 280;
      Float_t  minHue = 0;
      Int_t    maxPretty = 50;
      Float_t  hue;

      for (i=0 ; i<maxPretty ; i++) {
         hue = maxHue-(i+1)*((maxHue-minHue)/maxPretty);
         TColor::HLStoRGB(hue, lightness, saturation, r, g, b);
         new TColor(i+51, r, g, b);
      }

      // Initialize special colors for x3d
      for (i = 1; i < 8; i++) {
         s0 = gROOT->GetColor(i);
         s0->GetRGB(r,g,b);
         if (i == 1) { r = 0.6; g = 0.6; b = 0.6; }
         if (r == 1) r = 0.9; if (r == 0) r = 0.1;
         if (g == 1) g = 0.9; if (g == 0) g = 0.1;
         if (b == 1) b = 0.9; if (b == 0) b = 0.1;
         TColor::RGBtoHLS(r,g,b,h,l,s);
         TColor::HLStoRGB(h,0.6*l,s,r,g,b);
         new TColor(200+4*i-3,r,g,b);
         TColor::HLStoRGB(h,0.8*l,s,r,g,b);
         new TColor(200+4*i-2,r,g,b);
         TColor::HLStoRGB(h,1.2*l,s,r,g,b);
         new TColor(200+4*i-1,r,g,b);
         TColor::HLStoRGB(h,1.4*l,s,r,g,b);
         new TColor(200+4*i  ,r,g,b);
      }
   }
}

//______________________________________________________________________________
void TApplication::LoadGraphicsLibs()
{
   // Load shared libs neccesary for graphics. These libraries are only
   // loaded when gROOT->IsBatch() is kFALSE.

   if (gROOT->IsBatch()) return;

   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPad")))
      h->LoadPlugin();

   TString name;
   TString title1 = "ROOT interface to ";
   TString nativex, title;
   TString nativeg = "root";
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
      name  = guiBackend;
      title = title1 + guiBackend;
   }
   TString guiFactory(gEnv->GetValue("Gui.Factory", "native"));
   guiFactory.ToLower();
   if (guiFactory == "native")
      guiFactory = nativeg;

   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualX", guiBackend))) {
      if (h->LoadPlugin() == -1)
         return;
      gVirtualX = (TVirtualX *) h->ExecPlugin(2, name.Data(), title.Data());
   }
   if ((h = gROOT->GetPluginManager()->FindHandler("TGuiFactory", guiFactory))) {
      if (h->LoadPlugin() == -1)
         return;
      gGuiFactory = (TGuiFactory *) h->ExecPlugin(0);
   }
}

//______________________________________________________________________________
void TApplication::MakeBatch()
{
   // Switch to batch mode.

   gROOT->SetBatch();
   if (gGuiFactory != gBatchGuiFactory) delete gGuiFactory;
   gGuiFactory = gBatchGuiFactory;
#ifndef R__WIN32
   if (gVirtualX != gGXBatch) delete gVirtualX;
#endif
   gVirtualX = gGXBatch;
}

//______________________________________________________________________________
Long_t TApplication::ProcessLine(const char *line, Bool_t sync, Int_t *err)
{
   // Process a single command line, either a C++ statement or an interpreter
   // command starting with a ".".
   // Return the return value of the command casted to a long.

   if (!line || !*line) return 0;

   if (!strncasecmp(line, ".qqqqqqq", 7)) {
      gSystem->Abort();
   } else if (!strncasecmp(line, ".qqqqq", 5)) {
      Info("ProcessLine", "Bye... (try '.qqqqqqq' if still running)");
      gSystem->Exit(1);
   } else if (!strncasecmp(line, ".exit", 4) || !strncasecmp(line, ".quit", 2)) {
      gInterpreter->ResetGlobals();
      Terminate(0);
      return 0;
   }

   if (!strncmp(line, "?", 1)) {
      Help(line);
      return 1;
   }

   if (!strncmp(line, ".pwd", 4)) {
      if (gDirectory)
         Printf("Current directory: %s", gDirectory->GetPath());
      if (gPad)
         Printf("Current pad:       %s", gPad->GetName());
      if (gStyle)
         Printf("Current style:     %s", gStyle->GetName());
      return 1;
   }

   if (!strncmp(line, ".ls", 3)) {
      const char *opt = 0;
      if (line[3]) opt = &line[3];
      if (gDirectory) gDirectory->ls(opt);
      return 1;
   }

   if (!strncmp(line, ".which", 6)) {
      char *fn  = Strip(line+7);
      char *s   = strtok(fn, "+(");
      char *mac = gSystem->Which(TROOT::GetMacroPath(), s, kReadPermission);
      if (!mac)
         Printf("No macro %s in path %s", s, TROOT::GetMacroPath());
      else
         Printf("%s", mac);
      delete [] fn;
      delete [] mac;
      return mac ? 1 : 0;
   }

   if (!strncmp(line, ".L", 2) || !strncmp(line, ".U", 2)) {
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname = gSystem->SplitAclicMode(line+3, aclicMode, arguments, io);

      char *mac = gSystem->Which(TROOT::GetMacroPath(), fname, kReadPermission);
      if (arguments.Length()) {
         Warning("ProcessLine", "argument(s) \"%s\" ignored with .%c", arguments.Data(),
                 line[1],TROOT::GetMacroPath());
      }
      Long_t retval = 0;
      if (!mac)
         Error("ProcessLine", "macro %s not found in path %s", fname.Data(),
               TROOT::GetMacroPath());
      else {
         char cmd = line[1];
         if (sync)
            retval = gInterpreter->ProcessLineSynch(Form(".%c %s%s%s", cmd, mac, aclicMode.Data(),io.Data()),
                                                   (TInterpreter::EErrorCode*)err);
         else {
            retval = gInterpreter->ProcessLine(Form(".%c %s%s%s", cmd, mac, aclicMode.Data(),io.Data()),
                                              (TInterpreter::EErrorCode*)err);
         }
      }

      delete [] mac;

      return retval;
   }

   if (!strncmp(line, ".X", 2) || !strncmp(line, ".x", 2)) {
      return ProcessFile(line+3, err);
   }

   if (!strcmp(line, ".reset")) {
      // Do nothing, .reset disabled in CINT because too many side effects
      Printf("*** .reset not allowed, please use gROOT->Reset() ***");
      return 0;

#if 0
      // delete the ROOT dictionary since CINT will destroy all objects
      // referenced by the dictionary classes (TClass et. al.)
      gROOT->GetListOfClasses()->Delete();
      // fall through
#endif
   }

   if (sync)
      return gInterpreter->ProcessLineSynch(line, (TInterpreter::EErrorCode*)err);
   else
      return gInterpreter->ProcessLine(line, (TInterpreter::EErrorCode*)err);
}

//______________________________________________________________________________
Long_t TApplication::ProcessFile(const char *name, int *error)
{
   // Process a file containing a C++ macro.

   const Int_t kBufSize = 1024;

   if (!name || !*name) return 0;

   TString aclicMode;
   TString arguments;
   TString io;
   TString fname = gSystem->SplitAclicMode(name, aclicMode, arguments, io);

   char *exnam = gSystem->Which(TROOT::GetMacroPath(), fname, kReadPermission);
   if (!exnam) {
      Error("ProcessFile", "macro %s not found in path %s", fname.Data(),
            TROOT::GetMacroPath());
      delete [] exnam;
      return 0;
   }

   ::ifstream file(exnam, ios::in);
   if (!file.good()) {
      Error("ProcessFile", "%s no such file", exnam);
      delete [] exnam;
      return 0;
   }

   char currentline[kBufSize];
   int tempfile = 0;
   int comment  = 0;
   int ifndefc  = 0;
   int ifdef    = 0;
   char *s      = 0;
   Bool_t execute = kFALSE;
   Long_t retval = 0;

   while (1) {
      file.getline(currentline, kBufSize);
      if (file.eof()) break;
      s = currentline;
      while (s && (*s == ' ' || *s == '\t')) s++;   // strip-off leading blanks

      // very simple minded pre-processor parsing, only works in case macro file
      // starts with "#ifndef __CINT__". In that case everything till next
      // "#else" or "#endif" will be skipped.
      if (*s == '#') {
         char *cs = Compress(currentline);
         if (strstr(cs, "#ifndef__CINT__") ||
             strstr(cs, "#if!defined(__CINT__)"))
            ifndefc = 1;
         else if (ifndefc && (strstr(cs, "#ifdef") || strstr(cs, "#ifndef") ||
                  strstr(cs, "#ifdefined") || strstr(cs, "#if!defined")))
            ifdef++;
         else if (ifndefc && strstr(cs, "#endif")) {
            if (ifdef)
               ifdef--;
            else
               ifndefc = 0;
         } else if (ifndefc && !ifdef && strstr(cs, "#else"))
            ifndefc = 0;
         delete [] cs;
      }
      if (!*s || *s == '#' || ifndefc || !strncmp(s, "//", 2)) continue;

      if (!strncmp(s, ".X", 2) || !strncmp(s, ".x", 2)) {
         retval = ProcessFile(s+3);
         execute = kTRUE;
         continue;
      }

      if (!strncmp(s, "/*", 2)) comment = 1;
      if (comment) {
         // handle slightly more complex cases like: /*  */  /*
again:
         s = strstr(s, "*/");
         if (s) {
            comment = 0;
            s += 2;

            while (s && (*s == ' ' || *s == '\t')) s++; // strip-off leading blanks
            if (!*s) continue;
            if (!strncmp(s, "//", 2)) continue;
            if (!strncmp(s, "/*", 2)) {
               comment = 1;
               goto again;
            }
         }
      }
      if (!comment && *s == '{') tempfile = 1;
      if (!comment) break;
   }
   file.close();

   if (!execute) {
      TString exname = exnam;
      if (!tempfile) {
         // We have a script that does NOT contain an unamed macro,
         // so we can call the script compiler on it.
         exname += aclicMode;
      }
      exname += arguments;
      exname += io;

      if (tempfile) {
         retval = gInterpreter->ProcessLineSynch(Form(".x %s", exname.Data()),
                                        (TInterpreter::EErrorCode*)error);
      } else
         retval = gInterpreter->ProcessLineSynch(Form(".X %s", exname.Data()),
                                        (TInterpreter::EErrorCode*)error);
   }

   delete [] exnam;
   return retval;
}

//______________________________________________________________________________
void TApplication::Run(Bool_t retrn)
{
   // Main application eventloop. Calls system dependent eventloop via gSystem.

   SetReturnFromRun(retrn);

   fIsRunning = kTRUE;

   gSystem->Run();

   fIsRunning = kFALSE;
}

//______________________________________________________________________________
void TApplication::SetIdleTimer(UInt_t idleTimeInSec, const char *command)
{
   // Set the command to be executed after the system has been idle for
   // idleTimeInSec seconds. Normally called via TROOT::Idle(...).

   if (fIdleTimer) RemoveIdleTimer();
   fIdleCommand = command;
   fIdleTimer = new TIdleTimer(idleTimeInSec*1000);
   gSystem->AddTimer(fIdleTimer);
}

//______________________________________________________________________________
void TApplication::RemoveIdleTimer()
{
   // Remove idle timer. Normally called via TROOT::Idle(0).

   if (fIdleTimer) {
      // timers are removed from the gSystem timer list by their dtor
      SafeDelete(fIdleTimer);
   }
}

//______________________________________________________________________________
void TApplication::StartIdleing()
{
   // Called when system starts idleing.

   if (fIdleTimer) {
      fIdleTimer->Reset();
      gSystem->AddTimer(fIdleTimer);
   }
}

//______________________________________________________________________________
void TApplication::StopIdleing()
{
   // Called when system stops idleing.

   if (fIdleTimer)
      gSystem->RemoveTimer(fIdleTimer);
}

//______________________________________________________________________________
void TApplication::Terminate(Int_t status)
{
   // Terminate the application by call TSystem::Exit() unless application has
   // been told to return from Run(), by a call to SetReturnFromRun().

   Emit("Terminate(Int_t)", status);

   if (fReturnFromRun)
      gSystem->ExitLoop();
   else
      gSystem->Exit(status);
}

//______________________________________________________________________________
void TApplication::KeyPressed(Int_t key)
{
   // Emit signal when console keyboard key was pressed.

   Emit("KeyPressed(Int_t)", key);
}

//______________________________________________________________________________
void TApplication::ReturnPressed(char *text )
{
   // Emit signal when return key was pressed.

   Emit("ReturnPressed(char*)", text);
}

//______________________________________________________________________________
void TApplication::SetEchoMode(Bool_t)
{
   // Set console echo mode:
   //
   //  mode = kTRUE  - echo input symbols
   //  mode = kFALSE - noecho input symbols
}

//______________________________________________________________________________
void TApplication::CreateApplication()
{
   // Static function used to create a default application environment.

   if (!gApplication) {
      char *a = StrDup("RootApp");
      char *b = StrDup("-b");
      char *argv[2];
      Int_t argc = 2;
      argv[0] = a;
      argv[1] = b;
      new TApplication("RootApp", &argc, argv, 0, 0);
      if (gDebug > 0)
         Printf("<TApplication::CreateApplication>: "
                "created default TApplication");
      delete [] a; delete [] b;
   }
}
