// @(#)root/base:$Id$
// Author: Fons Rademakers   22/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TApplication
\ingroup Base

This class creates the ROOT Application Environment that interfaces
to the windowing system eventloop and eventhandlers.
This class must be instantiated exactly once in any given
application. Normally the specific application class inherits from
TApplication (see TRint).
*/

#include "RConfigure.h"
#include "Riostream.h"
#include "TApplication.h"
#include "TException.h"
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
#include "TPluginManager.h"
#include "TClassTable.h"
#include "TBrowser.h"
#include "TUrl.h"
#include "TVirtualMutex.h"

#include "TApplicationCommandLineOptionsHelp.h"

#include <stdlib.h>

TApplication *gApplication = 0;
Bool_t TApplication::fgGraphNeeded = kFALSE;
Bool_t TApplication::fgGraphInit = kFALSE;
TList *TApplication::fgApplications = 0;  // List of available applications

////////////////////////////////////////////////////////////////////////////////

class TIdleTimer : public TTimer {
public:
   TIdleTimer(Long_t ms) : TTimer(ms, kTRUE) { }
   Bool_t Notify();
};

////////////////////////////////////////////////////////////////////////////////
/// Notify handler.

Bool_t TIdleTimer::Notify()
{
   gApplication->HandleIdleTimer();
   Reset();
   return kFALSE;
}


ClassImp(TApplication);

static void CallEndOfProcessCleanups()
{
   // Insure that the files, canvases and sockets are closed.

   // If we get here, the tear down has started.  We have no way to know what
   // has or has not yet been done.  In particular on Ubuntu, this was called
   // after the function static in TSystem.cxx has been destructed.  So we
   // set gROOT in its end-of-life mode which prevents executing code, like
   // autoloading libraries (!) that is pointless ...
   if (gROOT) {
      gROOT->SetBit(kInvalidObject);
      gROOT->EndOfProcessCleanups();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Default ctor. Can be used by classes deriving from TApplication.

TApplication::TApplication() :
   fArgc(0), fArgv(0), fAppImp(0), fIsRunning(kFALSE), fReturnFromRun(kFALSE),
   fNoLog(kFALSE), fNoLogo(kFALSE), fQuit(kFALSE), fUseMemstat(kFALSE),
   fFiles(0), fIdleTimer(0), fSigHandler(0), fExitOnException(kDontExit),
   fAppRemote(0)
{
   ResetBit(kProcessRemotely);
}

////////////////////////////////////////////////////////////////////////////////
/// Create an application environment. The application environment
/// provides an interface to the graphics system and eventloop
/// (be it X, Windows, MacOS or BeOS). After creating the application
/// object start the eventloop by calling its Run() method. The command
/// line options recognized by TApplication are described in the GetOptions()
/// method. The recognized options are removed from the argument array.
/// The original list of argument options can be retrieved via the Argc()
/// and Argv() methods. The appClassName "proofserv" is reserved for the
/// PROOF system. The "options" and "numOptions" arguments are not used,
/// except if you want to by-pass the argv processing by GetOptions()
/// in which case you should specify numOptions<0. All options will
/// still be available via the Argv() method for later use.

TApplication::TApplication(const char *appClassName, Int_t *argc, char **argv,
                           void * /*options*/, Int_t numOptions) :
   fArgc(0), fArgv(0), fAppImp(0), fIsRunning(kFALSE), fReturnFromRun(kFALSE),
   fNoLog(kFALSE), fNoLogo(kFALSE), fQuit(kFALSE), fUseMemstat(kFALSE),
   fFiles(0), fIdleTimer(0), fSigHandler(0), fExitOnException(kDontExit),
   fAppRemote(0)
{
   R__LOCKGUARD(gInterpreterMutex);

   // Create the list of applications the first time
   if (!fgApplications)
      fgApplications = new TList;

   // Add the new TApplication early, so that the destructor of the
   // default TApplication (if it is called in the block of code below)
   // will not destroy the files, socket or TColor that have already been
   // created.
   fgApplications->Add(this);

   if (gApplication && gApplication->TestBit(kDefaultApplication)) {
      // allow default TApplication to be replaced by a "real" TApplication
      delete gApplication;
      gApplication = 0;
      gROOT->SetBatch(kFALSE);
      fgGraphInit = kFALSE;
   }

   if (gApplication) {
      Error("TApplication", "only one instance of TApplication allowed");
      fgApplications->Remove(this);
      return;
   }

   if (!gROOT)
      ::Fatal("TApplication::TApplication", "ROOT system not initialized");

   if (!gSystem)
      ::Fatal("TApplication::TApplication", "gSystem not initialized");

   static Bool_t hasRegisterAtExit(kFALSE);
   if (!hasRegisterAtExit) {
      // If we are the first TApplication register the atexit)
      atexit(CallEndOfProcessCleanups);
      hasRegisterAtExit = kTRUE;
   }
   gROOT->SetName(appClassName);

   // copy command line arguments, can be later accessed via Argc() and Argv()
   if (argc && *argc > 0) {
      fArgc = *argc;
      fArgv = (char **)new char*[fArgc];
   }

   for (int i = 0; i < fArgc; i++)
      fArgv[i] = StrDup(argv[i]);

   if (numOptions >= 0)
      GetOptions(argc, argv);

   if (fArgv)
      gSystem->SetProgname(fArgv[0]);

   // Tell TSystem the TApplication has been created
   gSystem->NotifyApplicationCreated();

   fAppImp = gGuiFactory->CreateApplicationImp(appClassName, argc, argv);
   ResetBit(kProcessRemotely);

   // Initialize the graphics environment
   if (gClassTable->GetDict("TPad")) {
      fgGraphNeeded = kTRUE;
      InitializeGraphics();
   }

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // to allow user to interact with TCanvas's under WIN32
   gROOT->SetLineHasBeenProcessed();

   // activate TMemStat
   if (fUseMemstat || gEnv->GetValue("Root.TMemStat", 0)) {
      fUseMemstat = kTRUE;
      Int_t buffersize = gEnv->GetValue("Root.TMemStat.buffersize", 100000);
      Int_t maxcalls   = gEnv->GetValue("Root.TMemStat.maxcalls", 5000000);
      const char *ssystem = gEnv->GetValue("Root.TMemStat.system","gnubuiltin");
      if (maxcalls > 0) {
         gROOT->ProcessLine(Form("new TMemStat(\"%s\",%d,%d);",ssystem,buffersize,maxcalls));
      }
   }

   //Needs to be done last
   gApplication = this;
   gROOT->SetApplication(this);

}

////////////////////////////////////////////////////////////////////////////////
/// TApplication dtor.

TApplication::~TApplication()
{
   for (int i = 0; i < fArgc; i++)
      if (fArgv[i]) delete [] fArgv[i];
   delete [] fArgv;

   if (fgApplications)
      fgApplications->Remove(this);

   //close TMemStat
   if (fUseMemstat) {
      ProcessLine("TMemStat::Close()");
      fUseMemstat = kFALSE;
   }

   // Reduce the risk of the files or sockets being closed after the
   // end of 'main' (or more exactly before the library start being
   // unloaded).
   if (fgApplications == 0 || fgApplications->FirstLink() == 0 ) {
      TROOT::ShutDown();
   }

   // Now that all the canvases and files have been closed we can
   // delete the implementation.
   SafeDelete(fAppImp);
}

////////////////////////////////////////////////////////////////////////////////
/// Static method. This method should be called from static library
/// initializers if the library needs the low level graphics system.

void TApplication::NeedGraphicsLibs()
{
   fgGraphNeeded = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the graphics environment.

void TApplication::InitializeGraphics()
{
   if (fgGraphInit || !fgGraphNeeded) return;

   // Load the graphics related libraries
   LoadGraphicsLibs();

   // Try to load TrueType font renderer. Only try to load if not in batch
   // mode and Root.UseTTFonts is true and Root.TTFontPath exists. Abort silently
   // if libttf or libGX11TTF are not found in $ROOTSYS/lib or $ROOTSYS/ttf/lib.
   const char *ttpath = gEnv->GetValue("Root.TTFontPath",
                                       TROOT::GetTTFFontDir());
   char *ttfont = gSystem->Which(ttpath, "arialbd.ttf", kReadPermission);
   // Check for use of DFSG - fonts
   if (!ttfont)
      ttfont = gSystem->Which(ttpath, "FreeSansBold.ttf", kReadPermission);

#if !defined(R__WIN32)
   if (!gROOT->IsBatch() && !strcmp(gVirtualX->GetName(), "X11") &&
       ttfont && gEnv->GetValue("Root.UseTTFonts", 1)) {
      if (gClassTable->GetDict("TGX11TTF")) {
         // in principle we should not have linked anything against libGX11TTF
         // but with ACLiC this can happen, initialize TGX11TTF by hand
         // (normally this is done by the static library initializer)
         ProcessLine("TGX11TTF::Activate();");
      } else {
         TPluginHandler *h;
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualX", "x11ttf")))
            if (h->LoadPlugin() == -1)
               Info("InitializeGraphics", "no TTF support");
      }
   }
#endif
   delete [] ttfont;

   // Create WM dependent application environment
   if (fAppImp)
      delete fAppImp;
   fAppImp = gGuiFactory->CreateApplicationImp(gROOT->GetName(), &fArgc, fArgv);
   if (!fAppImp) {
      MakeBatch();
      fAppImp = gGuiFactory->CreateApplicationImp(gROOT->GetName(), &fArgc, fArgv);
   }

   // Create the canvas colors early so they are allocated before
   // any color table expensive bitmaps get allocated in GUI routines (like
   // creation of XPM bitmaps).
   TColor::InitializeColors();

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
}

////////////////////////////////////////////////////////////////////////////////
/// Clear list containing macro files passed as program arguments.
/// This method is called from TRint::Run() to ensure that the macro
/// files are only executed the first time Run() is called.

void TApplication::ClearInputFiles()
{
   if (fFiles) {
      fFiles->Delete();
      SafeDelete(fFiles);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return specified argument.

char *TApplication::Argv(Int_t index) const
{
   if (fArgv) {
      if (index >= fArgc) {
         Error("Argv", "index (%d) >= number of arguments (%d)", index, fArgc);
         return 0;
      }
      return fArgv[index];
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get and handle command line options. Arguments handled are removed
/// from the argument array. See CommandLineOptionsHelp.h for options.

void TApplication::GetOptions(Int_t *argc, char **argv)
{
   static char null[1] = { "" };

   fNoLog = kFALSE;
   fQuit  = kFALSE;
   fFiles = 0;

   if (!argc)
      return;

   int i, j;
   TString pwd;

   for (i = 1; i < *argc; i++) {
      if (!strcmp(argv[i], "-?") || !strncmp(argv[i], "-h", 2) ||
          !strncmp(argv[i], "--help", 6)) {
         fprintf(stderr, kCommandLineOptionsHelp);
         Terminate(0);
      } else if (!strncmp(argv[i], "--version", 9)) {
         fprintf(stderr, "ROOT Version: %s\n", gROOT->GetVersion());
         fprintf(stderr, "Built for %s on %s\n",
                 gSystem->GetBuildArch(),
                 gROOT->GetGitDate());

         fprintf(stderr, "From %s@%s\n",
                gROOT->GetGitBranch(),
                gROOT->GetGitCommit());

         Terminate(0);
      } else if (!strcmp(argv[i], "-config")) {
         fprintf(stderr, "ROOT ./configure options:\n%s\n", gROOT->GetConfigOptions());
         Terminate(0);
      } else if (!strcmp(argv[i], "-memstat")) {
         fUseMemstat = kTRUE;
         argv[i] = null;
      } else if (!strcmp(argv[i], "-b")) {
         MakeBatch();
         argv[i] = null;
      } else if (!strcmp(argv[i], "-n")) {
         fNoLog = kTRUE;
         argv[i] = null;
      } else if (!strcmp(argv[i], "-t")) {
         ROOT::EnableImplicitMT();
         // EnableImplicitMT() only enables thread safety if IMT was configured;
         // enable thread safety even with IMT off:
         ROOT::EnableThreadSafety();
         argv[i] = null;
      } else if (!strcmp(argv[i], "-q")) {
         fQuit = kTRUE;
         argv[i] = null;
      } else if (!strcmp(argv[i], "-l")) {
         // used by front-end program to not display splash screen
         fNoLogo = kTRUE;
         argv[i] = null;
      } else if (!strcmp(argv[i], "-x")) {
         fExitOnException = kExit;
         argv[i] = null;
      } else if (!strcmp(argv[i], "-splash")) {
         // used when started by front-end program to signal that
         // splash screen can be popped down (TRint::PrintLogo())
         argv[i] = null;
      } else if (strncmp(argv[i], "--web", 5) == 0) {
         // the web mode is requested
         const char *opt = argv[i] + 5;
         argv[i] = null;
         TString argw;
         if (gROOT->IsBatch()) argw = "batch";
         if (*opt == '=') argw.Append(opt+1);
         if (gSystem->Load("libROOTWebDisplay") >= 0) {
            gROOT->SetWebDisplay(argw.Data());
            gEnv->SetValue("Gui.Factory", "web");
         } else {
            Error("GetOptions", "--web option not supported, ROOT should be built with at least c++14 enabled");
         }
      } else if (!strcmp(argv[i], "-e")) {
         argv[i] = null;
         ++i;

         if ( i < *argc ) {
            if (!fFiles) fFiles = new TObjArray;
            TObjString *expr = new TObjString(argv[i]);
            expr->SetBit(kExpression);
            fFiles->Add(expr);
            argv[i] = null;
         } else {
            Warning("GetOptions", "-e must be followed by an expression.");
         }
      } else if (!strcmp(argv[i], "--")) {
         TObjString* macro = nullptr;
         bool warnShown = false;

         if (fFiles) {
            for (auto f: *fFiles) {
               TObjString* file = dynamic_cast<TObjString*>(f);
               if (!file) {
                  if (!dynamic_cast<TNamed*>(f)) {
                     Error("GetOptions()", "Inconsistent file entry (not a TObjString)!");
                     f->Dump();
                  } // else we did not find the file.
                  continue;
               }

               if (file->TestBit(kExpression))
                  continue;
               if (file->String().EndsWith(".root"))
                  continue;
               if (file->String().Contains('('))
                  continue;

               if (macro && !warnShown && (warnShown = true))
                  Warning("GetOptions", "-- is used with several macros. "
                                        "The arguments will be passed to the last one.");

               macro = file;
            }
         }

         if (macro) {
            argv[i] = null;
            ++i;
            TString& str = macro->String();

            str += '(';
            for (; i < *argc; i++) {
               str += argv[i];
               str += ',';
               argv[i] = null;
            }
            str.EndsWith(",") ? str[str.Length() - 1] = ')' : str += ')';
         } else {
            Warning("GetOptions", "no macro to pass arguments to was provided. "
                                  "Everything after the -- will be ignored.");
            for (; i < *argc; i++)
               argv[i] = null;
         }
      } else if (argv[i][0] != '-' && argv[i][0] != '+') {
         Long64_t size;
         Long_t id, flags, modtime;
         char *arg = strchr(argv[i], '(');
         if (arg) *arg = '\0';
         char *dir = gSystem->ExpandPathName(argv[i]);
         // ROOT-9959: we do not continue if we could not expand the path
         if (!dir) continue;
         TUrl udir(dir, kTRUE);
         // remove options and anchor to check the path
         TString sfx = udir.GetFileAndOptions();
         TString fln = udir.GetFile();
         sfx.Replace(sfx.Index(fln), fln.Length(), "");
         TString path = udir.GetFile();
         if (strcmp(udir.GetProtocol(), "file")) {
            path = udir.GetUrl();
            path.Replace(path.Index(sfx), sfx.Length(), "");
         }
         // 'path' is the full URL without suffices (options and/or anchor)
         if (arg) *arg = '(';
         if (!arg && !gSystem->GetPathInfo(path.Data(), &id, &size, &flags, &modtime)) {
            if ((flags & 2)) {
               // if directory set it in fWorkDir
               if (pwd == "") {
                  pwd = gSystem->WorkingDirectory();
                  fWorkDir = dir;
                  gSystem->ChangeDirectory(dir);
                  argv[i] = null;
               } else if (!strcmp(gROOT->GetName(), "Rint")) {
                  Warning("GetOptions", "only one directory argument can be specified (%s)", dir);
               }
            } else if (size > 0) {
               // if file add to list of files to be processed
               if (!fFiles) fFiles = new TObjArray;
               fFiles->Add(new TObjString(path.Data()));
               argv[i] = null;
            } else {
               Warning("GetOptions", "file %s has size 0, skipping", dir);
            }
         } else {
            if (TString(udir.GetFile()).EndsWith(".root")) {
               if (!strcmp(udir.GetProtocol(), "file")) {
                  // file ending on .root but does not exist, likely a typo
                  // warn user if plain root...
                  if (!strcmp(gROOT->GetName(), "Rint"))
                     Warning("GetOptions", "file %s not found", dir);
               } else {
                  // remote file, give it the benefit of the doubt and add it to list of files
                  if (!fFiles) fFiles = new TObjArray;
                  fFiles->Add(new TObjString(argv[i]));
                  argv[i] = null;
               }
            } else {
               TString mode,fargs,io;
               TString fname = gSystem->SplitAclicMode(dir,mode,fargs,io);
               char *mac;
               if (!fFiles) fFiles = new TObjArray;
               if ((mac = gSystem->Which(TROOT::GetMacroPath(), fname,
                                         kReadPermission))) {
                  // if file add to list of files to be processed
                  fFiles->Add(new TObjString(argv[i]));
                  argv[i] = null;
                  delete [] mac;
               } else {
                  // if file add an invalid entry to list of files to be processed
                  fFiles->Add(new TNamed("NOT FOUND!", argv[i]));
                  // only warn if we're plain root,
                  // other progs might have their own params
                  if (!strcmp(gROOT->GetName(), "Rint"))
                     Warning("GetOptions", "macro %s not found", fname.Data());
               }
            }
         }
         delete [] dir;
      }
      // ignore unknown options
   }

   // go back to startup directory
   if (pwd != "")
      gSystem->ChangeDirectory(pwd);

   // remove handled arguments from argument array
   j = 0;
   for (i = 0; i < *argc; i++) {
      if (strcmp(argv[i], "")) {
         argv[j] = argv[i];
         j++;
      }
   }

   *argc = j;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle idle timeout. When this timer expires the registered idle command
/// will be executed by this routine and a signal will be emitted.

void TApplication::HandleIdleTimer()
{
   if (!fIdleCommand.IsNull())
      ProcessLine(GetIdleCommand());

   Emit("HandleIdleTimer()");
}

////////////////////////////////////////////////////////////////////////////////
/// Handle exceptions (kSigBus, kSigSegmentationViolation,
/// kSigIllegalInstruction and kSigFloatingException) trapped in TSystem.
/// Specific TApplication implementations may want something different here.

void TApplication::HandleException(Int_t sig)
{
   if (TROOT::Initialized()) {
      if (gException) {
         gInterpreter->RewindDictionary();
         gInterpreter->ClearFileBusy();
      }
      if (fExitOnException == kExit)
         gSystem->Exit(128 + sig);
      else if (fExitOnException == kAbort)
         gSystem->Abort();
      else
         Throw(sig);
   }
   gSystem->Exit(128 + sig);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the exit on exception option. Setting this option determines what
/// happens in HandleException() in case an exception (kSigBus,
/// kSigSegmentationViolation, kSigIllegalInstruction or kSigFloatingException)
/// is trapped. Choices are: kDontExit (default), kExit or kAbort.
/// Returns the previous value.

TApplication::EExitOnException TApplication::ExitOnException(TApplication::EExitOnException opt)
{
   EExitOnException old = fExitOnException;
   fExitOnException = opt;
   return old;
}

////////////////////////////////////////////////////////////////////////////////
/// Print help on interpreter.

void TApplication::Help(const char *line)
{
   gInterpreter->ProcessLine(line);

   Printf("\nROOT special commands.");
   Printf("===========================================================================");
   Printf("             pwd          : show current directory, pad and style");
   Printf("             ls           : list contents of current directory");
   Printf("             which [file] : shows path of macro file");
}

////////////////////////////////////////////////////////////////////////////////
/// Load shared libs necessary for graphics. These libraries are only
/// loaded when gROOT->IsBatch() is kFALSE.

void TApplication::LoadGraphicsLibs()
{
   if (gROOT->IsBatch()) return;

   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPad")))
      if (h->LoadPlugin() == -1)
         return;

   TString name;
   TString title1 = "ROOT interface to ";
   TString nativex, title;
   TString nativeg = "root";

#ifdef R__WIN32
   nativex = "win32gdk";
   name    = "Win32gdk";
   title   = title1 + "Win32gdk";
#elif defined(R__HAS_COCOA)
   nativex = "quartz";
   name    = "quartz";
   title   = title1 + "Quartz";
#else
   nativex = "x11";
   name    = "X11";
   title   = title1 + "X11";
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
      if (h->LoadPlugin() == -1) {
         gROOT->SetBatch(kTRUE);
         return;
      }
      gVirtualX = (TVirtualX *) h->ExecPlugin(2, name.Data(), title.Data());
      fgGraphInit = kTRUE;
   }
   if ((h = gROOT->GetPluginManager()->FindHandler("TGuiFactory", guiFactory))) {
      if (h->LoadPlugin() == -1) {
         gROOT->SetBatch(kTRUE);
         return;
      }
      gGuiFactory = (TGuiFactory *) h->ExecPlugin(0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Switch to batch mode.

void TApplication::MakeBatch()
{
   gROOT->SetBatch();
   if (gGuiFactory != gBatchGuiFactory) delete gGuiFactory;
   gGuiFactory = gBatchGuiFactory;
#ifndef R__WIN32
   if (gVirtualX != gGXBatch) delete gVirtualX;
#endif
   gVirtualX = gGXBatch;
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the content of a line starting with ".R" (already stripped-off)
/// The format is
/// ~~~ {.cpp}
///      [user@]host[:dir] [-l user] [-d dbg] [script]
/// ~~~
/// The variable 'dir' is the remote directory to be used as working dir.
/// The username can be specified in two ways, "-l" having the priority
/// (as in ssh).
/// A 'dbg' value > 0 gives increasing verbosity.
/// The last argument 'script' allows to specify an alternative script to
/// be executed remotely to startup the session.

Int_t TApplication::ParseRemoteLine(const char *ln,
                                   TString &hostdir, TString &user,
                                   Int_t &dbg, TString &script)
{
   if (!ln || strlen(ln) <= 0)
      return 0;

   Int_t rc = 0;
   Bool_t isHostDir = kTRUE;
   Bool_t isScript = kFALSE;
   Bool_t isUser = kFALSE;
   Bool_t isDbg = kFALSE;

   TString line(ln);
   TString tkn;
   Int_t from = 0;
   while (line.Tokenize(tkn, from, " ")) {
      if (tkn == "-l") {
         // Next is a user name
         isUser = kTRUE;
      } else if (tkn == "-d") {
         isDbg = kTRUE;
      } else if (tkn == "-close") {
         rc = 1;
      } else if (tkn.BeginsWith("-")) {
         ::Warning("TApplication::ParseRemoteLine","unknown option: %s", tkn.Data());
      } else {
         if (isUser) {
            user = tkn;
            isUser = kFALSE;
         } else if (isDbg) {
            dbg = tkn.Atoi();
            isDbg = kFALSE;
         } else if (isHostDir) {
            hostdir = tkn;
            hostdir.ReplaceAll(":","/");
            isHostDir = kFALSE;
            isScript = kTRUE;
         } else if (isScript) {
            // Add everything left
            script = tkn;
            script.Insert(0, "\"");
            script += "\"";
            isScript = kFALSE;
            break;
         }
      }
   }

   // Done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Process the content of a line starting with ".R" (already stripped-off)
/// The format is
/// ~~~ {.cpp}
///      [user@]host[:dir] [-l user] [-d dbg] [script] | [host] -close
/// ~~~
/// The variable 'dir' is the remote directory to be used as working dir.
/// The username can be specified in two ways, "-l" having the priority
/// (as in ssh).
/// A 'dbg' value > 0 gives increasing verbosity.
/// The last argument 'script' allows to specify an alternative script to
/// be executed remotely to startup the session.

Long_t TApplication::ProcessRemote(const char *line, Int_t *)
{
   if (!line) return 0;

   if (!strncmp(line, "-?", 2) || !strncmp(line, "-h", 2) ||
       !strncmp(line, "--help", 6)) {
      Info("ProcessRemote", "remote session help:");
      Printf(".R [user@]host[:dir] [-l user] [-d dbg] [[<]script] | [host] -close");
      Printf("Create a ROOT session on the specified remote host.");
      Printf("The variable \"dir\" is the remote directory to be used as working dir.");
      Printf("The username can be specified in two ways, \"-l\" having the priority");
      Printf("(as in ssh). A \"dbg\" value > 0 gives increasing verbosity.");
      Printf("The last argument \"script\" allows to specify an alternative script to");
      Printf("be executed remotely to startup the session, \"roots\" being");
      Printf("the default. If the script is preceded by a \"<\" the script will be");
      Printf("sourced, after which \"roots\" is executed. The sourced script can be ");
      Printf("used to change the PATH and other variables, allowing an alternative");
      Printf("\"roots\" script to be found.");
      Printf("To close down a session do \".R host -close\".");
      Printf("To switch between sessions do \".R host\", to switch to the local");
      Printf("session do \".R\".");
      Printf("To list all open sessions do \"gApplication->GetApplications()->Print()\".");
      return 0;
   }

   TString hostdir, user, script;
   Int_t dbg = 0;
   Int_t rc = ParseRemoteLine(line, hostdir, user, dbg, script);
   if (hostdir.Length() <= 0) {
      // Close the remote application if required
      if (rc == 1) {
         TApplication::Close(fAppRemote);
         delete fAppRemote;
      }
      // Return to local run
      fAppRemote = 0;
      // Done
      return 1;
   } else if (rc == 1) {
      // close an existing remote application
      TApplication *ap = TApplication::Open(hostdir, 0, 0);
      if (ap) {
         TApplication::Close(ap);
         delete ap;
      }
   }
   // Attach or start a remote application
   if (user.Length() > 0)
      hostdir.Insert(0,Form("%s@", user.Data()));
   const char *sc = (script.Length() > 0) ? script.Data() : 0;
   TApplication *ap = TApplication::Open(hostdir, dbg, sc);
   if (ap) {
      fAppRemote = ap;
   }

   // Done
   return 1;
}

namespace {
   static int PrintFile(const char* filename) {
      TString sFileName(filename);
      gSystem->ExpandPathName(sFileName);
      if (gSystem->AccessPathName(sFileName)) {
         Error("ProcessLine()", "Cannot find file %s", filename);
         return 1;
      }
      std::ifstream instr(sFileName);
      TString content;
      content.ReadFile(instr);
      Printf("%s", content.Data());
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Process a single command line, either a C++ statement or an interpreter
/// command starting with a ".".
/// Return the return value of the command cast to a long.

Long_t TApplication::ProcessLine(const char *line, Bool_t sync, Int_t *err)
{
   if (!line || !*line) return 0;

   // If we are asked to go remote do it
   if (!strncmp(line, ".R", 2)) {
      Int_t n = 2;
      while (*(line+n) == ' ')
         n++;
      return ProcessRemote(line+n, err);
   }

   // Redirect, if requested
   if (fAppRemote && TestBit(kProcessRemotely)) {
      ResetBit(kProcessRemotely);
      return fAppRemote->ProcessLine(line, err);
   }

   if (!strncasecmp(line, ".qqqqqqq", 7)) {
      gSystem->Abort();
   } else if (!strncasecmp(line, ".qqqqq", 5)) {
      Info("ProcessLine", "Bye... (try '.qqqqqqq' if still running)");
      gSystem->Exit(1);
   } else if (!strncasecmp(line, ".exit", 4) || !strncasecmp(line, ".quit", 2)) {
      Terminate(0);
      return 0;
   }

   if (!strncmp(line, ".?", 2) || !strncmp(line, ".help", 5)) {
      Help(line);
      return 1;
   }

   if (!strncmp(line, ".demo", 5)) {
      if (gROOT->IsBatch()) {
         Error("ProcessLine", "Cannot show demos in batch mode!");
         return 1;
      }
      ProcessLine(".x " + TROOT::GetTutorialDir() + "/demos.C");
      return 0;
   }

   if (!strncmp(line, ".license", 8)) {
      return PrintFile(TROOT::GetDocDir() + "/LICENSE");
   }

   if (!strncmp(line, ".credits", 8)) {
      TString credits = TROOT::GetDocDir() + "/CREDITS";
      if (gSystem->AccessPathName(credits, kReadPermission))
         credits = TROOT::GetDocDir() + "/README/CREDITS";
      return PrintFile(credits);
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
      char *s = strtok(fn, "+("); // this method does not need to be reentrant
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
                 line[1]);
      }
      Long_t retval = 0;
      if (!mac)
         Error("ProcessLine", "macro %s not found in path %s", fname.Data(),
               TROOT::GetMacroPath());
      else {
         TString cmd(line+1);
         Ssiz_t posSpace = cmd.Index(' ');
         if (posSpace == -1) cmd.Remove(1);
         else cmd.Remove(posSpace);
         TString tempbuf;
         if (sync) {
            tempbuf.Form(".%s %s%s%s", cmd.Data(), mac, aclicMode.Data(),io.Data());
            retval = gInterpreter->ProcessLineSynch(tempbuf,
                                                   (TInterpreter::EErrorCode*)err);
         } else {
            tempbuf.Form(".%s %s%s%s", cmd.Data(), mac, aclicMode.Data(),io.Data());
            retval = gInterpreter->ProcessLine(tempbuf,
                                              (TInterpreter::EErrorCode*)err);
         }
      }

      delete [] mac;

      InitializeGraphics();

      return retval;
   }

   if (!strncmp(line, ".X", 2) || !strncmp(line, ".x", 2)) {
      return ProcessFile(line+3, err, line[2] == 'k');
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

////////////////////////////////////////////////////////////////////////////////
/// Process a file containing a C++ macro.

Long_t TApplication::ProcessFile(const char *file, Int_t *error, Bool_t keep)
{
   return ExecuteFile(file, error, keep);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a file containing a C++ macro (static method). Can be used
/// while TApplication is not yet created.

Long_t TApplication::ExecuteFile(const char *file, Int_t *error, Bool_t keep)
{
   static const Int_t kBufSize = 1024;

   if (!file || !*file) return 0;

   TString aclicMode;
   TString arguments;
   TString io;
   TString fname = gSystem->SplitAclicMode(file, aclicMode, arguments, io);

   char *exnam = gSystem->Which(TROOT::GetMacroPath(), fname, kReadPermission);
   if (!exnam) {
      ::Error("TApplication::ExecuteFile", "macro %s not found in path %s", fname.Data(),
              TROOT::GetMacroPath());
      delete [] exnam;
      if (error)
         *error = (Int_t)TInterpreter::kRecoverable;
      return 0;
   }

   ::std::ifstream macro(exnam, std::ios::in);
   if (!macro.good()) {
      ::Error("TApplication::ExecuteFile", "%s no such file", exnam);
      if (error)
         *error = (Int_t)TInterpreter::kRecoverable;
      delete [] exnam;
      return 0;
   }

   char currentline[kBufSize];
   char dummyline[kBufSize];
   int tempfile = 0;
   int comment  = 0;
   int ifndefc  = 0;
   int ifdef    = 0;
   char *s      = 0;
   Bool_t execute = kFALSE;
   Long_t retval = 0;

   while (1) {
      bool res = (bool)macro.getline(currentline, kBufSize);
      if (macro.eof()) break;
      if (!res) {
         // Probably only read kBufSize, let's ignore the remainder of
         // the line.
         macro.clear();
         while (!macro.getline(dummyline, kBufSize) && !macro.eof()) {
            macro.clear();
         }
      }
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

      if (!comment && (!strncmp(s, ".X", 2) || !strncmp(s, ".x", 2))) {
         retval = ExecuteFile(s+3);
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
   macro.close();

   if (!execute) {
      TString exname = exnam;
      if (!tempfile) {
         // We have a script that does NOT contain an unnamed macro,
         // so we can call the script compiler on it.
         exname += aclicMode;
      }
      exname += arguments;
      exname += io;

      TString tempbuf;
      if (tempfile) {
         tempbuf.Form(".x %s", exname.Data());
      } else {
         tempbuf.Form(".X%s %s", keep ? "k" : " ", exname.Data());
      }
      retval = gInterpreter->ProcessLineSynch(tempbuf,(TInterpreter::EErrorCode*)error);
   }

   delete [] exnam;
   return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// Main application eventloop. Calls system dependent eventloop via gSystem.

void TApplication::Run(Bool_t retrn)
{
   SetReturnFromRun(retrn);

   fIsRunning = kTRUE;

   gSystem->Run();
   fIsRunning = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the command to be executed after the system has been idle for
/// idleTimeInSec seconds. Normally called via TROOT::Idle(...).

void TApplication::SetIdleTimer(UInt_t idleTimeInSec, const char *command)
{
   if (fIdleTimer) RemoveIdleTimer();
   fIdleCommand = command;
   fIdleTimer = new TIdleTimer(idleTimeInSec*1000);
   gSystem->AddTimer(fIdleTimer);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove idle timer. Normally called via TROOT::Idle(0).

void TApplication::RemoveIdleTimer()
{
   if (fIdleTimer) {
      // timers are removed from the gSystem timer list by their dtor
      SafeDelete(fIdleTimer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when system starts idleing.

void TApplication::StartIdleing()
{
   if (fIdleTimer) {
      fIdleTimer->Reset();
      gSystem->AddTimer(fIdleTimer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called when system stops idleing.

void TApplication::StopIdleing()
{
   if (fIdleTimer)
      gSystem->RemoveTimer(fIdleTimer);
}

////////////////////////////////////////////////////////////////////////////////
/// What to do when tab is pressed. Re-implemented by TRint.
/// See TTabCom::Hook() for meaning of return values.

Int_t TApplication::TabCompletionHook(char* /*buf*/, int* /*pLoc*/, std::ostream& /*out*/)
{
   return -1;
}


////////////////////////////////////////////////////////////////////////////////
/// Terminate the application by call TSystem::Exit() unless application has
/// been told to return from Run(), by a call to SetReturnFromRun().

void TApplication::Terminate(Int_t status)
{
   Emit("Terminate(Int_t)", status);

   if (fReturnFromRun)
      gSystem->ExitLoop();
   else {
      //close TMemStat
      if (fUseMemstat) {
         ProcessLine("TMemStat::Close()");
         fUseMemstat = kFALSE;
      }

      gSystem->Exit(status);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Emit signal when a line has been processed.

void TApplication::LineProcessed(const char *line)
{
   Emit("LineProcessed(const char*)", line);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit signal when console keyboard key was pressed.

void TApplication::KeyPressed(Int_t key)
{
   Emit("KeyPressed(Int_t)", key);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit signal when return key was pressed.

void TApplication::ReturnPressed(char *text )
{
   Emit("ReturnPressed(char*)", text);
}

////////////////////////////////////////////////////////////////////////////////
/// Set console echo mode:
///
///  - mode = kTRUE  - echo input symbols
///  - mode = kFALSE - noecho input symbols

void TApplication::SetEchoMode(Bool_t)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Static function used to create a default application environment.

void TApplication::CreateApplication()
{
   R__LOCKGUARD(gROOTMutex);
   // gApplication is set at the end of 'new TApplication.
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
      gApplication->SetBit(kDefaultApplication);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Static function used to attach to an existing remote application
/// or to start one.

TApplication *TApplication::Open(const char *url,
                                  Int_t debug, const char *script)
{
   TApplication *ap = 0;
   TUrl nu(url);
   Int_t nnew = 0;

   // Look among the existing ones
   if (fgApplications) {
      TIter nxa(fgApplications);
      while ((ap = (TApplication *) nxa())) {
         TString apn(ap->ApplicationName());
         if (apn == url) {
            // Found matching application
            return ap;
         } else {
            // Check if same machine and user
            TUrl au(apn);
            if (strlen(au.GetUser()) > 0 && strlen(nu.GetUser()) > 0 &&
                !strcmp(au.GetUser(), nu.GetUser())) {
               if (!strncmp(au.GetHost(), nu.GetHost(), strlen(nu.GetHost())))
                  // New session on a known machine
                  nnew++;
            }
         }
      }
   } else {
      ::Error("TApplication::Open", "list of applications undefined - protocol error");
      return ap;
   }

   // If new session on a known machine pass the number as option
   if (nnew > 0) {
      nnew++;
      nu.SetOptions(Form("%d", nnew));
   }

   // Instantiate the TApplication object to be run
   TPluginHandler *h = 0;
   if ((h = gROOT->GetPluginManager()->FindHandler("TApplication","remote"))) {
      if (h->LoadPlugin() == 0) {
         ap = (TApplication *) h->ExecPlugin(3, nu.GetUrl(), debug, script);
      } else {
         ::Error("TApplication::Open", "failed to load plugin for TApplicationRemote");
      }
   } else {
      ::Error("TApplication::Open", "failed to find plugin for TApplicationRemote");
   }

   // Add to the list
   if (ap && !(ap->TestBit(kInvalidObject))) {
      fgApplications->Add(ap);
      gROOT->GetListOfBrowsables()->Add(ap, ap->ApplicationName());
      TIter next(gROOT->GetListOfBrowsers());
      TBrowser *b;
      while ((b = (TBrowser*) next()))
         b->Add(ap, ap->ApplicationName());
      gROOT->RefreshBrowsers();
   } else {
      SafeDelete(ap);
      ::Error("TApplication::Open",
              "TApplicationRemote for %s could not be instantiated", url);
   }

   // Done
   return ap;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function used to close a remote application

void TApplication::Close(TApplication *app)
{
   if (app) {
      app->Terminate(0);
      fgApplications->Remove(app);
      gROOT->GetListOfBrowsables()->RecursiveRemove(app);
      TIter next(gROOT->GetListOfBrowsers());
      TBrowser *b;
      while ((b = (TBrowser*) next()))
         b->RecursiveRemove(app);
      gROOT->RefreshBrowsers();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Show available sessions

void TApplication::ls(Option_t *opt) const
{
   if (fgApplications) {
      TIter nxa(fgApplications);
      TApplication *a = 0;
      while ((a = (TApplication *) nxa())) {
         a->Print(opt);
      }
   } else {
      Print(opt);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Static method returning the list of available applications

TList *TApplication::GetApplications()
{
   return fgApplications;
}
