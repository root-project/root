// @(#)root/rint:$Name:  $:$Id: TRint.cxx,v 1.14 2001/10/22 14:55:47 rdm Exp $
// Author: Rene Brun   17/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rint                                                                 //
//                                                                      //
// Rint is the ROOT Interactive Interface. It allows interactive access //
// to the ROOT system via the CINT C/C++ interpreter.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TClass.h"
#include "TVirtualX.h"
#include "Getline.h"
#include "TStyle.h"
#include "TObjectTable.h"
#include "TClassTable.h"
#include "TStopwatch.h"
#include "TBenchmark.h"
#include "TRint.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TSysEvtHandler.h"
#include "TError.h"
#include "TException.h"
#include "TInterpreter.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TFile.h"
#include "TMapFile.h"
#include "TTabCom.h"

#ifdef R__UNIX
#include <signal.h>

extern "C" {
   extern int G__get_security_error();
   extern int G__genericerror(const char* msg);
}
#endif



//----- Interrupt signal handler -----------------------------------------------
//______________________________________________________________________________
class TInterruptHandler : public TSignalHandler {
public:
   TInterruptHandler() : TSignalHandler(kSigInterrupt, kFALSE) { }
   Bool_t  Notify();
};

//______________________________________________________________________________
Bool_t TInterruptHandler::Notify()
{
   // TRint interrupt handler.

   if (fDelay) {
      fDelay++;
      return kTRUE;
   }

   // make sure we use the sbrk heap (in case of mapped files)
   gMmallocDesc = 0;

   if (!G__get_security_error())
      G__genericerror("\n *** Break *** keyboard interrupt");
   else {
      Printf("\n *** Break *** keyboard interrupt");
      if (TROOT::Initialized()) {
         Getlinem(kInit, "Root > ");
         gInterpreter->RewindDictionary();
         Throw(GetSignal());
      }
   }
   return kTRUE;
}

//----- Terminal Input file handler --------------------------------------------
//______________________________________________________________________________
class TTermInputHandler : public TFileHandler {
public:
   TTermInputHandler(int fd) : TFileHandler(fd, 1) { }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TTermInputHandler::Notify()
{
   return gApplication->HandleTermInput();
}


ClassImp(TRint)

//______________________________________________________________________________
TRint::TRint(const char *appClassName, int *argc, char **argv, void *options,
             int numOptions, Bool_t noLogo)
       : TApplication(appClassName, argc, argv, options, numOptions)
{
   // Create an application environment. The TRint environment provides an
   // interface to the WM manager functionality and eventloop via inheritance
   // of TApplication and in addition provides interactive access to
   // the CINT C++ interpreter via the command line.

   fNcmd          = 0;
   fDefaultPrompt = "root [%d] ";
   fInterrupt     = kFALSE;

   gBenchmark = new TBenchmark();

   if (!noLogo && !NoLogoOpt())
      PrintLogo();

   // Everybody expects iostream to be available, so load it...
#ifndef WIN32
   ProcessLine("#include <iostream>");
#endif
   // Allow the usage of ClassDef and ClassImp in interpreted macros
   ProcessLine("#include <RtypesCint.h>");

   // The following libs are also useful to have,
   // make sure they are loaded...
   gROOT->LoadClass("TGeometry",   "Graf3d");
   gROOT->LoadClass("TTree",       "Tree");
   gROOT->LoadClass("TMatrix",     "Matrix");
   gROOT->LoadClass("TMinuit",     "Minuit");
   gROOT->LoadClass("TPostScript", "Postscript");
   gROOT->LoadClass("TCanvas",     "Gpad");
   gROOT->LoadClass("THtml",       "Html");

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Rint.Load", (char*)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s",logon));
      delete [] mac;
   }

   // Execute logon macro
   logon = gEnv->GetValue("Rint.Logon", (char*)0);
   if (logon && !NoLogOpt()) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessFile(logon);
      delete [] mac;
   }

   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and terminal input handlers
   TInterruptHandler *ih = new TInterruptHandler();
   ih->Add();
   SetSignalHandler(ih);

   TTermInputHandler *th = new TTermInputHandler(0);
   th->Add();

   // Goto into raw terminal input mode
   char defhist[128];
#ifndef R__VMS
   sprintf(defhist, "%s/.root_hist", gSystem->Getenv("HOME"));
#else
   sprintf(defhist, "%s.root_hist", gSystem->Getenv("HOME"));
#endif
   logon = gEnv->GetValue("Rint.History", defhist);
   Gl_histinit((char *)logon);
   Gl_windowchanged();

   // Setup for tab completion
   gTabCom = new TTabCom;
}

//______________________________________________________________________________
TRint::~TRint()
{

}

//______________________________________________________________________________
void TRint::Run(Bool_t retrn)
{
   // Main application eventloop. First process files given on the command
   // line and then go into the main application event loop, unless the -q
   // command line option was specfied in which case the program terminates.
   // When retrun is true this method returns even when -q was specified.

   Getlinem(kInit, GetPrompt());

   // Process shell command line input files
   if (InputFiles()) {
      TObjString *file;
      TIter next(InputFiles());
      RETRY {
         while ((file = (TObjString *)next())) {
            char cmd[256];
            if (file->String().EndsWith(".root")) {
               const char *rfile = (const char*)file->String();
               Printf("\nAttaching file %s...", rfile);
               char *base = StrDup(gSystem->BaseName(rfile));
               char *s = strchr(base, '.'); *s = 0;
               sprintf(cmd, "TFile *%s = TFile::Open(\"%s\")", base, rfile);
               delete [] base;
            } else {
               Printf("\nProcessing %s...", (const char*)file->String());
               sprintf(cmd, ".x %s", (const char*)file->String());
            }
            Getlinem(kCleanUp, 0);
            Gl_histadd(cmd);
            fNcmd++;
            ProcessLine(cmd);
         }
      } ENDTRY;

      if (QuitOpt()) {
         if (retrn) return;
         Terminate(0);
      }

      ClearInputFiles();

      Getlinem(kInit, GetPrompt());
   }

   if (QuitOpt()) {
      printf("\n");
      if (retrn) return;
      Terminate(0);
   }

   TApplication::Run(retrn);

   Getlinem(kCleanUp, 0);
}

//______________________________________________________________________________
void TRint::PrintLogo()
{
   // Print the ROOT logo on standard output.

   Int_t iday,imonth,iyear;
   static const char *months[] = {"January","February","March","April","May",
                                  "June","July","August","September","October",
                                  "November","December"};
   const char *root_version = gROOT->GetVersion();
   Int_t idatqq = gROOT->GetVersionDate();
   iday   = idatqq%100;
   imonth = (idatqq/100)%100;
   iyear  = (idatqq/10000);
   char *root_date = Form("%d %s %4d",iday,months[imonth-1],iyear);

   Printf("  *******************************************");
   Printf("  *                                         *");
   Printf("  *        W E L C O M E  to  R O O T       *");
   Printf("  *                                         *");
   Printf("  *   Version%10s %17s   *",root_version,root_date);
// Printf("  *            Development version          *");
   Printf("  *                                         *");
   Printf("  *  You are welcome to visit our Web site  *");
   Printf("  *          http://root.cern.ch            *");
   Printf("  *                                         *");
   Printf("  *******************************************");

#ifdef R__UNIX
   if (!strcmp(gVirtualX->GetName(), "X11TTF"))
      Printf("\nFreeType Engine v1.x used to render TrueType fonts.");
#endif
#ifdef _REENTRANT
#ifdef R__UNIX
   else
#endif
      printf("\n");
   Printf("Compiled for %s with thread support.", gSystem->GetBuildArch());
#else
#ifdef R__UNIX
   else
#endif
      printf("\n");
   Printf("Compiled for %s.", gSystem->GetBuildArch());
#endif

   gInterpreter->PrintIntro();

#ifdef R__UNIX
   // Popdown X logo, only if started with -splash option
   for (int i = 0; i < Argc(); i++)
      if (!strcmp(Argv(i), "-splash"))
         kill(getppid(), SIGUSR1);
#endif
}

//______________________________________________________________________________
char *TRint::GetPrompt()
{
   // Get prompt from interpreter. Either "root [n]" or "end with '}'".

   char *s = gInterpreter->GetPrompt();
   if (s[0])
      strcpy(fPrompt, s);
   else
      sprintf(fPrompt, fDefaultPrompt.Data(), fNcmd);

   return fPrompt;
}

//______________________________________________________________________________
const char *TRint::SetPrompt(const char *newPrompt)
{
   // Set a new default prompt. It returns the previous prompt.
   // The prompt may contain a %d which will be replaced by the commend
   // number. The default prompt is "root [%d] ". The maximum length of
   // the prompt is 55 characters. To set the prompt in an interactive
   // session do:
   // root [0] ((TRint*)gROOT->GetApplication())->SetPrompt("aap> ")
   // aap>

   static TString op = fDefaultPrompt;

   if (newPrompt && strlen(newPrompt) <= 55)
      fDefaultPrompt = newPrompt;
   else
      Error("SetPrompt", "newPrompt too long (> 55 characters)");

   return op.Data();
}

//______________________________________________________________________________
Bool_t TRint::HandleTermInput()
{
   // Handle input coming from terminal.

   static TStopwatch timer;
   char *line;

   if ((line = Getlinem(kOneChar, 0))) {
      if (line[0] == 0 && Gl_eof())
         Terminate(0);

      if (gROOT->Timer()) timer.Start();

      Gl_histadd(line);

      char *s = line;
      while (s && *s == ' ') s++;     // strip-off leading blanks
      s[strlen(s)-1] = '\0';          // strip also '\n' off
      fInterrupt = kFALSE;

      if (!gInterpreter->GetMore() && strlen(s) != 0) fNcmd++;

      ProcessLine(s);

      if (strstr(s,".reset") != s)
          gInterpreter->EndOfLineAction();

      if (gROOT->Timer()) timer.Print();

      gTabCom->ClearAll();
      Getlinem(kInit, GetPrompt());
   }
   return kTRUE;
}

//______________________________________________________________________________
void TRint::Terminate(int status)
{
   // Terminate the application. Reset the terminal to sane mode and call
   // the logoff macro defined via Rint.Logoff environment variable.

   Getlinem(kCleanUp, 0);

   if (ReturnFromRun()) {
      gSystem->ExitLoop();
   } else {
      //Execute logoff macro
      const char *logoff;
      logoff = gEnv->GetValue("Rint.Logoff", (char*)0);
      if (logoff && !NoLogOpt()) {
         char *mac = gSystem->Which(TROOT::GetMacroPath(), logoff, kReadPermission);
         if (mac)
            ProcessFile(logoff);
         delete [] mac;
      }

      gSystem->Exit(status);
   }
}
