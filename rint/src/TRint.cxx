// @(#)root/rint:$Name:  $:$Id: TRint.cxx,v 1.53 2006/03/28 23:59:13 rdm Exp $
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
#include "TError.h"

#ifdef R__UNIX
#include <signal.h>

extern "C" {
   extern int G__get_security_error();
   extern int G__genericerror(const char* msg);
}
#endif

//______________________________________________________________________________
static Int_t Key_Pressed(Int_t key)
{
   gApplication->KeyPressed(key);
   return 0;
}

//______________________________________________________________________________
static Int_t BeepHook()
{
   if (!gSystem) return 0;
   gSystem->Beep();
   return 1;
}


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
      Break("TInterruptHandler::Notify", "keyboard interrupt");
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
   TTermInputHandler(Int_t fd) : TFileHandler(fd, 1) { }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TTermInputHandler::Notify()
{
   // Notify implementation.  Call the application interupt handler.

   return gApplication->HandleTermInput();
}


ClassImp(TRint)

//______________________________________________________________________________
TRint::TRint(const char *appClassName, Int_t *argc, char **argv, void *options,
             Int_t numOptions, Bool_t noLogo)
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
   ProcessLine("#include <iostream>", kTRUE);
   ProcessLine("#include <_string>", kTRUE); // for std::string iostream.
   ProcessLine("#include <vector>", kTRUE);  // Needed because std::vector and std::pair are
   ProcessLine("#include <pair>", kTRUE);    //   used within the core ROOT dictionaries
                                             //   and CINT will not be able to properly unload these files

   // Allow the usage of ClassDef and ClassImp in interpreted macros
   ProcessLine("#include <RtypesCint.h>", kTRUE);

   // Disallow the interpretation of Rtypes.h, TError.h and TGenericClassInfo.h
   ProcessLine("#define ROOT_Rtypes 0", kTRUE);
   ProcessLine("#define ROOT_TError 0", kTRUE);
   ProcessLine("#define ROOT_TGenericClassInfo 0", kTRUE);

   // The following libs are also useful to have, make sure they are loaded...
   gROOT->LoadClass("TMinuit",     "Minuit");
   gROOT->LoadClass("TPostScript", "Postscript");
   gROOT->LoadClass("THtml",       "Html");

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Rint.Load", (char*)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s",logon),kTRUE);
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

   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Install interrupt and terminal input handlers
   TInterruptHandler *ih = new TInterruptHandler();
   ih->Add();
   SetSignalHandler(ih);

   // Handle stdin events
   fInputHandler = new TTermInputHandler(0);
   fInputHandler->Add();

   // Goto into raw terminal input mode
   char defhist[kMAXPATHLEN];
#ifndef R__VMS
   sprintf(defhist, "%s/.root_hist", gSystem->HomeDirectory());
#else
   sprintf(defhist, "%s.root_hist", gSystem->HomeDirectory());
#endif
   logon = gEnv->GetValue("Rint.History", defhist);
   Gl_histinit((char *)logon);
   Gl_windowchanged();

   // Setup for tab completion
   gTabCom      = new TTabCom;
   Gl_in_key    = &Key_Pressed;
   Gl_beep_hook = &BeepHook;
}

//______________________________________________________________________________
TRint::~TRint()
{
   // Destructor.
}

//______________________________________________________________________________
void TRint::Run(Bool_t retrn)
{
   // Main application eventloop. First process files given on the command
   // line and then go into the main application event loop, unless the -q
   // command line option was specfied in which case the program terminates.
   // When retrun is true this method returns even when -q was specified.
   //
   // When QuitOpt is true and retrn is false, terminate the application with
   // an error code equal to either the ProcessLine error (if any) or the
   // return value of the command casted to a long.

   Getlinem(kInit, GetPrompt());

   Long_t retval = 0;
   Int_t  error = 0;

   // Process shell command line input files
   if (InputFiles()) {
      // Make sure that calls into the event loop
      // ignore end-of-file on the terminal.
      fInputHandler->DeActivate();
      Bool_t needGetlinemInit = kFALSE;
      TIter next(InputFiles());
      RETRY {
         retval = 0; error = 0;
         Int_t nfile = 0;
         TObjString *file;
         while ((file = (TObjString *)next())) {
            char cmd[256];
            if (!fNcmd)
               printf("\n");
            if (file->String().EndsWith(".root")) {
               file->String().ReplaceAll("\\","/");
               const char *rfile = (const char*)file->String();
               Printf("Attaching file %s as _file%d...", rfile, nfile);
               sprintf(cmd, "TFile *_file%d = TFile::Open(\"%s\")", nfile++, rfile);
            } else {
               Printf("Processing %s...", (const char*)file->String());
               sprintf(cmd, ".x %s", (const char*)file->String());
            }
            Getlinem(kCleanUp, 0);
            Gl_histadd(cmd);
            fNcmd++;

            // The ProcessLine might throw an 'exception'.  In this case,
            // GetLinem(kInit,"Root >") is called and we are jump back
            // to RETRY ... and we have to avoid the Getlinem(kInit, GetPrompt());
            needGetlinemInit = kFALSE;
            retval = ProcessLine(cmd, kFALSE, &error);
            gInterpreter->EndOfLineAction();

            // The ProcessLine has successfully completed and we need
            // to call Getlinem(kInit, GetPrompt());
            needGetlinemInit = kTRUE;

            if (error != 0) break;
         }
      } ENDTRY;

      // Allow end-of-file on the terminal to be noticed
      // after we finish processing the command line input files.
      fInputHandler->Activate();

      if (QuitOpt()) {
         if (retrn) return;
         Terminate(error == 0 ? retval : error);
      }

      ClearInputFiles();

      if (needGetlinemInit) Getlinem(kInit, GetPrompt());
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
   char *version_date = Form("%d %s %4d",iday,months[imonth-1],iyear);
   idatqq = gROOT->GetBuiltDate();
   iday   = idatqq%100;
   imonth = (idatqq/100)%100;
   iyear  = (idatqq/10000);
   char *built_date = Form("%d %s %4d",iday,months[imonth-1],iyear);


   Printf("  *******************************************");
   Printf("  *                                         *");
   Printf("  *        W E L C O M E  to  R O O T       *");
   Printf("  *                                         *");
   Printf("  *   Version%10s %17s   *", root_version, version_date);
// Printf("  *            Development version          *");
   Printf("  *                                         *");
   Printf("  *  You are welcome to visit our Web site  *");
   Printf("  *          http://root.cern.ch            *");
   Printf("  *                                         *");
   Printf("  *******************************************");

   if (strstr(gVirtualX->GetName(), "TTF")) {
      Int_t major, minor, patch;
      //TTF::Version(major, minor, patch);
      // avoid dependency on libGraf and hard code, will not change too often
      major = 2; minor = 1; patch = 9;
      Printf("\nFreeType Engine v%d.%d.%d used to render TrueType fonts.",
             major, minor, patch);
   }
#ifdef _REENTRANT
   else
      printf("\n");
   Printf("Compiled on %s for %s with thread support.", built_date,
          gSystem->GetBuildArch());
#else
   else
      printf("\n");
   Printf("Compiled on %s for %s.", built_date, gSystem->GetBuildArch());
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

      gVirtualX->SetKeyAutoRepeat(kTRUE);

      Gl_histadd(line);

      TString sline = line;
      line[0] = 0;

      // strip off '\n' and leading and trailing blanks
      sline = sline.Chop();
      sline = sline.Strip(TString::kBoth);
      ReturnPressed((char*)sline.Data());

      fInterrupt = kFALSE;

      if (!gInterpreter->GetMore() && !sline.IsNull()) fNcmd++;

      // prevent recursive calling of this input handler
      fInputHandler->DeActivate();

      if (gROOT->Timer()) timer.Start();

      Bool_t added = kFALSE;
#ifdef R__EH
      try {
#endif
         TRY {
            ProcessLine(sline);
         } CATCH(excode) {
            // enable again input handler
            fInputHandler->Activate();
            added = kTRUE;
            Throw(excode);
         } ENDTRY;
#ifdef R__EH
      }
      // handle every exception
      catch (...) {
         // enable again intput handler
         if (!added) fInputHandler->Activate();
         throw;
      }
#endif

      if (gROOT->Timer()) timer.Print("u");

      // enable again intput handler
      fInputHandler->Activate();

      if (!sline.BeginsWith(".reset"))
         gInterpreter->EndOfLineAction();

      gTabCom->ClearAll();
      Getlinem(kInit, GetPrompt());
   }
   return kTRUE;
}

//______________________________________________________________________________
void TRint::Terminate(Int_t status)
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

//______________________________________________________________________________
void TRint::SetEchoMode(Bool_t mode)
{
   // Set console mode:
   //
   //  mode = kTRUE  - echo input symbols
   //  mode = kFALSE - noecho input symbols

   Gl_config("noecho", mode ? 0 : 1);
}
