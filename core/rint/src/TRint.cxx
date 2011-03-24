// @(#)root/rint:$Id$
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
#include "TStyle.h"
#include "TObjectTable.h"
#include "TClassTable.h"
#include "TStopwatch.h"
#include "TBenchmark.h"
#include "TRint.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TSysEvtHandler.h"
#include "TSystemDirectory.h"
#include "TError.h"
#include "TException.h"
#include "TInterpreter.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TTabCom.h"
#include "TError.h"
#include <stdlib.h>

#ifdef R__BUILDEDITLINE
#include "Getline_el.h"
#else
#include "Getline.h"
#endif

#ifdef R__UNIX
#include <signal.h>
#endif

R__EXTERN void *gMmallocDesc; //is used and set in TMapFile and TClass

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

//______________________________________________________________________________
static void ResetTermAtExit()
{
   // Restore terminal to non-raw mode.

   Getlinem(kCleanUp, 0);
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

   if (!gCint->GetSecurityError())
      gCint->GenericError("\n *** Break *** keyboard interrupt");
   else {
      Break("TInterruptHandler::Notify", "keyboard interrupt");
      if (TROOT::Initialized()) {
         Getlinem(kInit, "Root > ");
         gCint->RewindDictionary();
#ifndef WIN32
         Throw(GetSignal());
#endif
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

   if (!noLogo && !NoLogoOpt()) {
      Bool_t lite = (Bool_t) gEnv->GetValue("Rint.WelcomeLite", 0);
      PrintLogo(lite);
   }

   // Load some frequently used includes
   Int_t includes = gEnv->GetValue("Rint.Includes", 1);
   // When the interactive ROOT starts, it can automatically load some frequently
   // used includes. However, this introduces several overheads
   //   -A long list of cint and system files must be kept open during the session
   //   -The initialisation takes more time (noticeable when using gdb or valgrind)
   //   -Memory overhead of about 5 Mbytes (1/3 of the ROOT executable) when including <vector>
   // In $ROOTSYS/etc/system.rootrc, you can set the variable Rint.Includes to 0
   //  to disable the loading of these includes at startup.
   // You can set the variable to 1 (default) to load only <iostream>, <string> and <RTypesCint.h>
   // You can set it to 2 to load in addition <vector> and <pair>
   // We strongly recommend setting the variable to 2 if your scripts include <vector>
   // and you execute your scripts multiple times.
   if (includes > 0) {
      ProcessLine("#include <iostream>", kTRUE);
      ProcessLine("#include <string>", kTRUE); // for std::string iostream.
      ProcessLine("#include <DllImport.h>", kTRUE);// Defined R__EXTERN
      if (includes > 1) {
         ProcessLine("#include <vector>", kTRUE);  // Needed because std::vector and std::pair are
         ProcessLine("#include <pair>", kTRUE);    // used within the core ROOT dictionaries
                                                   // and CINT will not be able to properly unload these files
      }
   }

   // Load user functions
   const char *logon;
   logon = gEnv->GetValue("Rint.Load", (char*)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessLine(Form(".L %s",logon), kTRUE);
      delete [] mac;
   }

   // Execute logon macro
   ExecLogon();

   // Save current interpreter context
   gCint->SaveContext();
   gCint->SaveGlobalsContext();

   // Install interrupt and terminal input handlers
   TInterruptHandler *ih = new TInterruptHandler();
   ih->Add();
   SetSignalHandler(ih);

   // Handle stdin events
   fInputHandler = new TTermInputHandler(0);
   fInputHandler->Add();

   // Goto into raw terminal input mode
   char defhist[kMAXPATHLEN];
   snprintf(defhist, sizeof(defhist), "%s/.root_hist", gSystem->HomeDirectory());
   logon = gEnv->GetValue("Rint.History", defhist);
   // In the code we had HistorySize and HistorySave, in the rootrc and doc
   // we have HistSize and HistSave. Keep the doc as it is and check
   // now also for HistSize and HistSave in case the user did not use
   // the History versions
   int hist_size = gEnv->GetValue("Rint.HistorySize", 500);
   if (hist_size == 500)
      hist_size = gEnv->GetValue("Rint.HistSize", 500);
   int hist_save = gEnv->GetValue("Rint.HistorySave", 400);
   if (hist_save == 400)
      hist_save = gEnv->GetValue("Rint.HistSave", 400);
   const char *envHist = gSystem->Getenv("ROOT_HIST");
   if (envHist) {
      hist_size = atoi(envHist);
      envHist = strchr(envHist, ':');
      if (envHist)
         hist_save = atoi(envHist+1);
   }
   Gl_histsize(hist_size, hist_save);
   Gl_histinit((char *)logon);

#ifdef R__BUILDEDITLINE
   // black on white or white on black?
   static const char* defaultColorsBW[] = {
      "bold blue", "magenta", "bold green", "bold red underlined", "default"
   };
   static const char* defaultColorsWB[] = {
      "yellow", "magenta", "bold green", "bold red underlined", "default"
   };

   const char** defaultColors = defaultColorsBW;
   TString revColor = gEnv->GetValue("Rint.ReverseColor", "no");
   if (revColor.Contains("yes", TString::kIgnoreCase)) {
      defaultColors = defaultColorsWB;
   }
   TString colorType = gEnv->GetValue("Rint.TypeColor", defaultColors[0]);
   TString colorTabCom = gEnv->GetValue("Rint.TabComColor", defaultColors[1]);
   TString colorBracket = gEnv->GetValue("Rint.BracketColor", defaultColors[2]);
   TString colorBadBracket = gEnv->GetValue("Rint.BadBracketColor", defaultColors[3]);
   TString colorPrompt = gEnv->GetValue("Rint.PromptColor", defaultColors[4]);
   Gl_setColors(colorType, colorTabCom, colorBracket, colorBadBracket, colorPrompt);
#endif

   Gl_windowchanged();

   atexit(ResetTermAtExit);

   // Setup for tab completion
   gTabCom      = new TTabCom;
   Gl_in_key    = &Key_Pressed;
   Gl_beep_hook = &BeepHook;

   // tell CINT to use our getline
   gCint->SetGetline(Getline, Gl_histadd);
}

//______________________________________________________________________________
TRint::~TRint()
{
   // Destructor.

   delete gTabCom;
   gTabCom = 0;
   Gl_in_key = 0;
   Gl_beep_hook = 0;
   fInputHandler->Remove();
   delete fInputHandler;
   // We can't know where the signal handler was changed since we started ...
   // so for now let's not delete it.
//   TSignalHandler *ih  = GetSignalHandler();
//   ih->Remove();
//   SetSignalHandler(0);
//   delete ih;
}

//______________________________________________________________________________
void TRint::ExecLogon()
{
   // Execute logon macro's. There are three levels of logon macros that
   // will be executed: the system logon etc/system.rootlogon.C, the global
   // user logon ~/.rootlogon.C and the local ./.rootlogon.C. For backward
   // compatibility also the logon macro as specified by the Rint.Logon
   // environment setting, by default ./rootlogon.C, will be executed.
   // No logon macros will be executed when the system is started with
   // the -n option.

   if (NoLogOpt()) return;

   TString name = ".rootlogon.C";
   TString sname = "system";
   sname += name;
#ifdef ROOTETCDIR
   char *s = gSystem->ConcatFileName(ROOTETCDIR, sname);
#else
   TString etc = gRootDir;
#ifdef WIN32
   etc += "\\etc";
#else
   etc += "/etc";
#endif
   char *s = gSystem->ConcatFileName(etc, sname);
#endif
   if (!gSystem->AccessPathName(s, kReadPermission)) {
      ProcessFile(s);
   }
   delete [] s;
   s = gSystem->ConcatFileName(gSystem->HomeDirectory(), name);
   if (!gSystem->AccessPathName(s, kReadPermission)) {
      ProcessFile(s);
   }
   delete [] s;
   // avoid executing ~/.rootlogon.C twice
   if (strcmp(gSystem->HomeDirectory(), gSystem->WorkingDirectory())) {
      if (!gSystem->AccessPathName(name, kReadPermission))
         ProcessFile(name);
   }

   // execute also the logon macro specified by "Rint.Logon"
   const char *logon = gEnv->GetValue("Rint.Logon", (char*)0);
   if (logon) {
      char *mac = gSystem->Which(TROOT::GetMacroPath(), logon, kReadPermission);
      if (mac)
         ProcessFile(logon);
      delete [] mac;
   }
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
   volatile Bool_t needGetlinemInit = kFALSE;

   if (strlen(WorkingDirectory())) {
      // if directory specified as argument make it the working directory
      gSystem->ChangeDirectory(WorkingDirectory());
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
   }

   // Process shell command line input files
   if (InputFiles()) {
      // Make sure that calls into the event loop
      // ignore end-of-file on the terminal.
      fInputHandler->DeActivate();
      TIter next(InputFiles());
      RETRY {
         retval = 0; error = 0;
         Int_t nfile = 0;
         TObjString *file;
         while ((file = (TObjString *)next())) {
            char cmd[kMAXPATHLEN+50];
            if (!fNcmd)
               printf("\n");
            if (file->String().EndsWith(".root") || file->String().BeginsWith("file:")) {
               file->String().ReplaceAll("\\","/");
               const char *rfile = (const char*)file->String();
               Printf("Attaching file %s as _file%d...", rfile, nfile);
               snprintf(cmd, kMAXPATHLEN+50, "TFile *_file%d = TFile::Open(\"%s\")", nfile++, rfile);
            } else {
               Printf("Processing %s...", (const char*)file->String());
               snprintf(cmd, kMAXPATHLEN+50, ".x %s", (const char*)file->String());
            }
            Getlinem(kCleanUp, 0);
            Gl_histadd(cmd);
            fNcmd++;

            // The ProcessLine might throw an 'exception'.  In this case,
            // GetLinem(kInit,"Root >") is called and we are jump back
            // to RETRY ... and we have to avoid the Getlinem(kInit, GetPrompt());
            needGetlinemInit = kFALSE;
            retval = ProcessLine(cmd, kFALSE, &error);
            gCint->EndOfLineAction();

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
void TRint::PrintLogo(Bool_t lite)
{
   // Print the ROOT logo on standard output.

   const char *root_version = gROOT->GetVersion();

   if (!lite) {
      static const char *months[] = {"January","February","March","April","May",
                                     "June","July","August","September","October",
                                     "November","December"};
      Int_t idatqq = gROOT->GetVersionDate();
      Int_t iday   = idatqq%100;
      Int_t imonth = (idatqq/100)%100;
      Int_t iyear  = (idatqq/10000);
      char *version_date = Form("%d %s %4d",iday,months[imonth-1],iyear);

      Printf("  *******************************************");
      Printf("  *                                         *");
      Printf("  *        W E L C O M E  to  R O O T       *");
      Printf("  *                                         *");
      Printf("  *   Version%10s %17s   *", root_version, version_date);
      Printf("  *                                         *");
      Printf("  *  You are welcome to visit our Web site  *");
      Printf("  *          http://root.cern.ch            *");
      Printf("  *                                         *");
      Printf("  *******************************************\n");
   }

   Printf("ROOT %s (%s@%d, %s on %s)", root_version, gROOT->GetSvnBranch(),
          gROOT->GetSvnRevision(), gROOT->GetSvnDate(),
          gSystem->GetBuildArch());

   if (!lite)
      gCint->PrintIntro();

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

   char *s = gCint->GetPrompt();
   if (s[0])
      strlcpy(fPrompt, s, sizeof(fPrompt));
   else
      snprintf(fPrompt, sizeof(fPrompt), fDefaultPrompt.Data(), fNcmd);

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

      if (!gCint->GetMore() && !sline.IsNull()) fNcmd++;

      // prevent recursive calling of this input handler
      fInputHandler->DeActivate();

      if (gROOT->Timer()) timer.Start();

      Bool_t added = kFALSE;

      // This is needed when working with remote sessions
      SetBit(kProcessRemotely);

#ifdef R__EH
      try {
#endif
         TRY {
            if (!sline.IsNull())
               LineProcessed(sline);
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
         gCint->EndOfLineAction();

      gTabCom->ClearAll();
      Getlinem(kInit, GetPrompt());
   }
   return kTRUE;
}

//______________________________________________________________________________
void TRint::HandleException(Int_t sig)
{
   // Handle exceptions (kSigBus, kSigSegmentationViolation,
   // kSigIllegalInstruction and kSigFloatingException) trapped in TSystem.
   // Specific TApplication implementations may want something different here.

   if (TROOT::Initialized()) {
      if (gException) {
         Getlinem(kCleanUp, 0);
         Getlinem(kInit, "Root > ");
      }
   }
   TApplication::HandleException(sig);
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
      delete gTabCom;
      gTabCom = 0;

      //Execute logoff macro
      const char *logoff;
      logoff = gEnv->GetValue("Rint.Logoff", (char*)0);
      if (logoff && !NoLogOpt()) {
         char *mac = gSystem->Which(TROOT::GetMacroPath(), logoff, kReadPermission);
         if (mac)
            ProcessFile(logoff);
         delete [] mac;
      }

      gROOT->CloseFiles(); // Close any files or sockets before emptying CINT.
      gInterpreter->ResetGlobals();

      TApplication::Terminate(status);
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

//______________________________________________________________________________
Long_t TRint::ProcessRemote(const char *line, Int_t *)
{
   // Process the content of a line starting with ".R" (already stripped-off)
   // The format is
   //      [user@]host[:dir] [-l user] [-d dbg] [script]
   // The variable 'dir' is the remote directory to be used as working dir.
   // The username can be specified in two ways, "-l" having the priority
   // (as in ssh).
   // A 'dbg' value > 0 gives increasing verbosity.
   // The last argument 'script' allows to specify an alternative script to
   // be executed remotely to startup the session.

   Long_t ret = TApplication::ProcessRemote(line);

   if (ret == 1) {
      if (fAppRemote) {
         TString prompt = Form("%s:root [%%d] ", fAppRemote->ApplicationName());
         SetPrompt(prompt);
      } else {
         SetPrompt("root [%d] ");
      }
   }

   return ret;
}
