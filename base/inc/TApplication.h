// @(#)root/base:$Name:  $:$Id: TApplication.h,v 1.4 2001/06/01 16:18:44 rdm Exp $
// Author: Fons Rademakers   22/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TApplication
#define ROOT_TApplication


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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TApplicationImp
#include "TApplicationImp.h"
#endif

class TObjArray;
class TTimer;
class TSignalHandler;


class TApplication : public TObject {

private:
   Int_t              fArgc;           //Number of command line arguments
   char             **fArgv;           //Command line arguments
   TApplicationImp   *fAppImp;         //!Window system specific application implementation
   Bool_t             fReturnFromRun;  //When true return from Run()
   Bool_t             fNoLog;          //Do not process logon and logoff macros
   Bool_t             fNoLogo;         //Do not show splash screen and welcome message
   Bool_t             fQuit;           //Exit after having processed input files
   TObjArray         *fFiles;          //Array of input files (TObjString's)
   char              *fIdleCommand;    //Command to execute while application is idle
   TTimer            *fIdleTimer;      //Idle timer
   TSignalHandler    *fSigHandler;     //Interrupt handler

protected:
   TApplication();
   virtual void Help(const char *line);
   virtual void InitializeColors();
   virtual void LoadGraphicsLibs();
   void         SetReturnFromRun(Bool_t ret) { fReturnFromRun = ret; }
   void         SetSignalHandler(TSignalHandler *sh) { fSigHandler = sh; }

public:
   TApplication(const char *appClassName, int *argc, char **argv,
                void *options = 0, int numOptions = 0);
   virtual ~TApplication();

   virtual void    GetOptions(int *argc, char **argv);
   TSignalHandler *GetSignalHandler() const { return fSigHandler; }
   virtual void    HandleIdleTimer();
   virtual void    HandleTermInput() { }
   virtual void    Init() { fAppImp->Init(); }
   virtual void    ProcessLine(const char *line, Bool_t sync = kFALSE);
   virtual void    ProcessFile(const char *line);
   virtual void    Run(Bool_t retrn = kFALSE);
   virtual void    SetIdleTimer(UInt_t idleTimeInSec, const char *command);
   virtual void    RemoveIdleTimer();
   const char     *GetIdleCommand() const { return fIdleCommand; }
   virtual void    StartIdleing();
   virtual void    StopIdleing();
   virtual void    Terminate(int status = 0);

   virtual const char *ApplicationName() const { return fAppImp->ApplicationName(); }
   virtual void    Show()    { fAppImp->Show(); }
   virtual void    Hide()    { fAppImp->Hide(); }
   virtual TApplicationImp *GetApplicationImp(){ return fAppImp;}
   virtual void    Iconify() { fAppImp->Iconify(); }
   virtual Bool_t  IsCmdThread(){ return fAppImp->IsCmdThread(); }
   virtual void    Open()    { fAppImp->Open(); }
   virtual void    Raise()   { fAppImp->Raise(); }
   virtual void    Lower()   { fAppImp->Lower(); }

   int             Argc() const  { return fArgc; }
   char          **Argv() const  { return fArgv; }
   char           *Argv(int index) const { return fArgv[index]; }
   Bool_t          NoLogOpt() const { return fNoLog; }
   Bool_t          NoLogoOpt() const { return fNoLogo; }
   Bool_t          QuitOpt() const { return fQuit; }
   TObjArray      *InputFiles() const { return fFiles; }
   void            ClearInputFiles();

   Bool_t          ReturnFromRun() const { return fReturnFromRun; }

   static void     CreateApplication();

   ClassDef(TApplication,0)  //GUI application singleton
};

R__EXTERN TApplication *gApplication;

#endif
