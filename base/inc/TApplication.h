// @(#)root/base:$Name:  $:$Id: TApplication.h,v 1.21 2007/02/13 21:23:10 rdm Exp $
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

#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

#ifndef ROOT_TApplicationImp
#include "TApplicationImp.h"
#endif

class TObjArray;
class TTimer;
class TSignalHandler;


class TApplication : public TObject, public TQObject {

private:
   Int_t              fArgc;           //Number of com   mand line arguments
   char             **fArgv;           //Command line arguments
   TApplicationImp   *fAppImp;         //!Window system specific application implementation
   Bool_t             fIsRunning;      //True when in event loop (Run() has been called)
   Bool_t             fReturnFromRun;  //When true return from Run()
   Bool_t             fNoLog;          //Do not process logon and logoff macros
   Bool_t             fNoLogo;         //Do not show splash screen and welcome message
   Bool_t             fQuit;           //Exit after having processed input files
   Bool_t             fGraphInit;      //True if graphics has been initialized
   TObjArray         *fFiles;          //Array of input files (TObjString's)
   TString            fIdleCommand;    //Command to execute while application is idle
   TTimer            *fIdleTimer;      //Idle timer
   TSignalHandler    *fSigHandler;     //Interrupt handler

   static Bool_t      fgGraphInit;     // True if graphics libs initialized

   TApplication(const TApplication&);             // not implemented
   TApplication& operator=(const TApplication&);  // not implemented

protected:
   TApplication();

   void InitializeGraphics();
   virtual void Help(const char *line);
   virtual void InitializeColors();
   virtual void LoadGraphicsLibs();
   virtual void MakeBatch();
   void SetSignalHandler(TSignalHandler *sh) { fSigHandler = sh; }

public:
   // Load and initialize the graphics libraries
   class TLoadGraphicsLibs {
      public:
      TLoadGraphicsLibs();
   };

   TApplication(const char *appClassName, Int_t *argc, char **argv,
                void *options = 0, Int_t numOptions = 0);
   virtual ~TApplication();

   virtual void    GetOptions(Int_t *argc, char **argv);
   TSignalHandler *GetSignalHandler() const { return fSigHandler; }
   virtual void    SetEchoMode(Bool_t mode);

   virtual void    HandleIdleTimer();   //*SIGNAL*
   virtual Bool_t  HandleTermInput() { return kFALSE; }
   virtual void    Init() { fAppImp->Init(); }
   virtual Long_t  ProcessLine(const char *line, Bool_t sync = kFALSE, Int_t *error = 0);
   virtual Long_t  ProcessFile(const char *line, Int_t *error = 0);
   virtual void    Run(Bool_t retrn = kFALSE);
   virtual void    SetIdleTimer(UInt_t idleTimeInSec, const char *command);
   virtual void    RemoveIdleTimer();
   const char     *GetIdleCommand() const { return fIdleCommand; }
   virtual void    StartIdleing();
   virtual void    StopIdleing();

   virtual const char *ApplicationName() const { return fAppImp->ApplicationName(); }
   virtual void    Show()    { fAppImp->Show(); }
   virtual void    Hide()    { fAppImp->Hide(); }
   virtual TApplicationImp *GetApplicationImp(){ return fAppImp;}
   virtual void    Iconify() { fAppImp->Iconify(); }
   virtual Bool_t  IsCmdThread(){ return fAppImp->IsCmdThread(); }
   virtual void    Open()    { fAppImp->Open(); }
   virtual void    Raise()   { fAppImp->Raise(); }
   virtual void    Lower()   { fAppImp->Lower(); }

   Int_t           Argc() const  { return fArgc; }
   char          **Argv() const  { return fArgv; }
   char           *Argv(Int_t index) const { return fArgv ? fArgv[index] : 0; }
   Bool_t          NoLogOpt() const { return fNoLog; }
   Bool_t          NoLogoOpt() const { return fNoLogo; }
   Bool_t          QuitOpt() const { return fQuit; }
   TObjArray      *InputFiles() const { return fFiles; }
   void            ClearInputFiles();

   Bool_t          IsRunning() const { return fIsRunning; }
   Bool_t          ReturnFromRun() const { return fReturnFromRun; }
   void            SetReturnFromRun(Bool_t ret) { fReturnFromRun = ret; }

   static void     CreateApplication();

   virtual void    Terminate(Int_t status = 0);   //*SIGNAL*
   virtual void    KeyPressed(Int_t key);         //*SIGNAL*
   virtual void    ReturnPressed(char *text );    //*SIGNAL*

   ClassDef(TApplication,0)  //GUI application singleton
};

R__EXTERN TApplication *gApplication;

inline TApplication::TLoadGraphicsLibs::TLoadGraphicsLibs()
 { if (gApplication) gApplication->InitializeGraphics(); }

#endif
