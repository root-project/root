// @(#)root/win32:$Name:  $:$Id: TWin32Application.h,v 1.2 2001/04/22 16:00:56 rdm Exp $
// Author: Valery Fine   10/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Application
#define ROOT_TWin32Application

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32                                                              //
//                                                                      //
// Interface to low level Windows32. This class gives access to basic   //
// Win32 graphics, pixmap, text and font handling routines.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TApplicationImp
#include "TApplicationImp.h"
#endif

#include "Windows4Root.h"
#include <commctrl.h>

class TGWin32Command;

#ifdef _SC_
LPTHREAD_START_ROUTINE ROOT_CmdLoop(HANDLE ThrSem);
#else
unsigned int _stdcall ROOT_CmdLoop(HANDLE ThrSem);
#endif

class TWin32Application : public TApplicationImp  {

private:

  DWORD  fIDCmdThread;
  HANDLE fhdCmdThread;

  Int_t   CreateCmdThread();

public:

   TWin32Application() {};
   TWin32Application(const char *appClassName, int *argc, char **argv,
                   void *options, int numOptions);
   virtual ~TWin32Application();

   BOOL    ExecCommand(TGWin32Command *command, Bool_t synch=kFALSE);   // To exec a command coming from the other threads
   DWORD   GetCmdThreadID(){return fIDCmdThread;}
   void    Show();
   void    Hide();
   void    Iconify();
   Bool_t  IsCmdThread();
   void    Init();
   void    Open();
   void    Raise();
   void    Lower();

   // ClassDef(TWin32Application,0)

};

#endif
