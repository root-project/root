// @(#)root/win32:$Name:  $:$Id: TWin32Application.cxx,v 1.3 2001/04/23 08:11:52 brun Exp $
// Author: Valery Fine   10/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*The  W I N 3 2 A p p l i c a t i o n class-*-*-*-*-*-*-*
//*-*              ==========================================
//*-*
//*-*  Basic interface to the WIN32 window system
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include <process.h>

#include "TWin32Application.h"
#include "TApplication.h"
#include "TROOT.h"

#include "TException.h"
#include "TWin32ContextMenuImp.h"
#include "TGWin32Object.h"
#include "TContextMenu.h"
#include "TError.h"
#include "TControlBarButton.h"
#include "TWin32ControlBarImp.h"

#include "TWin32HookViaThread.h"

#include <windows.h>
#undef GetWindow

//______________________________________________________________________________
#ifdef _SC_
LPTHREAD_START_ROUTINE ROOT_CmdLoop(HANDLE ThrSem)
#else
unsigned int _stdcall ROOT_CmdLoop(HANDLE ThrSem)
#endif
//*-*-*-*-*-*-*-*-*-*-*-*-* ROOT_CmdLoop*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                       ============
//*-*  Launch a separate thread to handle the ROOT command  messages
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 {
   MSG msg;
   int erret;  // GetMessage result

   ReleaseSemaphore(ThrSem, 1, NULL);
   Bool_t EventLoopStop = kFALSE;
   while(!EventLoopStop)
   {
       if (EventLoopStop = (!(erret=GetMessage(&msg,NULL,0,0)) || erret == -1))
       {
           int err = GetLastError();
           if (err) printf( "ROOT_CmdLoop: GetMessage Error %d Last error was %d\n", erret, err);
           continue;
       }

//*-*
//*-*   GetMessage( ... ):
//*-* If the function retrieves a message other than WM_QUIT,
//*-*     the return value is TRUE.
//*-* If the function retrieves the WM_QUIT,
//*-*     the return value is FALSE.
//*-* If there is an error,
//*-*     the return value is -1.
//*-*

       if ((msg.hwnd == NULL) && (msg.message == ROOT_CMD || msg.message == ROOT_SYNCH_CMD)) {

           if (TWin32HookViaThread::ExecuteEvent(&msg, msg.message==ROOT_SYNCH_CMD)) continue;
       }
       TranslateMessage(&msg);
       DispatchMessage(&msg);
   }

   printf(" Leaving thread \n");
   if (erret == -1)
   {
       erret = GetLastError();
       Error("CmdLoop", "Error in GetMessage");
       Printf(" %d \n", erret);
   }
//   _endthreadex(0);
   return 0;
} /* ROOT_CmdLoop */


//______________________________________________________________________________
#ifdef _SC_
LPTHREAD_START_ROUTINE ROOT_DlgLoop(HANDLE ThrSem)
#else
unsigned int ROOT_DlgLoop(HANDLE ThrSem)
#endif
//*-*-*-*-*-*-*-*-*-*-*-*-* ROOT_DlgLoop*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                       ============
//*-*  Launch a separate thread to handle the ROOT command  messages
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 {
   MSG msg;
   int erret;  // GetMessage result

   ReleaseSemaphore(ThrSem, 1, NULL);
   Bool_t EventLoopStop = kFALSE;
   while(!EventLoopStop)
   {
     if (EventLoopStop = (!(erret=GetMessage(&msg,NULL,0,0)) || erret == -1))
                                                                   continue;
//*-*
//*-*   GetMessage( ... ):
//*-* If the function retrieves a message other than WM_QUIT,
//*-*     the return value is TRUE.
//*-* If the function retrieves the WM_QUIT,
//*-*     the return value is FALSE.
//*-* If there is an error,
//*-*     the return value is -1.
//*-*

       if (msg.hwnd == NULL) ;
       else {
          if (msg.message != ROOT_CMD && msg.message != ROOT_SYNCH_CMD)
                           TranslateMessage(&msg);
          DispatchMessage(&msg);
       }
    }
    if (erret == -1)
    {
        erret = GetLastError();
        Error("CmdLoop", "Error in GetMessage");
        Printf( "%d \n", erret);
    }
    _endthreadex(0);
    return 0;
} /* ROOT_DlgLoop */



// ClassImp(TWin32Application)

//______________________________________________________________________________
TWin32Application::TWin32Application(const char *appClassName, int *argc,
                                     char **argv, void *options, int numOptions)
{
   fApplicationName = appClassName;
   SetConsoleTitle(appClassName);
   CreateCmdThread();
//   gVirtualX->Init();
//   CreateDlgThread();
}
//______________________________________________________________________________
   TWin32Application::~TWin32Application() {

    if (fIDCmdThread) {
        PostThreadMessage(fIDCmdThread,WM_QUIT,0,0);
        if (WaitForSingleObject(fhdCmdThread,10000)==WAIT_FAILED)
                               TerminateThread(fhdCmdThread, -1); ;
        CloseHandle(fhdCmdThread);
    }

    if (fIDDlgThread) {
        PostThreadMessage(fIDDlgThread,WM_QUIT,0,0);
        if (WaitForSingleObject(fhdDlgThread,10000)==WAIT_FAILED)
                               TerminateThread(fhdDlgThread, -1);
        CloseHandle(fhdDlgThread);
    }

}

//______________________________________________________________________________
Int_t TWin32Application::CreateCmdThread()
{
  HANDLE ThrSem;

  //
  //  Create thread to do the cmd loop
  //

  ThrSem = CreateSemaphore(NULL, 0, 1, NULL);



#ifdef _SC_
  if ((Int_t)(fhdCmdThread = (HANDLE)_beginthreadex(NULL,0, (void *) ROOT_CmdLoop,
                    (LPVOID) ThrSem, 0, (void *)&fIDCmdThread)) == -1){
#else
  if ((Int_t)(fhdCmdThread = (HANDLE)_beginthreadex(NULL,0,  ROOT_CmdLoop,
                    (LPVOID) ThrSem, 0, (unsigned int *)&fIDCmdThread)) == -1){
#endif

      int  erret = GetLastError();
      Error("CreatCmdThread", "Thread was not created");
      Printf(" %d \n", erret);
  }

  WaitForSingleObject(ThrSem, INFINITE);
  CloseHandle(ThrSem);

  return 0;
}

//______________________________________________________________________________
Int_t TWin32Application::CreateDlgThread()
{
  HANDLE ThrSem;

  //
  //  Create thread to do the Dialogs loop
  //

  ThrSem = CreateSemaphore(NULL, 0, 1, NULL);

  if (!(fhdDlgThread = CreateThread(NULL,0, (LPTHREAD_START_ROUTINE) ROOT_DlgLoop,
                 (LPVOID) ThrSem, 0,  &fIDDlgThread))){
      int  erret = GetLastError();
      Error("CreatCmdThread", "Thread was not created");
      Printf("%d \n", erret);
  }

  WaitForSingleObject(ThrSem, INFINITE);
  CloseHandle(ThrSem);

  return 0;
}

//______________________________________________________________________________
BOOL  TWin32Application::ExecCommand(TGWin32Command *command,Bool_t synch)
{
// To exec a command coming from the other threads

 BOOL postresult;
 ERoot_Msgs cmd = ROOT_CMD;
 if (fIDCmdThread == GetCurrentThreadId())
         Warning("ExecCommand","The dead lock danger");

 if (synch) cmd =  ROOT_SYNCH_CMD;
 while (!(postresult = PostThreadMessage(fIDCmdThread,
                             cmd,
                             (WPARAM)command->GetCOP(),
                             (LPARAM)command))
       ){ ; }
 return postresult;
}

//______________________________________________________________________________
void    TWin32Application::Show(){; }
//______________________________________________________________________________
void    TWin32Application::Hide(){; }
//______________________________________________________________________________
void    TWin32Application::Iconify(){; }
//______________________________________________________________________________
void    TWin32Application::Init(){ ; }
//______________________________________________________________________________
Bool_t  TWin32Application::IsCmdThread()
{
   return (GetCurrentThreadId() == fIDCmdThread) ? kTRUE : kFALSE;
}
//______________________________________________________________________________
   void    TWin32Application::Open(){; }
//______________________________________________________________________________
   void    TWin32Application::Raise(){; }
//______________________________________________________________________________
   void    TWin32Application::Lower(){; }
