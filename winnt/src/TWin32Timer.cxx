// @(#)root/winnt:$Name:  $:$Id: TWin32Timer.cxx,v 1.2 2001/05/16 08:53:16 brun Exp $
// Author: Valery Fine(fine@mail.cern.ch)   29/09/98

#include <process.h>

#include "Windows4Root.h"
#include "TTimer.h"
#include "TROOT.h"
#include "TWin32Timer.h"
#include "TWin32HookViaThread.h"
#include "TGWin32Command.h"
#include "TInterpreter.h"

struct WIN32TIMETHREAD {
                        HANDLE ThrSem;
                        TWin32Timer *ti;
                       }  ;

enum ETimerCallbackCmd {kCreateTimer, kKillTimer};
const Char_t *TIMERCLASS = "Timer";
//*-*
//*-* Macros to call the Callback methods via Timer thread:
//*-*
#define CallMethodThread(_function,_p1,_p2,_p3)    \
  else                                             \
  {                                                \
      TWin32SendWaitClass code(this,(UInt_t)k##_function,(UInt_t)(_p1),(UInt_t)(_p2),(UInt_t)(_p3)); \
      ExecTimerThread(&code);                      \
      code.Wait();                                 \
  }
#define  ReturnMethodThread(_type,_function,_p1,_p2) \
  else                                               \
  {                                                  \
      _type _local;                                  \
      TWin32SendWaitClass code(this,(UInt_t)k##_function,(UInt_t)(_p1),(UInt_t)(_p2),(UInt_t)(&_local)); \
      ExecTimerThread(&code);                        \
      code.Wait();                                   \
      return _local;                                 \
  }

//*-*
#define CallWindowMethod1(_function,_p1)             \
  if ( IsTimeThread())                               \
  {TWin32Timer::_function(_p1);}                       \
    CallMethodThread(_function,_p1,0,0)
//*-*
#define CallWindowMethod(_function)                  \
  if ( IsTimeThread())                               \
  {TWin32Timer::_function();}                          \
    CallMethodThread(_function,0,0,0)

//*-*
#define ReturnWindowMethod1(_type,_function,_p1)     \
  if ( IsTimeThread())                               \
  {return TWin32Timer::_function(_p1);}                \
    ReturnMethodThread(_type,_function,_p1,0)
//*-*
#define ReturnWindowMethod(_type,_function)          \
  if ( IsTimeThread())                               \
  {return TWin32Timer::_function();}                   \
    ReturnMethodThread(_type,_function,0,0)

//______________________________________________________________________________
static VOID CALLBACK DispatchTimers(HWND hwnd, UINT uMsg, UINT idEvent, DWORD dwTime)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
//*-*
//*-* HWND  hwnd,       // handle of window for timer messages
//*-* UINT uMsg,        // WM_TIMER message
//*-* UINT idEvent,     // timer identifier (pointer to TTimer object)
//*-* DWORD dwTime      // current system time
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

   TTimer *ti = (TTimer *)idEvent;
   if (ti) {
      if (ti->IsAsync())
         ti->Notify();
      else
         gROOT->ProcessLine(Form("((TTimer *)0x%lx)->Notify();",(Long_t)ti));
   }
}

//______________________________________________________________________________
static LRESULT APIENTRY WndTimer(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Main Universal Windows procedure to manage all dispatched events     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
   return ::DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//______________________________________________________________________________
static unsigned int _stdcall ROOT_TimerLoop(void *threadcmd)
{
  //---------------------------------------
  // Create windows
  HWND fhdTimerWindow = CreateWindowEx(NULL,
                                 TIMERCLASS,
                                 NULL,                             // address of window name
                                 WS_DISABLED ,                     // window style
                                 0,0,                              // start positio of the window,
                                 0, 0,                             // size of the window
                                 NULL,                             // handle of parent of owner window
                                 NULL,                             // handle of menu, or child-window identifier
                                 GetModuleHandle(NULL),            // handle of application instance
                                 NULL);                            // address of window-creation data
   HANDLE ThrSem = ((WIN32TIMETHREAD *)threadcmd)->ThrSem;
   ((WIN32TIMETHREAD *)threadcmd)->ti->SetHWND(fhdTimerWindow);
  //---------------------------------------
  MSG msg;
  int erret;  // GetMessage result

  ReleaseSemaphore(ThrSem, 1, NULL);
  Bool_t EventLoopStop = kFALSE;
  // create timer
  while(!EventLoopStop)
  {
     if (EventLoopStop = (!(erret=GetMessage(&msg,NULL,0,0)) || erret == -1))
                                                                   continue;
     if (msg.hwnd == NULL && (msg.message == ROOT_CMD || msg.message == ROOT_SYNCH_CMD))
           if (TWin32HookViaThread::ExecuteEvent(&msg, msg.message==ROOT_SYNCH_CMD)) continue;

     TranslateMessage(&msg);
     DispatchMessage(&msg);
  }
  if (erret == -1)
  {
       erret = GetLastError();
       fprintf(stderr," *** Error **** TimerLoop: %d \n", erret);
  }

  if (msg.wParam) ReleaseSemaphore((HANDLE) msg.wParam, 1, NULL);

  _endthreadex(0);
  return 0;
} /* ROOT_MsgLoop */


//______________________________________________________________________________
TWin32Timer::TWin32Timer()
{
  fhdTimerWindow   = 0;
  fhdTimerThread   = 0;
  fhdTimerThreadId = 0;
}
//______________________________________________________________________________
TWin32Timer::~TWin32Timer()
{
   if (fhdTimerThreadId) {
       PostThreadMessage(fhdTimerThreadId,WM_QUIT,0,0);
       if (WaitForSingleObject(fhdTimerThread,10000)==WAIT_FAILED)
                              TerminateThread(fhdTimerThread, -1);
       CloseHandle(fhdTimerThread);
   }
}
//______________________________________________________________________________
Int_t TWin32Timer::CreateTimerThread()
{
  // Register class "Timer"
  HMODULE instance = GetModuleHandle(NULL);
  static const WNDCLASS timerwindowclass = {
                                             CS_GLOBALCLASS
                                           , WndTimer
                                           , 0, 0
                                           , instance
                                           , NULL, NULL, NULL, NULL
                                           , TIMERCLASS};
  WNDCLASSEX timerinfo;
  if (GetClassInfoEx(instance,TIMERCLASS,&timerinfo))
       return 0;
  if (!RegisterClass( &timerwindowclass))
  {
       DWORD l_err = GetLastError();
       printf(" Last Error is %d \n", l_err);
       return -1;
  }

  WIN32TIMETHREAD threadcmd;


  //
  //  Create thread to do the cmd loop
  //

  threadcmd.ThrSem = CreateSemaphore(NULL, 0, 1, NULL);
  threadcmd.ti = this;

//  fhdTimerThread = (HANDLE)_beginthreadex(NULL,0,  ROOT_TimerLoop,
  fhdTimerThread = (unsigned long *) _beginthreadex(NULL,0,  ROOT_TimerLoop,
                   (LPVOID) &threadcmd, 0, ((unsigned *)&fhdTimerThreadId));

  if (Int_t(fhdTimerThread)  == -1){
    int  erret = GetLastError();
    printf(" *** Error *** CreatTimerThread <Thread was not created> %d \n", erret);
  }

  WaitForSingleObject(threadcmd.ThrSem, INFINITE);
  CloseHandle(threadcmd.ThrSem);

  return 0;
}
//______________________________________________________________________________
UInt_t TWin32Timer::CreateTimer(TTimer *timer)
{
  if(!fhdTimerThreadId) CreateTimerThread();
  CallWindowMethod1(CreateTimer,timer);
  return 0;
}
//______________________________________________________________________________
void TWin32Timer::CreateTimerCB(TTimer *timer)
{
  if (timer)
    timer->SetTimerID((UInt_t)(::SetTimer(fhdTimerWindow,(UINT)timer,
                      (unsigned long)timer->GetTime(),
                      (TIMERPROC) ::DispatchTimers)) );
}
//______________________________________________________________________________
void TWin32Timer::ExecTimerThread(TGWin32Command *command)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  Execute command via "Timer" thread
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
// Some extra flag is needed to mark the command = 0 turn !!!
 TGWin32Command *code = command;
 if (!code) code = new TWin32SendClass(this);
 fSendFlag = 1;
 int i = ExecCommand(code,kFALSE);
}
//______________________________________________________________________________
Bool_t  TWin32Timer::ExecCommand(TGWin32Command *command,Bool_t synch)
{
// To exec a command coming from the other threads

 BOOL postresult;
 ERoot_Msgs cmd = ROOT_CMD;
 if (fhdTimerThreadId == GetCurrentThreadId())
         printf("TWin32Timer::ExecCommand --- > The dead lock danger\n");

 if (synch) cmd =  ROOT_SYNCH_CMD;
 while (!(postresult = PostThreadMessage(fhdTimerThreadId,
                             cmd,
                             (WPARAM)command->GetCOP(),
                             (LPARAM)command))
       ){ ; }
 return postresult;
}

//______________________________________________________________________________
Bool_t TWin32Timer::IsTimeThread(){
  return fhdTimerThreadId  == GetCurrentThreadId();
}

//______________________________________________________________________________
void TWin32Timer::KillTimer(TTimer *timer)
{
  CallWindowMethod1(KillTimer,timer);
}
//______________________________________________________________________________
void TWin32Timer::KillTimerCB(TTimer *timer)
{
   if(timer) {
 //      ::KillTimer(NULL,timer->GetTimerID());
       ::KillTimer(fhdTimerWindow,(UINT)timer);
       timer->SetTimerID(0);
   }
}


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   Callback methods:
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//______________________________________________________________________________
void TWin32Timer::ExecThreadCB(TWin32SendClass *command)
{
    ETimerCallbackCmd cmd = (ETimerCallbackCmd)(command->GetData(0));
    Bool_t debug = kFALSE;
    char *listcmd[] = {
                 "CreateTimer"
                ,"KillTimer"
    };

    if (gDebug) printf("TWin32Timer: commamd %d: %s",cmd,listcmd[cmd]);
    switch (cmd)
    {
    case kCreateTimer:
        {
          TTimer *ti = (TTimer *)(command->GetData(1));
          if (gDebug) printf(" %lx ", (Long_t)ti);
          CreateTimerCB(ti);
          break;
        }
    case kKillTimer:
        {
          TTimer *ti = (TTimer *)(command->GetData(1));
          if (gDebug) printf(" %lx ", (Long_t)ti);
          KillTimerCB(ti);
          break;
        }
    default:
        break;
    }
    if (gDebug) printf(" \n");
    if (LOWORD(command->GetCOP()) == kSendWaitClass)
        ((TWin32SendWaitClass *)command)->Release();
    else
        delete command;
}
