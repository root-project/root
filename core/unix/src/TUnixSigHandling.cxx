// @(#)root/unix:$Id: 887c618d89c4ed436e4034fc133f468fecad651b $
// Author: Zhe Zhang   10/03/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUnixSigHandling                                                     //
//                                                                      //
// Class providing an interface to the UNIX Operating System.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"
#include "RConfig.h"
#include "TUnixSigHandling.h"
#include "TUnixSystem.h"
#include "TROOT.h"
#include "TError.h"
#include "TEnv.h"
#include "Getline.h"
#include "TOrdCollection.h"
#include "TApplication.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TVirtualMutex.h"
#include "TObjArray.h"
#include <map>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#ifdef __linux__
#include <syscall.h>
#endif

//#define G__OLDEXPAND

#include <unistd.h>
#include <stdlib.h>

#ifndef R__WIN32
#include <poll.h>
#endif

#include <sys/types.h>
#if defined(R__AIX) || defined(R__SOLARIS)
#   include <sys/select.h>
#endif
#if defined(R__LINUX) || defined(R__HURD)
#   ifndef SIGSYS
#      define SIGSYS  SIGUNUSED       // SIGSYS does not exist in linux ??
#   endif
#endif

#include <utime.h>
#include <signal.h>
#include <errno.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>

// stack trace code
#if (defined(R__LINUX) || defined(R__HURD)) && !defined(R__WINGCC)
#   if __GLIBC__ == 2 && __GLIBC_MINOR__ >= 1
#      define HAVE_BACKTRACE_SYMBOLS_FD
#   endif
#   define HAVE_DLADDR
#endif
#if defined(R__MACOSX)
#   if defined(MAC_OS_X_VERSION_10_5)
#      define HAVE_BACKTRACE_SYMBOLS_FD
#      define HAVE_DLADDR
#   else
#      define USE_GDB_STACK_TRACE
#   endif
#endif

#ifdef HAVE_BACKTRACE_SYMBOLS_FD
#   include <execinfo.h>
#endif
#ifdef HAVE_DLADDR
#   ifndef __USE_GNU
#      define __USE_GNU
#   endif
#   include <dlfcn.h>
#endif

#ifdef HAVE_BACKTRACE_SYMBOLS_FD
   // The maximum stack trace depth for systems where we request the
   // stack depth separately (currently glibc-based systems).
   static const int kMAX_BACKTRACE_DEPTH = 128;
#endif

class TFdSet;
 
#if defined(HAVE_DLADDR) && !defined(R__MACOSX)
////////////////////////////////////////////////////////////////////////////////

static void SetRootSys()
{
#ifndef ROOTPREFIX
   void *addr = (void *)SetRootSys;
   Dl_info info;
   if (dladdr(addr, &info) && info.dli_fname && info.dli_fname[0]) {
      char respath[kMAXPATHLEN];
      if (!realpath(info.dli_fname, respath)) {
         if (!gSystem->Getenv("ROOTSYS"))
         ::SysError("TUnixSigHandling::SetRootSys", "error getting realpath of libCore, please set ROOTSYS in the shell");
      } else {
         TString rs = gSystem->DirName(respath);
         gSystem->Setenv("ROOTSYS", gSystem->DirName(rs));
      }
   }
#else
   return;
#endif
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Unix signal handler.

static void SigHandler(ESignals sig)
{
   if (gSigHandling)
      gSigHandling->DispatchSignals(sig);
}

////////////////////////////////////////////////////////////////////////////////
/// Async-signal-safe Write functions.

static int SignalSafeWrite(int fd, const char *text) {
   const char *buffer = text;
   size_t count = strlen(text);
   ssize_t written = 0;
   while (count) {
      written = write(fd, buffer, count);
      if (written == -1) {
         if (errno == EINTR) { continue; }
         else { return -errno; }
      }
      count -= written;
      buffer += written;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Async-signal-safe Read functions.

static int SignalSafeRead(int fd, char *inbuf, size_t len, int timeout=-1) {
   char *buf = inbuf;
   size_t count = len;
   ssize_t complete = 0;
   std::chrono::time_point<std::chrono::steady_clock> endTime = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
   int flags;
   if (timeout < 0) {
      flags = O_NONBLOCK;  // Prevents us from trying to set / restore flags later.
   } else if ((-1 == (flags = fcntl(fd, F_GETFL)))) {
      return -errno;
   } else { }
   if ((flags & O_NONBLOCK) != O_NONBLOCK) {
      if (-1 == fcntl(fd, F_SETFL, flags | O_NONBLOCK)) {
         return -errno;
      }
   }
   while (count) {
      if (timeout >= 0) {
         struct pollfd pollInfo{fd, POLLIN, 0};
         int msRemaining = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-std::chrono::steady_clock::now()).count();
         if (msRemaining > 0) {
            if (poll(&pollInfo, 1, msRemaining) == 0) {
               if ((flags & O_NONBLOCK) != O_NONBLOCK) {
                  fcntl(fd, F_SETFL, flags);
               }
               return -ETIMEDOUT;
            }
         } else if (msRemaining < 0) {
            if ((flags & O_NONBLOCK) != O_NONBLOCK) {
               fcntl(fd, F_SETFL, flags);
            }
            return -ETIMEDOUT;
         } else { }
      }
      complete = read(fd, buf, count);
      if (complete == -1) {
         if (errno == EINTR) { continue; }
         else if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) { continue; }
         else {
            int origErrno = errno;
            if ((flags & O_NONBLOCK) != O_NONBLOCK) {
               fcntl(fd, F_SETFL, flags);
            }
            return -origErrno;
         }
      }
      count -= complete;
      buf += complete;
   }
   if ((flags & O_NONBLOCK) != O_NONBLOCK) {
      fcntl(fd, F_SETFL, flags);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Async-signal-safe Write Error functions.

static int SignalSafeErrWrite(const char *text) {
   return SignalSafeWrite(2, text);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Static Protected Unix StackTrace functions.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//---- helper -----------------------------------------------------------------

static const int kStringLength = 255;

static struct StackTraceHelper_t {
   char  fShellExec[kStringLength];
   char  fPidString[kStringLength];
   char  fPidNum[kStringLength];
   int   fParentToChild[2];
   int   fChildToParent[2];
   std::unique_ptr<std::thread> fHelperThread;
} gStackTraceHelper = {       // the order of the signals should be identical
   { },
   { },
   { },
   {-1,-1},
   {-1,-1},
   nullptr
};

static char * const kStackArgv[] = {gStackTraceHelper.fShellExec, gStackTraceHelper.fPidString, gStackTraceHelper.fPidNum, nullptr};

//////////////////////////////////////////////////////////////////////////////
static char * const *GetStackArgv() {
   return kStackArgv;
}

ClassImp(TUnixSigHandling)

////////////////////////////////////////////////////////////////////////////////

TUnixSigHandling::TUnixSigHandling() : TSigHandling("Unix", "Unix Signal Handling")
{ }

////////////////////////////////////////////////////////////////////////////////
/// Reset to original state.

TUnixSigHandling::~TUnixSigHandling()
{
   UnixResetSignals();
   delete fSignals;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize Unix system interface.

void TUnixSigHandling::Init()
{
   TSigHandling::Init();

   fSignals    = new TFdSet;

   //--- install default handlers
   UnixSignal(kSigChild,                 SigHandler);
   UnixSignal(kSigBus,                   SigHandler);
   UnixSignal(kSigSegmentationViolation, SigHandler);
   UnixSignal(kSigIllegalInstruction,    SigHandler);
   UnixSignal(kSigSystem,                SigHandler);
   UnixSignal(kSigPipe,                  SigHandler);
   UnixSignal(kSigAlarm,                 SigHandler);
   UnixSignal(kSigUrgent,                SigHandler);
   UnixSignal(kSigFloatingException,     SigHandler);
   UnixSignal(kSigWindowChanged,         SigHandler);

#if defined(R__MACOSX)
   // trap loading of all dylibs to register dylib name,
   // sets also ROOTSYS if built without ROOTPREFIX
   _dyld_register_func_for_add_image(DylibAdded);
#elif defined(HAVE_DLADDR)
   SetRootSys();
#endif

#ifndef ROOTPREFIX
   gRootDir = gSystem->Getenv("ROOTSYS");
   if (gRootDir == 0)
      gRootDir= "/usr/local/root";
#else
   gRootDir = ROOTPREFIX;
#endif

   if(snprintf(gStackTraceHelper.fShellExec, kStringLength-1, "/bin/sh") >= kStringLength) {
      SignalSafeErrWrite("Unable to pre-allocate shell command path");
      return;
   }

#ifdef ROOTETCDIR
   if(snprintf(gStackTraceHelper.fPidString, kStringLength-1, "%s/gdb-backtrace.sh", ROOTETCDIR) >= kStringLength) {
      SignalSafeErrWrite("Unable to pre-allocate executable information");
      return;
   }
#else
   if(snprintf(gStackTraceHelper.fPidString, kStringLength-1, "%s/etc/gdb-backtrace.sh", gSystem->Getenv("ROOTSYS")) >= kStringLength) {
      SignalSafeErrWrite("Unable to pre-allocate executable information");
      return;
   }   
#endif

   gStackTraceHelper.fParentToChild[0] = -1;
   gStackTraceHelper.fParentToChild[1] = -1;
   gStackTraceHelper.fChildToParent[0] = -1;
   gStackTraceHelper.fChildToParent[1] = -1;

   StackTraceHelperInit();
}

////////////////////////////////////////////////////////////////////////////////
/// Add a signal handler to list of system signal handlers. Only adds
/// the handler if it is not already in the list of signal handlers.

void TUnixSigHandling::AddSignalHandler(TSignalHandler *h)
{
   R__LOCKGUARD2(gSystemMutex);

   if (h && fSignalHandler && (fSignalHandler->FindObject(h) == 0))
      fSignalHandler->Add(h);

   UnixSignal(h->GetSignal(), SigHandler);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a signal handler from list of signal handlers. Returns
/// the handler or 0 if the handler was not in the list of signal handlers.

TSignalHandler *TUnixSigHandling::RemoveSignalHandler(TSignalHandler *h)
{
   if (!h) return 0;

   R__LOCKGUARD2(gSystemMutex);

   TSignalHandler *oh;

   if (fSignalHandler)
      oh = (TSignalHandler *)fSignalHandler->Remove(h);
   else
      oh = 0;

   Bool_t last = kTRUE;
   TSignalHandler *hs;
   TIter next(fSignalHandler);

   while ((hs = (TSignalHandler*) next())) {
      if (hs->GetSignal() == h->GetSignal())
         last = kFALSE;
   }
   if (last)
      ResetSignal(h->GetSignal(), kTRUE);

   return oh;
}

////////////////////////////////////////////////////////////////////////////////
/// If reset is true reset the signal handler for the specified signal
/// to the default handler, else restore previous behaviour.

void TUnixSigHandling::ResetSignal(ESignals sig, Bool_t reset)
{
   if (reset)
      UnixResetSignal(sig);
   else
      UnixSignal(sig, SigHandler);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset signals handlers to previous behaviour.

void TUnixSigHandling::ResetSignals()
{
   UnixResetSignals();
}

////////////////////////////////////////////////////////////////////////////////
/// If ignore is true ignore the specified signal, else restore previous
/// behaviour.

void TUnixSigHandling::IgnoreSignal(ESignals sig, Bool_t ignore)
{
   UnixIgnoreSignal(sig, ignore);
}

////////////////////////////////////////////////////////////////////////////////
/// When the argument is true the SIGALRM signal handler is set so that
/// interrupted syscalls will not be restarted by the kernel. This is
/// typically used in case one wants to put a timeout on an I/O operation.
/// By default interrupted syscalls will always be restarted (for all
/// signals). This can be controlled for each a-synchronous TTimer via
/// the method TTimer::SetInterruptSyscalls().

void TUnixSigHandling::SigAlarmInterruptsSyscalls(Bool_t set)
{
   UnixSigAlarmInterruptsSyscalls(set);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if there is any signal trapping.
Bool_t TUnixSigHandling::HaveTrappedSignal(Bool_t pendingOnly)
{
   if (fSigcnt > 0 && fSignalHandler->GetSize() > 0)
      if (CheckSignals(kTRUE))
         if (!pendingOnly) return 1;
   fSigcnt = 0;
   fSignals->Zero();
   return 0;
}

//---- handling of system events -----------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Check if some signals were raised and call their Notify() member.

Bool_t TUnixSigHandling::CheckSignals(Bool_t sync)
{
   TSignalHandler *sh;
   Int_t sigdone = -1;
   {
      TOrdCollectionIter it((TOrdCollection*)fSignalHandler);

      while ((sh = (TSignalHandler*)it.Next())) {
         if (sync == sh->IsSync()) {
            ESignals sig = sh->GetSignal();
            if ((fSignals->IsSet(sig) && sigdone == -1) || sigdone == sig) {
               if (sigdone == -1) {
                  fSignals->Clr(sig);
                  sigdone = sig;
                  fSigcnt--;
               }
               if (sh->IsActive())
                  sh->Notify();
            }
         }
      }
   }
   if (sigdone != -1)
      return kTRUE;

   return kFALSE;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Static Protected Unix Interface functions.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//---- signals -----------------------------------------------------------------

static struct Signalmap_t {
   int               fCode;
   SigHandler_t      fHandler;
   struct sigaction *fOldHandler;
   const char       *fSigName;
} gSignalMap[kMAXSIGNALS] = {       // the order of the signals should be identical
   { SIGBUS,   0, 0, "bus error" }, // to the one in TSysEvtHandler.h
   { SIGSEGV,  0, 0, "segmentation violation" },
   { SIGSYS,   0, 0, "bad argument to system call" },
   { SIGPIPE,  0, 0, "write on a pipe with no one to read it" },
   { SIGILL,   0, 0, "illegal instruction" },
   { SIGQUIT,  0, 0, "quit" },
   { SIGINT,   0, 0, "interrupt" },
   { SIGWINCH, 0, 0, "window size change" },
   { SIGALRM,  0, 0, "alarm clock" },
   { SIGCHLD,  0, 0, "death of a child" },
   { SIGURG,   0, 0, "urgent data arrived on an I/O channel" },
   { SIGFPE,   0, 0, "floating point exception" },
   { SIGTERM,  0, 0, "termination signal" },
   { SIGUSR1,  0, 0, "user-defined signal 1" },
   { SIGUSR2,  0, 0, "user-defined signal 2" }
};


////////////////////////////////////////////////////////////////////////////////
/// Call the signal handler associated with the signal.

static void sighandler(int sig)
{
   for (int i= 0; i < kMAXSIGNALS; i++) {
      if (gSignalMap[i].fCode == sig) {
         (*gSignalMap[i].fHandler)((ESignals)i);
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle and dispatch signals.

void TUnixSigHandling::DispatchSignals(ESignals sig)
{
   switch (sig) {
   case kSigAlarm:
//      gSystem->DispatchTimers(kFALSE);
      break;
   case kSigChild:
//      gSystem->CheckChilds();
      break;
   case kSigBus:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: bus error");
      break;
   case kSigSegmentationViolation:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: segmentation violation error");
      break;
   case kSigIllegalInstruction:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: illegal instruction error");
      break;
   case kSigFloatingException:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: floating exception error");
      break;
   case kSigSystem:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: system error");
      break;
   case kSigPipe:
      SignalSafeErrWrite("\n\nA fatal system signal has occurred: pipe error");
      break;
   case kSigWindowChanged:
      Gl_windowchanged();
      break;
   default:
      fSignals->Set(sig);
      fSigcnt++;
      break;
   }

   if ((sig == kSigIllegalInstruction) || (sig == kSigSegmentationViolation) || (sig == kSigBus) || (sig == kSigFloatingException))
   {
      StackTraceTriggerThread();
      signal(sig, SIG_DFL);
      raise(sig);
      if (gApplication)
         //sig is ESignal, should it be mapped to the correct signal number?
         gApplication->HandleException(sig);
      else
         //map to the real signal code + set the
         //high order bit to indicate a signal (?)
         gSystem->Exit(gSignalMap[sig].fCode + 0x80, 0);
   }

   // check a-synchronous signals
   if (fSigcnt > 0 && fSignalHandler->GetSize() > 0)
      CheckSignals(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Set a signal handler for a signal.

void TUnixSigHandling::UnixSignal(ESignals sig, SigHandler_t handler)
{
   if (gEnv && !gEnv->GetValue("Root.ErrorHandlers", 1))
      return;

   if (gSignalMap[sig].fHandler != handler) {
      struct sigaction sigact;

      gSignalMap[sig].fHandler    = handler;
      gSignalMap[sig].fOldHandler = new struct sigaction();

#if defined(R__SUN)
      sigact.sa_handler = (void (*)())sighandler;
#elif defined(R__SOLARIS)
      sigact.sa_handler = sighandler;
#elif defined(R__LYNXOS)
#  if (__GNUG__>=3)
      sigact.sa_handler = sighandler;
#  else
      sigact.sa_handler = (void (*)(...))sighandler;
#  endif
#else
      sigact.sa_handler = sighandler;
#endif
      sigemptyset(&sigact.sa_mask);
      sigact.sa_flags = 0;
#if defined(SA_RESTART)
      sigact.sa_flags |= SA_RESTART;
#endif
      if (sigaction(gSignalMap[sig].fCode, &sigact,
                    gSignalMap[sig].fOldHandler) < 0)
         ::SysError("TUnixSigHandling::UnixSignal", "sigaction");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// If ignore is true ignore the specified signal, else restore previous
/// behaviour.

void TUnixSigHandling::UnixIgnoreSignal(ESignals sig, Bool_t ignore)
{
   TTHREAD_TLS(Bool_t) ignoreSig[kMAXSIGNALS] = { kFALSE };
   TTHREAD_TLS_ARRAY(struct sigaction,kMAXSIGNALS,oldsigact);

   if (ignore != ignoreSig[sig]) {
      ignoreSig[sig] = ignore;
      if (ignore) {
         struct sigaction sigact;
#if defined(R__SUN)
         sigact.sa_handler = (void (*)())SIG_IGN;
#elif defined(R__SOLARIS)
         sigact.sa_handler = (void (*)(int))SIG_IGN;
#else
         sigact.sa_handler = SIG_IGN;
#endif
         sigemptyset(&sigact.sa_mask);
         sigact.sa_flags = 0;
         if (sigaction(gSignalMap[sig].fCode, &sigact, &oldsigact[sig]) < 0)
            ::SysError("TUnixSigHandling::UnixIgnoreSignal", "sigaction");
      } else {
         if (sigaction(gSignalMap[sig].fCode, &oldsigact[sig], 0) < 0)
            ::SysError("TUnixSigHandling::UnixIgnoreSignal", "sigaction");
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// When the argument is true the SIGALRM signal handler is set so that
/// interrupted syscalls will not be restarted by the kernel. This is
/// typically used in case one wants to put a timeout on an I/O operation.
/// By default interrupted syscalls will always be restarted (for all
/// signals). This can be controlled for each a-synchronous TTimer via
/// the method TTimer::SetInterruptSyscalls().

void TUnixSigHandling::UnixSigAlarmInterruptsSyscalls(Bool_t set)
{
   if (gSignalMap[kSigAlarm].fHandler) {
      struct sigaction sigact;
#if defined(R__SUN)
      sigact.sa_handler = (void (*)())sighandler;
#elif defined(R__SOLARIS)
      sigact.sa_handler = sighandler;
#elif defined(R__LYNXOS)
#  if (__GNUG__>=3)
      sigact.sa_handler = sighandler;
#  else
      sigact.sa_handler = (void (*)(...))sighandler;
#  endif
#else
      sigact.sa_handler = sighandler;
#endif
      sigemptyset(&sigact.sa_mask);
      sigact.sa_flags = 0;
      if (set) {
#if defined(SA_INTERRUPT)       // SunOS
         sigact.sa_flags |= SA_INTERRUPT;
#endif
      } else {
#if defined(SA_RESTART)
         sigact.sa_flags |= SA_RESTART;
#endif
      }
      if (sigaction(gSignalMap[kSigAlarm].fCode, &sigact, 0) < 0)
         ::SysError("TUnixSigHandling::UnixSigAlarmInterruptsSyscalls", "sigaction");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the signal name associated with a signal.

const char *TUnixSigHandling::UnixSigname(ESignals sig)
{
   return gSignalMap[sig].fSigName;
}

////////////////////////////////////////////////////////////////////////////////
/// Restore old signal handler for specified signal.

void TUnixSigHandling::UnixResetSignal(ESignals sig)
{
   if (gSignalMap[sig].fOldHandler) {
      // restore old signal handler
      if (sigaction(gSignalMap[sig].fCode, gSignalMap[sig].fOldHandler, 0) < 0)
         ::SysError("TUnixSigHandling::UnixSignal", "sigaction");
      delete gSignalMap[sig].fOldHandler;
      gSignalMap[sig].fOldHandler = 0;
      gSignalMap[sig].fHandler    = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Restore old signal handlers.

void TUnixSigHandling::UnixResetSignals()
{
   for (int sig = 0; sig < kMAXSIGNALS; sig++)
      UnixResetSignal((ESignals)sig);
}

////////////////////////////////////////////////////////////////////////////////
/// Set signal handlers to default signal handlers

void TUnixSigHandling::UnixSetDefaultSignals()
{
   signal(SIGILL, SIG_DFL);
   signal(SIGSEGV, SIG_DFL);
   signal(SIGBUS, SIG_DFL);
}

///////////////////////////////////////////////////////////////////////////////
/// Stack Trace

void TUnixSigHandling::StackTrace()
{
   gSystem->StackTrace();
}

//////////////////////////////////////////////////////////////////////////////
/// Initialize StackTrace helper structures

void TUnixSigHandling::StackTraceHelperInit()
{
   if(snprintf(gStackTraceHelper.fPidNum, kStringLength-1, "%d", gSystem->GetPid()) >= kStringLength) {
      SignalSafeErrWrite("Unable to pre-allocate process id information");
      return;
   }

   close(gStackTraceHelper.fChildToParent[0]);
   close(gStackTraceHelper.fChildToParent[1]);
   gStackTraceHelper.fChildToParent[0] = -1; gStackTraceHelper.fChildToParent[1] = -1;
   close(gStackTraceHelper.fParentToChild[0]);
   close(gStackTraceHelper.fParentToChild[1]);
   gStackTraceHelper.fParentToChild[0] = -1; gStackTraceHelper.fParentToChild[1] = -1;

   if (-1 == pipe2(gStackTraceHelper.fChildToParent, O_CLOEXEC)) {
      fprintf(stdout, "pipe gStackTraceHelper.fChildToParent failed\n");
      return;
   }
   if (-1 == pipe2(gStackTraceHelper.fParentToChild, O_CLOEXEC)){
      close(gStackTraceHelper.fChildToParent[0]); close(gStackTraceHelper.fChildToParent[1]);
      gStackTraceHelper.fChildToParent[0] = -1; gStackTraceHelper.fChildToParent[1] = -1;
      fprintf(stdout, "pipe parentToChild failed\n");
      return;
   }

   gStackTraceHelper.fHelperThread.reset(new std::thread(StackTraceMonitorThread));
   gStackTraceHelper.fHelperThread->detach();
}

//////////////////////////////////////////////////////////////////////////////
/// StackTrace helper thread to monitor the signal interrupts

void TUnixSigHandling::StackTraceMonitorThread()
{
   int toParent = gStackTraceHelper.fChildToParent[1];
   int fromParent = gStackTraceHelper.fParentToChild[0];
   char buf[2]; buf[1] = '\0';
   while(true) {
      int result = SignalSafeRead(fromParent, buf, 1, 5*60);
      if (result < 0) {
         UnixSetDefaultSignals();
         close(toParent);
         SignalSafeErrWrite("\n\nTraceback helper thread failed to read from parent: ");
         SignalSafeErrWrite(strerror(-result));
         SignalSafeErrWrite("\n");
         gSystem->Exit(1, kFALSE);
      }
      if (buf[0] == '1') {
          UnixSetDefaultSignals();
          StackTraceForkThread();
          SignalSafeWrite(toParent, buf);
      } else if (buf[0] == '2') {
          close(toParent);
          close(fromParent);
          toParent = gStackTraceHelper.fChildToParent[1];
          fromParent = gStackTraceHelper.fParentToChild[0];
      } else if (buf[0] == '3') {
          break;
      } else {
          UnixSetDefaultSignals();
          close(toParent);
          SignalSafeErrWrite("\n\nTraceback helper thread got unknown command from parent: ");
          SignalSafeErrWrite(buf);
          SignalSafeErrWrite("\n");
          gSystem->Exit(1, kFALSE);
      }
   }
   return;
}

//////////////////////////////////////////////////////////////////////////////
/// One of StackTrace helper threads.
/// This thread is to trigger the monitor thread by pipe.

void TUnixSigHandling::StackTraceTriggerThread()
{
   int result = SignalSafeWrite(gStackTraceHelper.fParentToChild[1], "1"); 
   if (result < 0) {
      SignalSafeErrWrite("\n\nAttempt to request stacktrace failed: ");
      SignalSafeErrWrite(strerror(-result));
      SignalSafeErrWrite("\n");
      return;
    }
    char buf[2]; buf[1] = '\0';
    if ((result = SignalSafeRead(gStackTraceHelper.fChildToParent[0], buf, 1)) < 0) {
       SignalSafeErrWrite("\n\nWaiting for stacktrace completion failed: ");
       SignalSafeErrWrite(strerror(-result));
       SignalSafeErrWrite("\n");
       return;
    }
}

//////////////////////////////////////////////////////////////////////////////
/// One of StackTrace helper threads.
/// This thread is to fork a new thread in order to print out StackTrace info.

void TUnixSigHandling::StackTraceForkThread()
{
   char childStack[4*1024];
   char *childStackPtr = childStack + 4*1024;
   int pid =
#ifdef __linux__
      clone(StackTraceExecScript, childStackPtr, CLONE_VM|CLONE_FS|SIGCHLD, nullptr);
#else
      fork();
   if (childStackPtr) {} // Suppress 'unused variable' warning on non-Linux
   if (pid == 0) { StackTraceExecScript(nullptr); gSystem->Exit(0, kFALSE); }
#endif
   if (pid == -1) {
      SignalSafeErrWrite("(Attempt to perform stack dump failed.)\n");
   } else {
      int status;
      if (waitpid(pid, &status, 0) == -1) {
         SignalSafeErrWrite("(Failed to wait on stack dump output.)\n");
         gSystem->Exit(1, kFALSE);
      } else {
         gSystem->Exit(0, kFALSE);
      }
   }
}

//////////////////////////////////////////////////////////////////////////////
/// The new thread in StackTraceForkThread() is forked to run this function 
/// to print out stack trace.

int TUnixSigHandling::StackTraceExecScript(void * /*arg*/)
{
   char *const *argv = GetStackArgv();
#ifdef __linux__
   syscall(SYS_execve, "/bin/sh", argv, __environ);
#else
   execv("/bin/sh", argv);
#endif
   gSystem->Exit(0, kFALSE);
   return 0;
}
