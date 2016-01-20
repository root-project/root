// @(#)root/unix:$Id: 887c618d89c4ed436e4034fc133f468fecad651b $
// Author: Fons Rademakers   15/09/95

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
#include "TOrdCollection.h"
#include "TRegexp.h"
#include "TPRegexp.h"
#include "TException.h"
#include "Demangle.h"
#include "TEnv.h"
#include "TSocket.h"
#include "Getline.h"
#include "TInterpreter.h"
#include "TApplication.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TVirtualMutex.h"
#include "TObjArray.h"
#include <map>
#include <algorithm>
#include <atomic>

//#define G__OLDEXPAND

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#if defined(R__SUN) || defined(R__AIX) || \
    defined(R__LINUX) || defined(R__SOLARIS) || \
    defined(R__FBSD) || defined(R__OBSD) || \
    defined(R__MACOSX) || defined(R__HURD)
#define HAS_DIRENT
#endif
#ifdef HAS_DIRENT
#   include <dirent.h>
#else
#   include <sys/dir.h>
#endif
#if defined(ULTRIX) || defined(R__SUN)
#   include <sgtty.h>
#endif
#if defined(R__AIX) || defined(R__LINUX) || \
    defined(R__FBSD) || defined(R__OBSD) || \
    defined(R__LYNXOS) || defined(R__MACOSX) || defined(R__HURD)
#   include <sys/ioctl.h>
#endif
#if defined(R__AIX) || defined(R__SOLARIS)
#   include <sys/select.h>
#endif
#if defined(R__LINUX) || defined(R__HURD)
#   ifndef SIGSYS
#      define SIGSYS  SIGUNUSED       // SIGSYS does not exist in linux ??
#   endif
#endif
#if defined(R__MACOSX)
#   include <mach-o/dyld.h>
#   include <sys/mount.h>
   extern "C" int statfs(const char *file, struct statfs *buffer);
#elif defined(R__LINUX) || defined(R__HURD)
#   include <sys/vfs.h>
#elif defined(R__FBSD) || defined(R__OBSD)
#   include <sys/param.h>
#   include <sys/mount.h>
#else
#   include <sys/statfs.h>
#endif

#include <utime.h>
#include <syslog.h>
#include <sys/stat.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/param.h>
#include <pwd.h>
#include <grp.h>
#include <errno.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#if defined(R__AIX)
#   define _XOPEN_EXTENDED_SOURCE
#   include <arpa/inet.h>
#   undef _XOPEN_EXTENDED_SOURCE
#   if !defined(_AIX41) && !defined(_AIX43)
    // AIX 3.2 doesn't have it
#   define HASNOT_INETATON
#   endif
#else
#   include <arpa/inet.h>
#endif
#include <sys/un.h>
#include <netdb.h>
#include <fcntl.h>
#if defined(R__SOLARIS)
#   include <sys/systeminfo.h>
#   include <sys/filio.h>
#   include <sys/sockio.h>
#   define HASNOT_INETATON
#   ifndef INADDR_NONE
#      define INADDR_NONE (UInt_t)-1
#   endif
#endif

#if defined(R__SOLARIS)
#   define HAVE_UTMPX_H
#   define UTMP_NO_ADDR
#endif

#if defined(MAC_OS_X_VERSION_10_5)
#   define HAVE_UTMPX_H
#   define UTMP_NO_ADDR
#endif

#if defined(R__FBSD)
#   include <sys/param.h>
#   if __FreeBSD_version >= 900007
#      define HAVE_UTMPX_H
#   endif
#endif

#if defined(R__AIX) || defined(R__FBSD) || \
    defined(R__OBSD) || defined(R__LYNXOS) || \
    (defined(R__MACOSX) && !defined(MAC_OS_X_VERSION_10_5))
#   define UTMP_NO_ADDR
#endif

#if (defined(R__AIX) && !defined(_AIX43)) || \
    (defined(R__SUNGCC3) && !defined(__arch64__))
#   define USE_SIZE_T
#elif defined(R__GLIBC) || defined(R__FBSD) || \
      (defined(R__SUNGCC3) && defined(__arch64__)) || \
      defined(R__OBSD) || defined(MAC_OS_X_VERSION_10_4) || \
      (defined(R__AIX) && defined(_AIX43)) || \
      (defined(R__SOLARIS) && defined(_SOCKLEN_T))
#   define USE_SOCKLEN_T
#endif

#if defined(R__LYNXOS)
extern "C" {
   extern int putenv(const char *);
   extern int inet_aton(const char *, struct in_addr *);
};
#endif

#ifdef HAVE_UTMPX_H
#include <utmpx.h>
#define STRUCT_UTMP struct utmpx
#else
#include <utmp.h>
#define STRUCT_UTMP struct utmp
#endif
#if !defined(UTMP_FILE) && defined(_PATH_UTMP)      // 4.4BSD
#define UTMP_FILE _PATH_UTMP
#endif
#if defined(UTMPX_FILE)                             // Solaris, SysVr4
#undef  UTMP_FILE
#define UTMP_FILE UTMPX_FILE
#endif
#ifndef UTMP_FILE
#define UTMP_FILE "/etc/utmp"
#endif

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

//------------------- Unix TFdSet ----------------------------------------------
#ifndef HOWMANY
#   define HOWMANY(x, y)   (((x)+((y)-1))/(y))
#endif

const Int_t kNFDBITS = (sizeof(Long_t) * 8);  // 8 bits per byte
#ifdef FD_SETSIZE
const Int_t kFDSETSIZE = FD_SETSIZE;          // Linux = 1024 file descriptors
#else
const Int_t kFDSETSIZE = 256;                 // upto 256 file descriptors
#endif


class TFdSet {
private:
   ULong_t fds_bits[HOWMANY(kFDSETSIZE, kNFDBITS)];
public:
   TFdSet() { memset(fds_bits, 0, sizeof(fds_bits)); }
   TFdSet(const TFdSet &org) { memcpy(fds_bits, org.fds_bits, sizeof(org.fds_bits)); }
   TFdSet &operator=(const TFdSet &rhs) { if (this != &rhs) { memcpy(fds_bits, rhs.fds_bits, sizeof(rhs.fds_bits));} return *this; }
   void   Zero() { memset(fds_bits, 0, sizeof(fds_bits)); }
   void   Set(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         fds_bits[n/kNFDBITS] |= (1UL << (n % kNFDBITS));
      } else {
         ::Fatal("TFdSet::Set","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
      }
   }
   void   Clr(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         fds_bits[n/kNFDBITS] &= ~(1UL << (n % kNFDBITS));
      } else {
         ::Fatal("TFdSet::Clr","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
      }
   }
   Int_t  IsSet(Int_t n)
   {
      if (n >= 0 && n < kFDSETSIZE) {
         return (fds_bits[n/kNFDBITS] & (1UL << (n % kNFDBITS))) != 0;
      } else {
         ::Fatal("TFdSet::IsSet","fd (%d) out of range [0..%d]", n, kFDSETSIZE-1);
         return 0;
      }
   }
   ULong_t *GetBits() { return (ULong_t *)fds_bits; }
};

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
      ((TUnixSigHandling*)gSigHandling)->DispatchSignals(sig);
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
   gRootDir = Getenv("ROOTSYS");
   if (gRootDir == 0)
      gRootDir= "/usr/local/root";
#else
   gRootDir = ROOTPREFIX;
#endif

}

//---- Misc --------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Get environment variable.

const char *TUnixSigHandling::Getenv(const char *name)
{
   return ::getenv(name);
}

////////////////////////////////////////////////////////////////////////////////
///// Exit the application.

void TUnixSigHandling::Exit(int code, Bool_t mode)
{
   // Insures that the files and sockets are closed before any library is unloaded
   // and before emptying CINT.
   if (gROOT) {
      gROOT->EndOfProcessCleanups();
   } else if (gInterpreter) {
      gInterpreter->ResetGlobals();
   }

   if (mode)
      ::exit(code);
   else
      ::_exit(code);
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
//      DispatchTimers(kFALSE);
      break;
   case kSigChild:
//      CheckChilds();
      break;
   case kSigBus:
   case kSigSegmentationViolation:
   case kSigIllegalInstruction:
   case kSigFloatingException:
      Break("TUnixSigHandling::DispatchSignals", "%s", UnixSigname(sig));
      StackTrace();
      if (gApplication)
         //sig is ESignal, should it be mapped to the correct signal number?
         gApplication->HandleException(sig);
      else
         //map to the real signal code + set the
         //high order bit to indicate a signal (?)
         Exit(gSignalMap[sig].fCode + 0x80);
      break;
   case kSigSystem:
   case kSigPipe:
      Break("TUnixSigHandling::DispatchSignals", "%s", UnixSigname(sig));
      break;
   case kSigWindowChanged:
      Gl_windowchanged();
      break;
   default:
      fSignals->Set(sig);
      fSigcnt++;
      break;
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

///////////////////////////////////////////////////////////////////////////////
/// Stack Trace

void TUnixSigHandling::StackTrace()
{
   gSystem->StackTrace();
}
