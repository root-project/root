// @(#)root/base:$Name:  $:$Id: TSystem.h,v 1.11 2001/04/23 08:04:48 brun Exp $
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSystem
#define ROOT_TSystem


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSystem                                                              //
//                                                                      //
// Abstract base class defining a generic interface to the underlying   //
// Operating System.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <stdio.h>
#endif

#ifndef NOCINT
#include "G__ci.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TInetAddress
#include "TInetAddress.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif

class TSeqCollection;


enum EAccessMode {
   kFileExists        = 0,
   kExecutePermission = 1,
   kWritePermission   = 2,
   kReadPermission    = 4
};

enum ELogOption {
   kLogPid            = 0x01,
   kLogCons           = 0x02
};

enum ELogLevel {
   kLogEmerg          = 0,
   kLogAlert          = 1,
   kLogCrit           = 2,
   kLogErr            = 3,
   kLogWarning        = 4,
   kLogNotice         = 5,
   kLogInfo           = 6,
   kLogDebug          = 7
};

enum ELogFacility {
   kLogLocal0,
   kLogLocal1,
   kLogLocal2,
   kLogLocal3,
   kLogLocal4,
   kLogLocal5,
   kLogLocal6,
   kLogLocal7
};

enum ESysConstants {
  kMAXSIGNALS       = 15,
  kMAXPATHLEN       = 1024,
  kBUFFERSIZE       = 8192,
  kItimerResolution = 10      // interval-timer resolution in ms
};

typedef void* Func_t;

R__EXTERN const char  *gRootDir;
R__EXTERN const char  *gProgName;
R__EXTERN const char  *gProgPath;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Asynchronous timer used for processing pending GUI and timer events  //
// every delay ms. Call in a tight computing loop                       //
// TProcessEventTimer::ProcessEvent(). If the timer did timeout this    //
// call will process the pending events and return kTRUE if the         //
// TROOT::IsInterrupted() flag is set (can be done by hitting key in    //
// canvas or selecting canvas menu item View/Interrupt.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TProcessEventTimer : public TTimer {
public:
   TProcessEventTimer(Long_t delay);
   Bool_t Notify() { return kTRUE; }
   Bool_t ProcessEvents();
   ClassDef(TProcessEventTimer,0)  // Process pending events at fixed time intervals
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFdSet                                                               //
//                                                                      //
// Wrapper class around the fd_set bit mask macros used by select().    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef HOWMANY
#   define HOWMANY(x, y)   (((x)+((y)-1))/(y))
#endif

const int kNFDBITS = (sizeof(Long_t) * 8);      // 8 bits per byte

class TFdSet {
private:
   Long_t fds_bits[HOWMANY(256, kNFDBITS)];     // upto 255 file descriptors
public:
   TFdSet() { memset(fds_bits, 0, sizeof(fds_bits)); }
   void Zero() { memset(fds_bits, 0, sizeof(fds_bits)); }
   void Set(int n) { fds_bits[n/kNFDBITS] |= (1 << (n % kNFDBITS)); }
   void Clr(int n) { fds_bits[n/kNFDBITS] &= ~(1 << (n % kNFDBITS)); }
   int  IsSet(int n) { return fds_bits[n/kNFDBITS] & (1 << (n % kNFDBITS)); }
   int *GetBits() { return (int *)fds_bits; }
};


class TSystem : public TNamed {

protected:
   TFdSet           fReadmask;         //!Files that should be checked for read events
   TFdSet           fWritemask;        //!Files that should be checked for write events
   TFdSet           fReadready;        //!Files with reads waiting
   TFdSet           fWriteready;       //!Files with writes waiting
   TFdSet           fSignals;          //!Signals that were trapped
   Int_t            fNfd;              //Number of fd's in masks
   Int_t            fMaxrfd;           //Largest fd in read mask
   Int_t            fMaxwfd;           //Largest fd in write mask
   Int_t            fSigcnt;           //Number of pending signals
   TString          fWdpath;           //Working directory
   TString          fHostname;         //Hostname
   Bool_t           fInsideNotify;     //Used by DispatchTimers()

   Bool_t           fInControl;        //True if in eventloop
   Bool_t           fDone;             //True if eventloop should be finished
   Int_t            fLevel;            //Level of nested eventloops
   TString          fLastErrorString;  //Last system error message

   TSeqCollection  *fTimers;           //List of timers
   TSeqCollection  *fSignalHandler;    //List of signal handlers
   TSeqCollection  *fFileHandler;      //List of file handlers
   TSeqCollection  *fOnExitList;       //List of items to be cleaned-up on exit

   TString          fListLibs;         //List shared libraries. Cache used by GetLibraries

   TString          fListPaths;        //List of all include (fIncludePath + interpreter include path). Cache used by GetIncludePath
   TString          fIncludePath;      //Used to expand $IncludePath in the directives given to SetMakeSharedLib and SetMakeExe
   TString          fLinkedLibs;       //Used to expand $LinkedLibs in the directives given to SetMakeSharedLib and SetMakeExe
   TString          fSoExt;            //Extension of shared library (.so, .sl, .a, .dll, etc.)
   TString          fObjExt;           //Extension of object files (.o, .obj, etc.)
   TString          fMakeSharedLib;    //Directive used to build a shared library
   TString          fMakeExe;          //Directive used to build an executable
   TSeqCollection  *fCompiled;         //List of shared libs from compiled macros to be deleted

   virtual const char    *ExpandFileName(const char *fname);

public:
   TSystem(const char *name = "Generic", const char *title = "Generic System");
   virtual ~TSystem();

   //---- Misc
   virtual Bool_t          Init();
   virtual void            SetProgname(const char *name);
   virtual void            SetDisplay();
   void                    SetErrorStr(const char *errstr) { fLastErrorString = errstr; }
   const char             *GetErrorStr() { return (const char *)fLastErrorString; }
   virtual const char     *GetError();
   void                    RemoveOnExit(TObject *obj);
   virtual const char     *HostName();

   static Int_t            GetErrno();
   static void             ResetErrno();

   //---- EventLoop
   virtual void            Run();
   virtual Bool_t          ProcessEvents();
   virtual void            DispatchOneEvent(Bool_t pendingOnly = kFALSE);
   void                    ExitLoop();
   Bool_t                  InControl() const { return fInControl; }
   void                    InnerLoop();

   //---- Handling of system events
   virtual void            AddSignalHandler(TSignalHandler *sh);
   virtual TSignalHandler *RemoveSignalHandler(TSignalHandler *sh);
   virtual void            AddFileHandler(TFileHandler *fh);
   virtual TFileHandler   *RemoveFileHandler(TFileHandler *fh);
   virtual void            IgnoreInterrupt(Bool_t ignore = kTRUE);

   //---- Time & Date
   virtual TTime           Now();
   virtual void            AddTimer(TTimer *t);
   virtual TTimer         *RemoveTimer(TTimer *t);
   virtual void            ResetTimer(TTimer *) { }
   virtual Long_t          NextTimeOut(Bool_t mode);
   virtual void            Sleep(UInt_t milliSec);

   //---- Processes
   virtual Int_t           Exec(const char *shellcmd);
   virtual FILE           *OpenPipe(const char *command, const char *mode);
   virtual int             ClosePipe(FILE *pipe);
   virtual void            Exit(int code, Bool_t mode = kTRUE);
   virtual void            Abort(int code = 0);
   virtual int             GetPid();
   virtual void            StackTrace();

   //---- Directories
   virtual int             MakeDirectory(const char *name);
   virtual void           *OpenDirectory(const char *name);
   virtual void            FreeDirectory(void *dirp);
   virtual const char     *GetDirEntry(void *dirp);
   virtual Bool_t          ChangeDirectory(const char *path);
   virtual const char     *WorkingDirectory();
   virtual const char     *HomeDirectory(const Char_t *userName = 0);
   int                     mkdir(const char *name) { return MakeDirectory(name); }
   Bool_t                  cd(const char *path) { return ChangeDirectory(path); }
   const char             *pwd() { return WorkingDirectory(); }

   //---- Paths & Files
   virtual const char     *BaseName(const char *pathname);
   virtual const char     *DirName(const char *pathname);
   virtual char           *ConcatFileName(const char *dir, const char *name);
   virtual Bool_t          IsAbsoluteFileName(const char *dir);
   virtual Bool_t          ExpandPathName(TString &path); // expand the metacharacters in buf as in the shell
   virtual char           *ExpandPathName(const char *path);
   virtual Bool_t          AccessPathName(const char *path, EAccessMode mode = kFileExists);
   virtual void            Rename(const char *from, const char *to);
   virtual int             Link(const char *from, const char *to);
   virtual int             Symlink(const char *from, const char *to);
   virtual int             Unlink(const char *name);
   virtual int             GetPathInfo(const char *path, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   virtual int             GetFsInfo(const char *path, Long_t *id, Long_t *bsize, Long_t *blocks, Long_t *bfree);
   virtual int             Umask(Int_t mask);
   virtual const char     *UnixPathName(const char *unixpathname);
   virtual char           *Which(const char *search, const char *file, EAccessMode mode = kFileExists);

   //---- Environment Manipulation
   virtual void            Setenv(const char *name, const char *value); // set environment variable name to value
   virtual void            Unsetenv(const char *name);  // remove environment variable
   virtual const char     *Getenv(const char *env);

   //---- System Logging
   virtual void            Openlog(const char *name, Int_t options, ELogFacility facility);
   virtual void            Syslog(ELogLevel level, const char *mess);
   virtual void            Closelog();

   //---- Dynamic Loading
   virtual char           *DynamicPathName(const char *lib, Bool_t quiet = kFALSE);
   virtual Func_t          DynFindSymbol(const char *module, const char *entry);
   virtual int             Load(const char *module, const char *entry = "", Bool_t system = kFALSE);
   virtual void            Unload(const char *module);
   virtual void            ListSymbols(const char *module, const char *re = "");
   virtual void            ListLibraries(const char *regexp = "");
   virtual const char     *GetLibraries(const char *regexp = "", const char* option = "");

   //---- RPC
   virtual TInetAddress    GetHostByName(const char *server);
   virtual TInetAddress    GetPeerName(int sock);
   virtual TInetAddress    GetSockName(int sock);
   virtual int             GetServiceByName(const char *service);
   virtual char           *GetServiceByPort(int port);
   virtual int             OpenConnection(const char *server, int port, int tcpwindowsize = -1);
   virtual int             AnnounceTcpService(int port, Bool_t reuse, int backlog, int tcpwindowsize = -1);
   virtual int             AnnounceUnixService(int port, int backlog);
   virtual int             AcceptConnection(int sock);
   virtual void            CloseConnection(int sock, Bool_t force = kFALSE);
   virtual int             RecvRaw(int sock, void *buffer, int length, int flag);
   virtual int             SendRaw(int sock, const void *buffer, int length, int flag);
   virtual int             RecvBuf(int sock, void *buffer, int length);
   virtual int             SendBuf(int sock, const void *buffer, int length);
   virtual int             SetSockOpt(int sock, int kind, int val);
   virtual int             GetSockOpt(int sock, int kind, int *val);

   //---- ACLiC (Automatic Compiler of Shared Library for CINT)
   virtual int             CompileMacro(const char *filename, Option_t *opt="", const char* library_name = "");
   virtual const char     *GetMakeSharedLib() const;
   virtual const char     *GetMakeExe() const;
   virtual const char     *GetIncludePath();
   virtual const char     *GetLinkedLibs() const;
   virtual const char     *GetSoExt() const;
   virtual const char     *GetObjExt() const;
   virtual void            SetMakeSharedLib(const char *directives);
   virtual void            SetMakeExe(const char *directives);
   virtual void            SetIncludePath(const char *IncludePath);
   virtual void            SetLinkedLibs(const char *LinkedLibs);
   virtual void            SetSoExt(const char *SoExt);
   virtual void            SetObjExt(const char *ObjExt);
   virtual void            CleanCompiledMacros();

   ClassDef(TSystem,0)  //ABC defining a generic interface to the OS
};

R__EXTERN TSystem *gSystem;
R__EXTERN TFileHandler *gXDisplay;  // Display server (X11) input event handler

#endif
