// @(#)root/unix:$Id$
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TUnixSystem
#define ROOT_TUnixSystem


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUnixSystem                                                          //
//                                                                      //
// Class providing an interface to the UNIX Operating System.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystem.h"
#include "TSysEvtHandler.h"
#include "TTimer.h"

typedef void (*SigHandler_t)(ESignals);


class TUnixSystem : public TSystem {

private:
   void FillWithCwd(char *cwd) const;

protected:
   const char    *FindDynamicLibrary(TString &lib, Bool_t quiet = kFALSE);
   const char    *GetLinkedLibraries();

   // static functions providing semi-low level interface to raw Unix
   static int          UnixMakedir(const char *name);
   static void        *UnixOpendir(const char *name);
   static const char  *UnixGetdirentry(void *dir);
   static const char  *UnixHomedirectory(const char *user = 0);
   static const char  *UnixHomedirectory(const char *user, char *path, char *mydir);
   static Long64_t     UnixNow();
   static int          UnixWaitchild();
   static int          UnixSetitimer(Long_t ms);
   static int          UnixSelect(Int_t nfds, TFdSet *readready, TFdSet *writeready,
                                  Long_t timeout);
   static void         UnixSignal(ESignals sig, SigHandler_t h);
   static const char  *UnixSigname(ESignals sig);
   static void         UnixSigAlarmInterruptsSyscalls(Bool_t set);
   static void         UnixResetSignal(ESignals sig);
   static void         UnixResetSignals();
   static void         UnixIgnoreSignal(ESignals sig, Bool_t ignore);
   static int          UnixFilestat(const char *path, FileStat_t &buf);
   static int          UnixFSstat(const char *path, Long_t *id, Long_t *bsize,
                                  Long_t *blocks, Long_t *bfree);
   static int          UnixTcpConnect(const char *hostname, int port, int tcpwindowsize);
   static int          UnixUdpConnect(const char *hostname, int port);
   static int          UnixUnixConnect(int port);
   static int          UnixUnixConnect(const char *path);
   static int          UnixTcpService(int port, Bool_t reuse, int backlog,
                                      int tcpwindowsize);
   static int          UnixUdpService(int port, int backlog);
   static int          UnixUnixService(int port, int backlog);
   static int          UnixUnixService(const char *sockpath, int backlog);
   static int          UnixRecv(int sock, void *buf, int len, int flag);
   static int          UnixSend(int sock, const void *buf, int len, int flag);

public:
   TUnixSystem();
   virtual ~TUnixSystem();

   //---- Misc -------------------------------------------------
   Bool_t            Init();
   void              SetProgname(const char *name);
   void              SetDisplay();
   const char       *GetError();
   const char       *HostName();

   //---- EventLoop --------------------------------------------
   void              DispatchOneEvent(Bool_t pendingOnly = kFALSE);
   Int_t             Select(TList *active, Long_t timeout);
   Int_t             Select(TFileHandler *fh, Long_t timeout);

   //---- Handling of system events ----------------------------
   void              CheckChilds();
   Bool_t            CheckSignals(Bool_t sync);
   Bool_t            CheckDescriptors();
   void              DispatchSignals(ESignals sig);
   void              AddSignalHandler(TSignalHandler *sh);
   TSignalHandler   *RemoveSignalHandler(TSignalHandler *sh);
   void              ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   void              ResetSignals();
   void              IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   void              SigAlarmInterruptsSyscalls(Bool_t set);
   void              AddFileHandler(TFileHandler *fh);
   TFileHandler     *RemoveFileHandler(TFileHandler *fh);

   //---- Floating Point Exceptions Control --------------------
   Int_t             GetFPEMask();
   Int_t             SetFPEMask(Int_t mask = kDefaultMask);

   //---- Time & Date ------------------------------------------
   TTime             Now();
   void              AddTimer(TTimer *ti);
   TTimer           *RemoveTimer(TTimer *ti);
   void              ResetTimer(TTimer *ti);
   Bool_t            DispatchTimers(Bool_t mode);
   void              Sleep(UInt_t milliSec);

   //---- Processes --------------------------------------------
   Int_t             Exec(const char *shellcmd);
   FILE             *OpenPipe(const char *shellcmd, const char *mode);
   int               ClosePipe(FILE *pipe);
   void              Exit(int code, Bool_t mode = kTRUE);
   void              Abort(int code = 0);
   int               GetPid();
   void              StackTrace();

   //---- Directories ------------------------------------------
   int               MakeDirectory(const char *name);
   void             *OpenDirectory(const char *name);
   void              FreeDirectory(void *dirp);
   const char       *GetDirEntry(void *dirp);
   Bool_t            ChangeDirectory(const char *path);
   const char       *WorkingDirectory();
   std::string       GetWorkingDirectory() const;
   const char       *HomeDirectory(const char *userName = 0);
   std::string       GetHomeDirectory(const char *userName = 0) const;
   const char       *TempDirectory() const;
   FILE             *TempFileName(TString &base, const char *dir = 0);

   //---- Paths & Files ----------------------------------------
   const char       *PrependPathName(const char *dir, TString& name);
   Bool_t            ExpandPathName(TString &patbuf);
   char             *ExpandPathName(const char *path);
   Bool_t            AccessPathName(const char *path, EAccessMode mode = kFileExists);
   Bool_t            IsPathLocal(const char *path);
   int               CopyFile(const char *from, const char *to, Bool_t overwrite = kFALSE);
   int               Rename(const char *from, const char *to);
   int               Link(const char *from, const char *to);
   int               Symlink(const char *from, const char *to);
   int               Unlink(const char *name);
   int               GetPathInfo(const char *path, FileStat_t &buf);
   int               GetFsInfo(const char *path, Long_t *id, Long_t *bsize,
                               Long_t *blocks, Long_t *bfree);
   int               Chmod(const char *file, UInt_t mode);
   int               Umask(Int_t mask);
   int               Utime(const char *file, Long_t modtime, Long_t actime);
   const char       *FindFile(const char *search, TString& file, EAccessMode mode = kFileExists);

   //---- Users & Groups ---------------------------------------
   Int_t             GetUid(const char *user = 0);
   Int_t             GetGid(const char *group = 0);
   Int_t             GetEffectiveUid();
   Int_t             GetEffectiveGid();
   UserGroup_t      *GetUserInfo(Int_t uid);
   UserGroup_t      *GetUserInfo(const char *user = 0);
   UserGroup_t      *GetGroupInfo(Int_t gid);
   UserGroup_t      *GetGroupInfo(const char *group = 0);

   //---- Environment Manipulation -----------------------------
   const char       *Getenv(const char *name);
   void              Setenv(const char *name, const char *value);
   void              Unsetenv(const char *name);

   //---- System Logging ---------------------------------------
   void              Openlog(const char *name, Int_t options, ELogFacility facility);
   void              Syslog(ELogLevel level, const char *mess);
   void              Closelog();

   //---- Standard Output redirection --------------------------
   Int_t             RedirectOutput(const char *name, const char *mode = "a",
                                    RedirectHandle_t *h = 0);

   //---- Dynamic Loading --------------------------------------
   void              AddDynamicPath(const char *lib);
   const char       *GetDynamicPath();
   void              SetDynamicPath(const char *lib);
   Func_t            DynFindSymbol(const char *module, const char *entry);
   int               Load(const char *module, const char *entry = "", Bool_t system = kFALSE);
   void              Unload(const char *module);
   void              ListSymbols(const char *module, const char *re = "");
   void              ListLibraries(const char *regexp = "");

   //---- RPC --------------------------------------------------
   TInetAddress      GetHostByName(const char *server);
   TInetAddress      GetPeerName(int sock);
   TInetAddress      GetSockName(int sock);
   int               GetServiceByName(const char *service);
   char             *GetServiceByPort(int port);
   int               ConnectService(const char *server, int port, int tcpwindowsize, const char *protocol = "tcp");
   int               OpenConnection(const char *server, int port, int tcpwindowsize = -1, const char *protocol = "tcp");
   int               AnnounceTcpService(int port, Bool_t reuse, int backlog, int tcpwindowsize = -1);
   int               AnnounceUdpService(int port, int backlog);
   int               AnnounceUnixService(int port, int backlog);
   int               AnnounceUnixService(const char *sockpath, int backlog);
   int               AcceptConnection(int sock);
   void              CloseConnection(int sock, Bool_t force = kFALSE);
   int               RecvRaw(int sock, void *buffer, int length, int flag);
   int               SendRaw(int sock, const void *buffer, int length, int flag);
   int               RecvBuf(int sock, void *buffer, int length);
   int               SendBuf(int sock, const void *buffer, int length);
   int               SetSockOpt(int sock, int option, int val);
   int               GetSockOpt(int sock, int option, int *val);

   //---- System, CPU and Memory info
   int               GetSysInfo(SysInfo_t *info) const;
   int               GetCpuInfo(CpuInfo_t *info, Int_t sampleTime = 1000) const;
   int               GetMemInfo(MemInfo_t *info) const;
   int               GetProcInfo(ProcInfo_t *info) const;

   ClassDef(TUnixSystem,0)  //Interface to Unix OS services
};

#endif
