// @(#)root/winnt:$Name:  $:$Id: TWinNTSystem.h,v 1.24 2004/01/26 09:49:26 brun Exp $
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWinNTSystem
#define ROOT_TWinNTSystem

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWinNTSystem                                                         //
//                                                                      //
// Class providing an interface to the Windows NT Operating System.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


#if !defined(__CINT__)
 #include "Windows4Root.h"
 #include <commctrl.h>
#else
 typedef ULong_t HANDLE;
 typedef ULong_t WIN32_FIND_DATA;
 typedef ULong_t HIMAGELIST;
 typedef ULong_t HICON;
 typedef UChar_t BOOL;
 struct FILE;
#endif



class TWin32Timer;

class TWinNTSystem : public TSystem {

private:
   HANDLE          fhProcess;         // Handle of the current process
   HANDLE          fhTermInputEvent;  // Handle of "event" to suspend "dummy" terminal loop
   char           *fDirNameBuffer;    // The string buffer to hold path name
   WIN32_FIND_DATA fFindFileData;     // Structure to look for files (aka OpenDir under UNIX)
   TWin32Timer    *fWin32Timer;       // Windows -asynch timer
   HIMAGELIST      fhSmallIconList;   // List of the small icons
   HIMAGELIST      fhNormalIconList;  // List of the normal icons
   const char     *fShellName;        // the name of the "shell" file to pool the icons
   void            CreateIcons();     // Create a list of the icons for ROOT appl

public:
   TWinNTSystem();
   ~TWinNTSystem();

   HIMAGELIST GetSmallIconList() { return fhSmallIconList; }
   HICON   GetSmallIcon(Int_t IconIdx) { return fhSmallIconList  ? ImageList_GetIcon(fhSmallIconList,IconIdx,ILD_NORMAL):0; }
   HICON   GetNormalIcon(Int_t IconIdx){ return fhNormalIconList ? ImageList_GetIcon(fhNormalIconList,IconIdx,ILD_NORMAL):0; }
   HIMAGELIST GetNormalIconList(){ return fhNormalIconList; }
   HANDLE     GetProcess();

   //---- Misc -------------------------------------------------
   Bool_t            Init();
   const char       *BaseName(const char *name);
   void              SetProgname(const char *name);
   const char       *GetError();
   const char       *HostName();
   const char       *GetShellName() {return fShellName;}
   void              SetShellName(const char *name=0);
   //---- EventLoop --------------------------------------------
   Bool_t            ProcessEvents();
   void              DispatchOneEvent(Bool_t pendingOnly = kFALSE);
   void              ExitLoop();
   //---- Handling of system events
   void              CheckChilds();
   Bool_t            CheckSignals(Bool_t sync);
   Bool_t            CheckDescriptors();
   void              DispatchSignals(ESignals sig);
   void              AddSignalHandler(TSignalHandler *sh);
   TSignalHandler   *RemoveSignalHandler(TSignalHandler *sh);
   void              ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   void              IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   void              AddFileHandler(TFileHandler *fh);
   TFileHandler     *RemoveFileHandler(TFileHandler *fh);
   Bool_t            HandleConsoleEvent();
   //---- Floating Point Exceptions Control
   Int_t             GetFPEMask();
   Int_t             SetFPEMask(Int_t mask = kDefaultMask);
   //---- Processes --------------------------------------------
   int               Exec(const char *shellcmd);
   FILE             *OpenPipe(const char *shellcmd, const char *mode);
   int               ClosePipe(FILE *pipe);
   void              Exit(int code, Bool_t mode = kTRUE);
   void              Abort(int code = 0);
   int               GetPid();
   //---- Environment manipulation -----------------------------
   const char       *Getenv(const char *name);
   void              Setenv(const char *name, const char *value);
   //---- Directories ------------------------------------------
   int               mkdir(const char *name, Bool_t recursive = kFALSE);
   int               MakeDirectory(const char *name);
   Bool_t            ChangeDirectory(const char *path);
   const char       *GetDirEntry(void *dirp);
   const char       *DirName(const char *pathname);
   void              FreeDirectory(void *dirp);
   void             *OpenDirectory(const char *name);
   const char       *WorkingDirectory(char driveletter);
   const char       *WorkingDirectory();
   const char       *HomeDirectory(const char *userName=0);
   char             *ConcatFileName(const char *dir, const char *name);
   const char       *TempDirectory() const;
   FILE             *TempFileName(TString &base, const char *dir = 0);
   //---- Paths & Files ----------------------------------------
   const char        DriveName(const char *pathname="/");
   Bool_t            ExpandPathName(TString &patbuf);
   char             *ExpandPathName(const char *path);
   Bool_t            AccessPathName(const char *path, EAccessMode mode = kFileExists);
   Bool_t            IsAbsoluteFileName(const char *dir);
   int               CopyFile(const char *from, const char *to, Bool_t overwrite = kFALSE);
   int               Rename(const char *from, const char *to);
   int               Link(const char *from, const char *to);
   int               Unlink(const char *name);
   int               SetNonBlock(int fd);
   int               GetPathInfo(const char *path, Long_t *id, Long_t *size,
                                 Long_t *flags, Long_t *modtime)
                        { return TSystem::GetPathInfo(path, id, size, flags, modtime); }
   int               GetPathInfo(const char *path, Long_t *id, Long64_t *size,
                                 Long_t *flags, Long_t *modtime);
   int               GetFsInfo(const char *path, Long_t *id, Long_t *bsize,
                                 Long_t *blocks, Long_t *bfree);
   int               Umask(Int_t mask);
   int               Utime(const char *file, Long_t modtime, Long_t actime);
   const char       *UnixPathName(const char *unixpathname);
   char             *Which(const char *search, const char *file, EAccessMode mode = kFileExists);
   //---- Dynamic Loading --------------------------------------
   char             *DynamicPathName(const char *lib, Bool_t quiet = kFALSE);
   const char       *GetLibraries(const char *regexp = "",
                                  const char *option = "",
                                  Bool_t isRegexp = kTRUE);
   //---- Time & Date
   TTime             Now();
   void              AddTimer(TTimer *ti);
   TTimer           *RemoveTimer(TTimer *ti);
   Bool_t            DispatchTimers(Bool_t mode);
   Bool_t            DispatchSynchTimers();
   void              Sleep(UInt_t milliSec);
   Double_t          GetRealTime();
   Double_t          GetCPUTime();
   //---- RPC --------------------------------------------------
   virtual int             ConnectService(const char *servername, int port, int tcpwindowsize);
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
   virtual int             SetSockOpt(int sock, int opt, int val);
   virtual int             GetSockOpt(int sock, int opt, int *val);
   //---- Static utility functions ------------------------------
   static const char  *GetDynamicPath();

   ClassDef(TWinNTSystem, 0)
};

#endif
