// @(#)root/winnt:$Name:  $:$Id: TWinNTSystem.h,v 1.2 2000/06/28 15:30:44 rdm Exp $
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

#if !defined(__CINT__)

#include <io.h>
#include <process.h>
#include "Windows4Root.h"

#endif

#include "TSystem.h"
#include "TApplication.h"
#include "TSysEvtHandler.h"
#include "TTimer.h"

#if !defined(__CINT__)

#include <commctrl.h>

#else

typedef ULong_t HANDLE;
typedef ULong_t WIN32_FIND_DATA;
typedef ULong_t HIMAGELIST;
typedef ULong_t HICON;
typedef UChar_t BOOL;

struct FILE;

#endif

class TSeqCollection;

typedef void (*SigHandler_t)(ESignals);


class TClass;
class TFunction;
class TMethod;
class TSeqCollection;
class TObjArray;
class TWin32Timer;

class TWinNTSystem : public TSystem {

protected:
   HANDLE          fhProcess;         // Handle of the current process
   char           *fDirNameBuffer;    // The string buffer to hold path name
   WIN32_FIND_DATA  fFindFileData;    // Structure to look for files (aka OpenDir under UNIX)

   const char     *fShellName;        // the name of the "shell" file to pool the icons
   TWin32Timer    *fWin32Timer;       // Windows -asynch timer

   HIMAGELIST fhSmallIconList;  // List of the small icons
   HIMAGELIST fhNormalIconList; // List of the normal icons

   void                CreateIcons();            // Create a list of the icons for Root appl

  // static functions providing semi-low level interface to raw WinNT
   static const char  *WinNTHomedirectory(const char *user = 0);
   static int          WinNTWaitchild();
   static int          WinNTSetitimer(TTimer *ti);
   static int          WinNTSelect(UInt_t nfds, TFdSet *readready, TFdSet *writeready,
                                  Long_t timeout);
   static void         WinNTSignal(ESignals sig, SigHandler_t h);
   static char        *WinNTSigname(ESignals sig);
   static int          WinNTFilestat(const char *path, Long_t *id, Long_t *size,
                                    Long_t *flags, Long_t *modtime);
   static int          WinNTTcpConnect(const char *hostname, int port);
   static int          WinNTWinNTConnect(const char *hostname, int port);
   static int          WinNTTcpService(int port,  Bool_t reuse, int backlog);
   static int          WinNTWinNTService(int port, int backlog);
   static int          WinNTSend(int socket, const void *buffer, int length, int flag);
   static int          WinNTRecv(int socket, void *buffer, int length, int flag);

   static int          WinNTIoctl(int fd, int code, void *vp);
   static int          WinNTNonblock(int fd);
   static long         WinNTNow();

   static int          WinNTDynLoad(const char *lib);
   static Func_t       WinNTDynFindSymbol(const char *lib, const char *entry);
   static void         WinNTDynUnload(const char *lib);
   static void         WinNTDynListSymbols(const char *lib, const char *re = "");
   static void         WinNTDynListLibs(const char *lib = "");
   //----> static HANDLE       WinNTGetCurrentProcess();

public:
   TWinNTSystem();
   ~TWinNTSystem();

   //---- Misc -------------------------------------------------
   Bool_t            Init();
   const char       *BaseName(const char *name);
   void              SetProgname(const char *name);
   const char       *GetError();
   const char       *Hostname();

   HIMAGELIST GetSmallIconList() { return fhSmallIconList; }
   HICON   GetSmallIcon(Int_t IconIdx) {return fhSmallIconList  ? ImageList_GetIcon(fhSmallIconList,IconIdx,ILD_NORMAL):0; }
   HICON   GetNormalIcon(Int_t IconIdx){return fhNormalIconList ? ImageList_GetIcon(fhNormalIconList,IconIdx,ILD_NORMAL):0; }
   HIMAGELIST GetNormalIconList(){ return fhNormalIconList; }

   const char       *GetShellName(){return fShellName;}
   void              SetShellName(const char *name=0);

   //---- EventLoop --------------------------------------------
   Bool_t            ProcessEvents();
   void              DispatchOneEvent(Bool_t pendingOnly = kFALSE);

   //---- Handling of system events
   void              CheckChilds();
   Bool_t            CheckSignals(Bool_t sync);
   void              DispatchSignals(ESignals sig);
   void              AddSignalHandler(TSignalHandler *sh);
   TSignalHandler   *RemoveSignalHandler(TSignalHandler *sh);
   void              AddFileHandler(TFileHandler *fh);
   TFileHandler     *RemoveFileHandler(TFileHandler *fh);
   BOOL              HandleConsoleEvent();
   void              IgnoreInterrupt(Bool_t ignore = kTRUE);

   //---- Processes --------------------------------------------
   int               Exec(const char *shellcmd);
   FILE             *OpenPipe(const char *shellcmd, const char *mode);
   int               ClosePipe(FILE *pipe);
   void              Exit(int code, Bool_t mode = kTRUE);
   void              Abort(int code = 0);
   int               GetPid();
   HANDLE            GetProcess();

   //---- Environment manipulation -----------------------------
   const char       *Getenv(const char *name);
   void              Setenv(const char *name, const char *value);

   //---- Directories ------------------------------------------
   int               MakeDirectory(const char *name);
   Bool_t            ChangeDirectory(const char *path);
   const char       *GetDirEntry(void *dirp);
   const char       *DirName(const char *pathname);
   void              FreeDirectory(void *dirp);
   void             *OpenDirectory(const char *name);

   const char       *WorkingDirectory(char driveletter);  // The working directory for the selecte drive
   const char       *WorkingDirectory();                  // The working directory for the default drive
   const char       *HomeDirectory(const Char_t *userName=0);
   char             *ConcatFileName(const char *dir, const char *name);

   //---- Paths & Files ----------------------------------------
   const char        DriveName(const char *pathname="/");
   Bool_t            ExpandPathName(TString &patbuf);
   char             *ExpandPathName(const char *path);
   Bool_t            AccessPathName(const char *path, EAccessMode mode = kFileExists);
   Bool_t            IsAbsoluteFileName(const char *dir);
   void              Rename(const char *from, const char *to);
   int               Link(const char *from, const char *to);
   int               Unlink(const char *name);
   int               SetNonBlock(int fd);
   int               GetPathInfo(const char *path, Long_t *id, Long_t *size,
                                 Long_t *flags, Long_t *modtime);
   const char       *UnixPathName(const char *unixpathname);

   //---- Dynamic Loading --------------------------------------
   char             *DynamicPathName(const char *lib, Bool_t quiet = kFALSE);
   Func_t            DynFindSymbol(const char *module, const char *entry);
   int               Load(const char *module, const char *entry = "", Bool_t system = kFALSE);
   void              Unload(const char *module);
   void              ListSymbols(const char *module, const char *re = "");
   void              ListLibraries(const char *regexp = "");
   const char       *GetLibraries(const char *regexp = "", const char* option = "");

   //---- Time & Date
   TTime             Now();
   void              AddTimer(TTimer *ti);
   TTimer           *RemoveTimer(TTimer *ti);
//   void              DispatchTimers();
   Bool_t            DispatchSynchTimers();
   void              Sleep(UInt_t milliSec);
   Double_t          GetRealTime();
   Double_t          GetCPUTime();

   //---- RPC --------------------------------------------------
   virtual int             ConnectService(const char *servername, int port);
   virtual TInetAddress    GetHostByName(const char *server);
   virtual TInetAddress    GetPeerName(int sock);
   virtual TInetAddress    GetSockName(int sock);
   virtual int             GetServiceByName(const char *service);
   virtual char           *GetServiceByPort(int port);
   virtual int             OpenConnection(const char *server, int port);
   virtual int             AnnounceTcpService(int port, Bool_t reuse, int backlog);
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
   char  *Which(const char *search, const char *file, EAccessMode mode = kFileExists);
   static const char  *GetDynamicPath();

   ClassDef(TWinNTSystem,0)
};

#endif
