// @(#)root/winnt:$Name:  $:$Id: TWinNTSystem.h,v 1.27 2004/01/29 16:02:03 brun Exp $
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

#ifndef MAX_SID_SIZE
#define MAX_SID_SIZE   1024 
#endif
#define MAX_NAME_STRING   1024

#define SID_GROUP         0
#define SID_MEMBER        1

struct passwd {
   char *pw_name;       // user name
   char *pw_passwd;     // user password
   int   pw_uid;        // user ID
   int   pw_gid;        // user's group ID
   int   pw_quota;      // 
   char *pw_gecos;      // user's real (long) name
   char *pw_dir;        // user home directory
   char *pw_shell;      // shell command
   char *pw_group;      // user's group name
};

struct group {
   char   *gr_name;     // group name
   char   *gr_passwd;   // group password
   int    gr_gid;       // group id
   char   **gr_mem;     // group members
};


class TWin32Timer;

class TWinNTSystem : public TSystem {

private:

   struct group     *fGroups;           // Groups on local computer
   struct passwd    *fPasswords;        // Users on local computer
   int               fNbUsers;          // Number of users on local computer
   int               fNbGroups;         // Number of groups on local computer
   int               fActUser;          // Index of actual user in User list


   HANDLE            fhProcess;         // Handle of the current process
   HANDLE            fhTermInputEvent;  // Handle of "event" to suspend "dummy" terminal loop
   char             *fDirNameBuffer;    // The string buffer to hold path name
   WIN32_FIND_DATA   fFindFileData;     // Structure to look for files (aka OpenDir under UNIX)
   TWin32Timer      *fWin32Timer;       // Windows -asynch timer
   HIMAGELIST        fhSmallIconList;   // List of the small icons
   HIMAGELIST        fhNormalIconList;  // List of the normal icons
   const char       *fShellName;        // the name of the "shell" file to pool the icons
   void              CreateIcons();     // Create a list of the icons for ROOT appl

   Bool_t            DispatchTimers(Bool_t mode);
   Bool_t            DispatchSynchTimers();
   Bool_t            CheckDescriptors();
   Bool_t            CheckSignals(Bool_t sync);

   Bool_t            CountMembers(const char *lpszGroupName);
   Bool_t            GetNbGroups();
   Long_t            LookupSID (const char *lpszAccountName, int what);
   Bool_t            CollectMembers(const char *lpszGroupName);
   Bool_t            CollectGroups();

public:
   TWinNTSystem();
   ~TWinNTSystem();

   //---- non-TSystem methods ----------------------------------
   HIMAGELIST GetSmallIconList() { return fhSmallIconList; }
   HICON   GetSmallIcon(Int_t IconIdx) { return fhSmallIconList  ? ImageList_GetIcon(fhSmallIconList,IconIdx,ILD_NORMAL):0; }
   HICON   GetNormalIcon(Int_t IconIdx){ return fhNormalIconList ? ImageList_GetIcon(fhNormalIconList,IconIdx,ILD_NORMAL):0; }
   HIMAGELIST        GetNormalIconList(){ return fhNormalIconList; }
   HANDLE            GetProcess();
   const char        *GetShellName() {return fShellName;}
   void              SetShellName(const char *name=0);
   Bool_t            HandleConsoleEvent();

   //---- Misc -------------------------------------------------
   Bool_t            Init();
   const char       *BaseName(const char *name);
   void              SetProgname(const char *name);
   const char       *GetError();
   const char       *HostName();
   //---- EventLoop --------------------------------------------
   Bool_t            ProcessEvents();
   void              DispatchOneEvent(Bool_t pendingOnly = kFALSE);
   void              ExitLoop();
   //---- Handling of system events
   void              AddSignalHandler(TSignalHandler *sh);
   TSignalHandler   *RemoveSignalHandler(TSignalHandler *sh);
   void              ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   void              IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   void              AddFileHandler(TFileHandler *fh);
   TFileHandler     *RemoveFileHandler(TFileHandler *fh);
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
   //---- Users & Groups
   Int_t             GetUid(const char *user = 0);
   Int_t             GetGid(const char *group = 0);
   Int_t             GetEffectiveUid();
   Int_t             GetEffectiveGid();
   UserGroup_t      *GetUserInfo(Int_t uid);
   UserGroup_t      *GetUserInfo(const char *user = 0);
   UserGroup_t      *GetGroupInfo(Int_t gid);
   UserGroup_t      *GetGroupInfo(const char *group = 0);
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
   void              Sleep(UInt_t milliSec);
   Double_t          GetRealTime();
   Double_t          GetCPUTime();
   //---- RPC --------------------------------------------------
   int               ConnectService(const char *servername, int port, int tcpwindowsize);
   TInetAddress      GetHostByName(const char *server);
   TInetAddress      GetPeerName(int sock);
   TInetAddress      GetSockName(int sock);
   int               GetServiceByName(const char *service);
   char              *GetServiceByPort(int port);
   int               OpenConnection(const char *server, int port, int tcpwindowsize = -1);
   int               AnnounceTcpService(int port, Bool_t reuse, int backlog, int tcpwindowsize = -1);
   int               AnnounceUnixService(int port, int backlog);
   int               AcceptConnection(int sock);
   void              CloseConnection(int sock, Bool_t force = kFALSE);
   int               RecvRaw(int sock, void *buffer, int length, int flag);
   int               SendRaw(int sock, const void *buffer, int length, int flag);
   int               RecvBuf(int sock, void *buffer, int length);
   int               SendBuf(int sock, const void *buffer, int length);
   int               SetSockOpt(int sock, int opt, int val);
   int               GetSockOpt(int sock, int opt, int *val);

   ClassDef(TWinNTSystem, 0)
};

#endif
