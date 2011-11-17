// @(#)root/winnt:$Id$
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
 typedef void* HANDLE;
 struct WIN32_FIND_DATA;
 typedef void* HIMAGELIST;
 typedef void* HICON;
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


class TWinNTSystem : public TSystem {
public:
   // pointer to message handler func
   typedef Bool_t (*ThreadMsgFunc_t)(MSG*);

private:
   struct group     *fGroups;           // Groups on local computer
   struct passwd    *fPasswords;        // Users on local computer
   int               fNbUsers;          // Number of users on local computer
   int               fNbGroups;         // Number of groups on local computer
   int               fActUser;          // Index of actual user in User list
   Bool_t            fGroupsInitDone;   // Flag used for Users and Groups initialization
   Bool_t            fFirstFile;        // Flag used by OpenDirectory/GetDirEntry

   HANDLE            fhProcess;         // Handle of the current process
   void             *fGUIThreadHandle;  // handle of GUI server (aka command) thread
   ULong_t           fGUIThreadId;      // id of GUI server (aka command) thread
   char             *fDirNameBuffer;    // The string buffer to hold path name
   WIN32_FIND_DATA   fFindFileData;     // Structure to look for files (aka OpenDir under UNIX)

   Bool_t            DispatchTimers(Bool_t mode);
   Bool_t            CheckDescriptors();
   Bool_t            CheckSignals(Bool_t sync);
   Bool_t            CountMembers(const char *lpszGroupName);
   const char       *GetLinkedLibraries();
   Bool_t            GetNbGroups();
   Long_t            LookupSID (const char *lpszAccountName, int what, int &groupIdx, int &memberIdx);
   Bool_t            CollectMembers(const char *lpszGroupName, int &groupIdx, int &memberIdx);
   Bool_t            CollectGroups();
   Bool_t            InitUsersGroups();
   void              DoBeep(Int_t freq=-1, Int_t duration=-1) const;

   static void       ThreadStub(void *Parameter) {((TWinNTSystem *)Parameter)->TimerThread();}
   void              TimerThread();

protected:
   static int        WinNTUnixConnect(int port);
   static int        WinNTUnixConnect(const char *path);
   static int        WinNTUdpConnect(const char *hostname, int port);

public:
   TWinNTSystem();
   virtual ~TWinNTSystem();

   //---- non-TSystem methods ----------------------------------
   HANDLE            GetProcess();
   Bool_t            HandleConsoleEvent();

   //---- Misc -------------------------------------------------
   Bool_t            Init();
   const char       *BaseName(const char *name);
   void              SetProgname(const char *name);
   const char       *GetError();
   const char       *HostName();
   void             *GetGUIThreadHandle() const {return fGUIThreadHandle;}
   ULong_t           GetGUIThreadId() const {return fGUIThreadId;}
   void              SetGUIThreadMsgHandler(ThreadMsgFunc_t func);
   void              NotifyApplicationCreated();


   //---- EventLoop --------------------------------------------
   Bool_t            ProcessEvents();
   void              DispatchOneEvent(Bool_t pendingOnly = kFALSE);
   void              ExitLoop();
   Int_t             Select(TList *active, Long_t timeout);
   Int_t             Select(TFileHandler *fh, Long_t timeout);

   //---- Handling of system events ----------------------------
   void              DispatchSignals(ESignals sig);
   void              AddSignalHandler(TSignalHandler *sh);
   TSignalHandler   *RemoveSignalHandler(TSignalHandler *sh);
   void              ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   void              ResetSignals();
   void              IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   void              AddFileHandler(TFileHandler *fh);
   TFileHandler     *RemoveFileHandler(TFileHandler *fh);
   void              StackTrace();

   //---- Floating Point Exceptions Control --------------------
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
   const char       *TempDirectory() const;
   FILE             *TempFileName(TString &base, const char *dir = 0);

   //---- Users & Groups ---------------------------------------
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
   const char       *PrependPathName(const char *dir, TString& name);
   Bool_t            ExpandPathName(TString &patbuf);
   char             *ExpandPathName(const char *path);
   Bool_t            AccessPathName(const char *path, EAccessMode mode = kFileExists);
   Bool_t            IsPathLocal(const char *path);
   Bool_t            IsAbsoluteFileName(const char *dir);
   int               CopyFile(const char *from, const char *to, Bool_t overwrite = kFALSE);
   int               Rename(const char *from, const char *to);
   int               Link(const char *from, const char *to);
   int               Symlink(const char *from, const char *to);
   int               Unlink(const char *name);
   int               SetNonBlock(int fd);
   int               GetPathInfo(const char *path, FileStat_t &buf);
   int               GetFsInfo(const char *path, Long_t *id, Long_t *bsize,
                                 Long_t *blocks, Long_t *bfree);
   int               Chmod(const char *file, UInt_t mode);
   int               Umask(Int_t mask);
   int               Utime(const char *file, Long_t modtime, Long_t actime);
   const char       *UnixPathName(const char *unixpathname);
   const char       *FindFile(const char *search, TString& file, EAccessMode mode = kFileExists);
   TList            *GetVolumes(Option_t *opt = "") const;

   //---- Standard Output redirection --------------------------
   Int_t             RedirectOutput(const char *name, const char *mode = "a", RedirectHandle_t *h = 0);

   //---- Dynamic Loading --------------------------------------
   void              AddDynamicPath(const char *dir);
   const char       *GetDynamicPath();
   void              SetDynamicPath(const char *path);
   char             *DynamicPathName(const char *lib, Bool_t quiet = kFALSE);
   int               Load(const char *module, const char *entry = "", Bool_t system = kFALSE);
   const char       *GetLibraries(const char *regexp = "",
                                  const char *option = "",
                                  Bool_t isRegexp = kTRUE);

   //---- Time & Date -------------------------------------------
   TTime             Now();
   void              AddTimer(TTimer *ti);
   TTimer           *RemoveTimer(TTimer *ti);
   void              Sleep(UInt_t milliSec);
   Double_t          GetRealTime();
   Double_t          GetCPUTime();

   //---- RPC --------------------------------------------------
   int               ConnectService(const char *servername, int port, int tcpwindowsize, const char *protocol = "tcp");
   TInetAddress      GetHostByName(const char *server);
   TInetAddress      GetPeerName(int sock);
   TInetAddress      GetSockName(int sock);
   int               GetServiceByName(const char *service);
   char              *GetServiceByPort(int port);
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
   int               SetSockOpt(int sock, int opt, int val);
   int               GetSockOpt(int sock, int opt, int *val);

   //---- System, CPU and Memory info
   Int_t             GetSysInfo(SysInfo_t *info) const;
   Int_t             GetCpuInfo(CpuInfo_t *info, Int_t sampleTime = 1000) const;
   Int_t             GetMemInfo(MemInfo_t *info) const;
   Int_t             GetProcInfo(ProcInfo_t *info) const;

   ClassDef(TWinNTSystem, 0)
};

R__EXTERN ULong_t gConsoleWindow;   // console window handle

#endif
