// @(#)root/winnt:$Name:  $:$Id: TWinNTSystem.cxx,v 1.4 2000/08/18 06:27:32 brun Exp $
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
// TWinNTSystem                                                                 //
//                                                                              //
// Class providing an interface to the Windows NT/Windows 95 Operating Systems. //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "Windows4Root.h"
#include "TWinNTSystem.h"
#include "TROOT.h"
#include "TError.h"
#include "TMath.h"
#include "TOrdCollection.h"
#include "TRegexp.h"
#include "TException.h"
#include "Demangle.h"
#include "TEnv.h"
#include "TWin32HookViaThread.h"
#include "TWin32Timer.h"
#include "TGWin32Command.h"
#include "TSocket.h"
#include "TApplication.h"

#include "Win32Constants.h"

#ifndef WIN32
# include <unistd.h>
#endif

#include <stdlib.h>
#include <sys/types.h>


// Below my portion

#include <direct.h>
#include <ctype.h>

#if defined(R__SUN) || defined(R__SGI) || defined(R__HPUX)
#   include <dirent.h>
#else
# if !defined(WIN32) || defined(_SC_)
#   include <sys/dir.h>
# endif
#endif
#if defined(ULTRIX) || defined(R__SUN)
#include <sgtty.h>
#endif
#ifdef R__AIX
#   include <sys/ioctl.h>
#endif
#include <sys/stat.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#ifndef WIN32
#include <sys/param.h>
#include <pwd.h>
#endif

#include <errno.h>

#ifndef WIN32
#include <sys/wait.h>
#endif

#if !defined(WIN32) || defined (_SC_)
 #include <sys/time.h>
#endif

#ifndef WIN32
#include <sys/file.h>
#endif

#include <sys/types.h>

#ifndef WIN32
# include <sys/socket.h>
# include <netinet/in.h>
# include <sys/un.h>
# include <netdb.h>
#else
# include <winsock.h>
#endif

#include <fcntl.h>
#ifdef R__HPUX
#   include <dl.h>
#   if defined(R__GNU)
       extern "C" {
          extern shl_t cxxshl_load(const char *path, int flags, long address);
          extern int   cxxshl_unload(shl_t handle);
       }
#   else
#      include <cxxdl.h>
#   endif
    // print stack trace (HP undocumented internal call)
    extern "C" void U_STACK_TRACE();
#endif

const char *kProtocolName   = "tcp";


//______________________________________________________________________________
BOOL ConsoleSigHandler(DWORD sig)
{

 // WinNT signal handler.

  switch (sig) {
  case CTRL_C_EVENT:
      printf(" CTRL-C hit !!! ROOT is terminated ! \n");
  case CTRL_BREAK_EVENT:
//      return ((TWinNTSystem*)gSystem)->HandleConsoleEvent();
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
  case CTRL_CLOSE_EVENT:
  default:
      gSystem->Exit(-1); return kTRUE;
  }
}


//______________________________________________________________________________
void SigHandler(ESignals sig)
{
   if (gSystem)
      ((TWinNTSystem*)gSystem)->DispatchSignals(sig);
}

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
class TTermInputLine :  public  TWin32HookViaThread
{
  protected:
    void ExecThreadCB(TWin32SendClass *sentclass);
  public:
    TTermInputLine::TTermInputLine();
};

//______________________________________________________________________________
TTermInputLine::TTermInputLine()
{
  TWin32SendWaitClass CodeOp(this);
  ExecCommandThread(&CodeOp,kFALSE);
  CodeOp.Wait();
}

//______________________________________________________________________________
void TTermInputLine::ExecThreadCB(TWin32SendClass *code)
{
// Dispatch a single event.
   gROOT->GetApplication()->HandleTermInput();
   ((TWin32SendWaitClass *)code)->Release();
}


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

ClassImp(TWinNTSystem)


//______________________________________________________________________________
BOOL TWinNTSystem::HandleConsoleEvent(){
 TSignalHandler *sh;
 TIter next(fSignalHandler);
 ESignals s;

 while (sh = (TSignalHandler*)next()) {
      s = sh->GetSignal();
      if (s == kSigInterrupt) {
              sh->Notify();
              Throw(SIGINT);
             return TRUE;
      }
  }
  return FALSE;
}

//______________________________________________________________________________
TWinNTSystem::TWinNTSystem() : TSystem("WinNT", "WinNT System")
{
   fhProcess = GetCurrentProcess();
   fDirNameBuffer = 0;
   fShellName = 0;

   WSADATA  WSAData;
   int initwinsock= 0;
   if (initwinsock = WSAStartup(MAKEWORD(2,0),&WSAData)) {
           Error("TWinNTSystem()","Starting sockets failed");  //  %d \n",initwinsock);
           // printf(" exit code = %d \n", 1);
           // return -1;
   }
}

//______________________________________________________________________________
TWinNTSystem::~TWinNTSystem()
{

 SafeDelete(fWin32Timer);

//*-* Clean up the WinSocket connectios
  WSACleanup();

  if (fDirNameBuffer) {
   delete [] fDirNameBuffer;  fDirNameBuffer = 0;
  }

 if (fhSmallIconList)  {ImageList_Destroy(fhSmallIconList);  fhSmallIconList = 0; }
 if (fhNormalIconList) {ImageList_Destroy(fhNormalIconList); fhNormalIconList = 0; }

}

//______________________________________________________________________________
Bool_t TWinNTSystem::Init()
{
   // Initialize WinNT system interface.

   const char *dir=0;

   if (TSystem::Init())
      return kTRUE;

   //--- install default handlers
   WinNTSignal(kSigChild,                 SigHandler);
   WinNTSignal(kSigBus,                   SigHandler);
   WinNTSignal(kSigSegmentationViolation, SigHandler);
   WinNTSignal(kSigIllegalInstruction,    SigHandler);
   WinNTSignal(kSigSystem,                SigHandler);
   WinNTSignal(kSigPipe,                  SigHandler);
   WinNTSignal(kSigAlarm,                 SigHandler);
   WinNTSignal(kSigFloatingException,     SigHandler);

   fSigcnt = 0;

   fNfd    = 0;
   fMaxrfd = 0;
   fMaxwfd = 0;
   fReadmask.Zero();
   fWritemask.Zero();

#ifndef ROOTPREFIX
   gRootDir = Getenv("ROOTSYS");
   if (gRootDir == 0) {
     static char lpFilename[MAX_PATH];
     if (GetModuleFileName(NULL,                 // handle to module to find filename for
                           lpFilename,           // pointer to buffer to receive module path
                           sizeof(lpFilename)))  // size of buffer, in characters
     {
        const char *dirName = DirName(DirName(lpFilename));
        gRootDir = StrDup(dirName);
     } else {
        gRootDir = 0;
     }
   }
#else
   gRootDir= ROOTPREFIX;
#endif

//*-*  The the name of the DLL to be used as a stock of the icon
   SetShellName();
   CreateIcons();

   return kFALSE;
}

//---- Misc --------------------------------------------------------------------

//______________________________________________________________________________
const char *TWinNTSystem::BaseName(const char *name)
{
   // Base name of a file name. Base name of /user/root is root.
   // But the base name of '/' is '/'
   //                      'c:\' is 'c:\'
   // The calling routine should use free() to free memory BaseName allocated
   // for the base name

   if (name) {
      int idx = 0;
      const char *symbol=name;
   //*-*
   //*-*  Skip leading blanks
   //*-*
      while ( (*symbol == ' ' || *symbol == '\t') && *symbol) symbol++;
      if (*symbol) {
        if (isalpha(symbol[idx]) && symbol[idx+1] == ':') idx = 2;
        if ( (symbol[idx] == '/'  ||  symbol[idx] == '\\')  &&  symbol[idx+1] == '\0')
                                      return StrDup(symbol);
      }
      else {
         Error("BaseName", "name = 0");
         return 0;
      }
      char *cp;
      char *bslash = (char *)strrchr(&symbol[idx],'\\');
      char *rslash = (char *)strrchr(&symbol[idx],'/');
      if (cp = max(rslash, bslash)) return ++cp;
      return StrDup(&symbol[idx]);
   }
   Error("BaseName", "name = 0");
   return 0;
}

 //______________________________________________________________________________
void TWinNTSystem::CreateIcons()
{
 //  char shellname[] =  "RootShell32.dll";
  const char *shellname =  fShellName;

  HINSTANCE hShellInstance  = LoadLibrary(shellname);
  fhSmallIconList  = 0;
  fhNormalIconList = 0;

  if (hShellInstance)
  {
      fhSmallIconList = ImageList_Create(GetSystemMetrics(SM_CXSMICON),
                                         GetSystemMetrics(SM_CYSMICON),
                                         ILC_MASK,kTotalNumOfICons,1);

      fhNormalIconList = ImageList_Create(GetSystemMetrics(SM_CXICON),
                                          GetSystemMetrics(SM_CYICON),
                                          ILC_MASK,kTotalNumOfICons,1);

      HICON hicon;
      HICON hDummyIcon =  LoadIcon(NULL, IDI_APPLICATION);

//*-*  Add "ROOT" main icon
      hicon = LoadIcon(GetModuleHandle(NULL),MAKEINTRESOURCE(101));
      if (!hicon)
           hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(101));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

//*-*  Add "Canvas" icon
      hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(16));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

//*-*  Add "Browser" icon
      hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(171));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

//*-*  Add "Closed Folder" icon
      hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(4));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

//*-*  Add the "Open Folder" icon
      hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(5));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

//*-*  Add the "Document" icon
      hicon = LoadIcon(hShellInstance,MAKEINTRESOURCE(152));
      if (!hicon) hicon = hDummyIcon;
      ImageList_AddIcon(fhSmallIconList,hicon);
      ImageList_AddIcon(fhNormalIconList,hicon);
      if (hicon != hDummyIcon) DeleteObject(hicon);

      FreeLibrary((HMODULE)hShellInstance);
   }
}

//______________________________________________________________________________
void  TWinNTSystem::SetShellName(const char *name)
 {
     const char *shellname = "SHELL32.DLL";
     if (name)
     {
         fShellName = new char[lstrlen(name)+1];
         strcpy((char *)fShellName,name);
     }
     else
     {
//*-* use the system "shell32.dll" file as the icons stock.
//*-*  Check the type of the OS
       OSVERSIONINFO OsVersionInfo;

//*-*         Value                      Platform
//*-*  ----------------------------------------------------
//*-*  VER_PLATFORM_WIN32s          Win32s on Windows 3.1
//*-*  VER_PLATFORM_WIN32_WINDOWS       Win32 on Windows 95
//*-*  VER_PLATFORM_WIN32_NT            Windows NT
//*-*
      OsVersionInfo.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
      GetVersionEx(&OsVersionInfo);
      if (OsVersionInfo.dwPlatformId == VER_PLATFORM_WIN32_NT) {
        fShellName = strcpy(new char[lstrlen(shellname)+1],shellname);
      }
      else
      {
//*-*  for Windows 95 we have to create a local copy this file
        const char *rootdir = gRootDir;
        const char newshellname[] = "bin/RootShell32.dll";
        fShellName = ConcatFileName(gRootDir,newshellname);

        char sysdir[1024];
        GetSystemDirectory(sysdir,1024);
        char *sysfile = (char *) ConcatFileName(sysdir,shellname);
        CopyFile(sysfile,fShellName,TRUE);  // TRUE means "don't overwrite if fShellName is exists
        delete [] sysfile;
      }

     }

 }


//______________________________________________________________________________
void TWinNTSystem::SetProgname(const char *name)
{
   // Set the application name (from command line, argv[0]) and copy it in
   // gProgName. Copy the application pathname in gProgPath.

   ULong_t  idot = 0;
   char *dot = 0;
   char *progname;
   const char *fullname=0; // the program name with extension

  // On command prompt the progname can be supplied with no extension (under Windows)
  // if it is case we have to guess that extension ourselves

   const Char_t *extlist[]={"exe","bat","cmd"}; // List of extensions to guess
   Int_t lextlist = 3;                          // number of the extra extensions to guess

   if (name && strlen(name) > 0) {
//*-* Check whether the name contains "extention"

      fullname = name;
      while (!(dot = strchr(fullname,'.')))
      {
        idot = strlen(fullname);
        const Char_t *b = Form("%s.exe",name);
        fullname = b;
      }

      idot = (ULong_t) (dot - fullname);

      progname = StrDup(BaseName(fullname));


      char *which = 0;

      if ( IsAbsoluteFileName(fullname) && !AccessPathName(fullname))
          which = StrDup(fullname);
      else
          which = Which(Form("%s;%s",WorkingDirectory(),Getenv("PATH")), progname);

      if (which)
      {
        const char *dirname;
        char driveletter = DriveName(which);
        const char *d = DirName(which);
        if (driveletter)
          dirname = Form("%c:%s",driveletter,d);
        else
          dirname = Form("%s",d);

        gProgPath = StrDup(dirname);
      }
      else {
          Warning("SetProgname","Wrong Program path");
          gProgPath = "c:/users/root/ms/bin";
      }

//*-*  Cut the extension for progname off
      progname[idot] = '\0';
      gProgName = StrDup(progname);
      if (which) delete [] which;
   }
}

//______________________________________________________________________________
const char *TWinNTSystem::GetError()
{
   // Return system error string.

  // GetLastError Could ne introduced in here

   if (GetErrno() < 0 || GetErrno() >= sys_nerr)
      return Form("errno out of range %d", GetErrno());
   return sys_errlist[GetErrno()];
}

//______________________________________________________________________________
const char *TWinNTSystem::Hostname()
{
   // Return the system's host name.

   if (fHostname == "") {
      char hn[64];
      int il = sizeof(hn);
      GetComputerName((LPTSTR)hn,( LPDWORD)&il);
//      gethostname(hn, sizeof(hn));
      fHostname = hn;
   }
   return (const char *)fHostname;
}

//---- EventLoop ---------------------------------------------------------------

//______________________________________________________________________________
void TWinNTSystem::AddFileHandler(TFileHandler *h)
{
   // Add a file handler to the list of system file handlers.

   TSystem::AddFileHandler(h);
   if (h) {
      int fd = h->GetFd();
      if (h->HasReadInterest()) {
         fReadmask.Set(fd);
         fMaxrfd = TMath::Max(fMaxrfd, fd);
      }
      if (h->HasWriteInterest()) {
         fWritemask.Set(fd);
         fMaxwfd = TMath::Max(fMaxwfd, fd);
      }
   }
}

//______________________________________________________________________________
TFileHandler *TWinNTSystem::RemoveFileHandler(TFileHandler *h)
{
   // Remove a file handler from the list of file handlers.

   TFileHandler *oh = TSystem::RemoveFileHandler(h);
   if (oh) {       // found
      fReadmask.Clr(oh->GetFd());
      fWritemask.Clr(oh->GetFd());
   }
   return oh;
}

//______________________________________________________________________________
void TWinNTSystem::IgnoreInterrupt(Bool_t ignore)
{
   // Ignore the interrupt signal if ignore == kTRUE else restore previous
   // behaviour. Typically call ignore interrupt before writing to disk.
}

//______________________________________________________________________________
void TWinNTSystem::AddSignalHandler(TSignalHandler *h)
{
   // Add a signal handler to list of system signal handlers.

   TSystem::AddSignalHandler(h);
   ESignals  sig = h->GetSignal();
//*-*  Add a new handler to the list of the console handlers
   if (sig == kSigInterrupt)
                     SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleSigHandler,TRUE);
   WinNTSignal(h->GetSignal(), SigHandler);
}

//______________________________________________________________________________
TSignalHandler *TWinNTSystem::RemoveSignalHandler(TSignalHandler *h)
{
   // Remove a signal handler from list of signal handlers.

   int sig = h->GetSignal();
   if (sig = kSigInterrupt){
//*-*  Remove a  handler to the list of the console handlers
        SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleSigHandler,FALSE);
}
   return TSystem::RemoveSignalHandler(h);
}

//______________________________________________________________________________
Bool_t TWinNTSystem::ProcessEvents()
{
   // Events are processed by separate thread. Here we just return the
   // interrupt value the might have been set in command thread.

   Bool_t intr = gROOT->IsInterrupted();
   gROOT->SetInterrupt(kFALSE);
   return intr;
}
//______________________________________________________________________________
void TWinNTSystem::DispatchOneEvent(Bool_t)
{
 // Dispatch a single event via Command thread

  gROOT->GetApplication()->HandleTermInput();
#if 0
     // check for file descriptors ready for reading/writing
  if (fNfd > 0 && fFileHandler->GetSize() > 0) {
      TFileHandler *fh;
      TIter next(fFileHandler);

      while (fh = (TFileHandler*) next()) {
          int fd = fh->GetFd();
          if (fd <= fMaxrfd && fReadready.IsSet(fd)) {
              fReadready.Clr(fd);
              if (fh->ReadNotify())
                  return;
          }
          if (fd <= fMaxwfd && fWriteready.IsSet(fd)) {
              fWriteready.Clr(fd);
              if (fh->WriteNotify())
                  return;
          }
      }
  }
  fNfd = 0;
  fReadready.Zero();
  fWriteready.Zero();

  // check synchronous signals
  if (fSigcnt > 0 && fSignalHandler->GetSize() > 0)
      if (CheckSignals(kTRUE))
      return;
  fSigcnt = 0;
  fSignals.Zero();

      // check synchronous timers
  if (fTimers && fTimers->GetSize() > 0)
      if (DispatchTimers(kTRUE))
      return;

  // nothing ready, so setup select call
  fReadready  = fReadmask;
  fWriteready = fWritemask;
  int mxfd = TMath::Max(fMaxrfd, fMaxwfd) + 1;
  fNfd = WinNTSelect(mxfd, &fReadready, &fWriteready, NextTimeOut(kTRUE));
  if (fNfd < 0 && fNfd != -2) {
      int fd, rc;
      TFdSet t;
      for (fd = 0; fd < mxfd; fd++) {
          t.Set(fd);
          if (fReadmask.IsSet(fd)) {
              rc = WinNTSelect(fd+1, &t, 0, 0);
              if (rc < 0 && rc != -2) {
                  fprintf(stderr, "select: read error on %d\n", fd);
                  fReadmask.Clr(fd);
              }
          }
          if (fWritemask.IsSet(fd)) {
              rc = WinNTSelect(fd+1, &t, 0, 0);
              if (rc < 0 && rc != -2) {
                  fprintf(stderr, "select: write error on %d\n", fd);
                  fWritemask.Clr(fd);
              }
          }
 x         t.Clr(fd);
      }
  }
#endif

}

//---- handling of system events -----------------------------------------------

//______________________________________________________________________________
void TWinNTSystem::DispatchSignals(ESignals sig)
{
#ifndef WIN32
   // Handle and dispatch signals.

   switch (sig) {
   case kSigAlarm:
      //DispatchTimers();  rdm
      break;
   case kSigChild:
      CheckChilds();
      return;
   case kSigBus:
   case kSigSegmentationViolation:
   case kSigIllegalInstruction:
   case kSigFloatingException:
      Printf("\n *** Break *** %s", WinNTSigname(sig));
#ifdef R__HPUX
      Printf("");
      U_STACK_TRACE();
      Printf("");
#endif
      if (TROOT::Initialized())
         Throw(sig);
      Abort(-1);
      break;
   case kSigSystem:
   case kSigPipe:
      Printf("\n *** Break *** %s", WinNTSigname(sig));
      break;
   default:
      fSignals.Set(sig);
      fSigcnt++;
      break;
   }

   // check a-synchronous signals
   if (fSigcnt && fSignalHandler->GetSize() > 0)
      CheckSignals(kFALSE);
#else
      if (TROOT::Initialized())
         Throw(sig);
      Abort(-1);
#endif
}

//______________________________________________________________________________
Bool_t TWinNTSystem::CheckSignals(Bool_t sync)
{
   // Check if some signals were raised and call their Notify() member.

   TSignalHandler *sh;
   {
      TIter next(fSignalHandler);

      while (sh = (TSignalHandler*)next()) {
         if (sync == sh->IsSync()) {
            ESignals sig = sh->GetSignal();
            if (fSignals.IsSet(sig)) {
               fSignals.Clr(sig);
               fSigcnt--;
               break;
            }
         }
      }
   }
   if (sh) {
      sh->Notify();
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TWinNTSystem::CheckChilds()
{
   // Check if childs have finished.

#if 0  //rdm
   int pid;
   while ((pid = WinNTWaitchild()) > 0) {
      Iter next(zombieHandler);
      register WinNTPtty *pty;
      while (pty = (WinNTPtty*) next())
         if (pty->GetPid() == pid) {
            zombieHandler->RemovePtr(pty);
            pty->DiedNotify();
         }
   }
#endif
}

//---- Directories -------------------------------------------------------------

//______________________________________________________________________________
int  TWinNTSystem::MakeDirectory(const char *name)
{
  // Make a WinNT file system directory.
  // Make a Unix file system directory. Returns 0 in case of success and
  // -1 if the directory could not be created.

#ifdef WATCOM
//*-* It must be as follows
   if (!name) return 0;
   return mkdir(name);
#else
//*-*  But to be in line with TUnixSystem I did like this
   if (!name) return 0;
   return _mkdir(name);
#endif
}

//______________________________________________________________________________
void TWinNTSystem::FreeDirectory(void *dirp)
{
   // Close a WinNT file system directory.

   if (dirp)
      ::FindClose(dirp);
}

//______________________________________________________________________________
const char *TWinNTSystem::GetDirEntry(void *dirp)
{
// Returns the next directory entry.

   if (dirp)
   {
     HANDLE SearchFile = (HANDLE)dirp;
     if (FindNextFile(SearchFile,&fFindFileData))
       return (const char *)(fFindFileData.cFileName);
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TWinNTSystem::ChangeDirectory(const char *path)
{
   // Change directory.

   Bool_t ret = (Bool_t) (::chdir(path) == 0);
   if (fWdpath != "")
      fWdpath = "";   // invalidate path cache
   return ret;
}

//______________________________________________________________________________
void *TWinNTSystem::OpenDirectory(const char *dir)
{
   // Open a directory. Returns 0 if directory does not exist.

   struct stat finfo;

   if (stat(dir, &finfo) < 0)
      return 0;

   if (finfo.st_mode & S_IFDIR) {

     char *entry = new char[strlen(dir)+3];
     strcpy(entry,dir);
     if (!(entry[strlen(dir)] == '/' || entry[strlen(dir)] == '\\' ))
         strcat(entry,"\\");
     strcat(entry,"*");

     HANDLE SearchFile;
     SearchFile = FindFirstFile(entry,&fFindFileData);
     if (SearchFile == INVALID_HANDLE_VALUE){
       ((TWinNTSystem *)gSystem)->Error( "Unable to find' for reading:", entry);
       delete [] entry;
       return 0;
     }
     delete [] entry;
     return SearchFile;
   } else
           return 0;

//   return  WinNTOpendir(name);
}

//______________________________________________________________________________
const char *TWinNTSystem::WorkingDirectory()
{
// Return the working directory for the default drive
  return WorkingDirectory('\0');
}
//______________________________________________________________________________
const char *TWinNTSystem::WorkingDirectory(char driveletter)
{
////////////////////////////////////////////////////////////////////////////////
//  Return working directory for the selected drive                           //
//  driveletter == 0 means return the working durectory for the default drive //
////////////////////////////////////////////////////////////////////////////////

   char *wdpath = 0;
   char drive = driveletter ? toupper( driveletter ) - 'A' + 1 : 0;

   if (fWdpath != "" )
          return fWdpath;

   if (!(wdpath = _getdcwd( (int)drive, wdpath, kMAXPATHLEN))) {
      free(wdpath);
      Warning("WorkingDirectory", "getcwd() failed");
      return 0;
   }
   fWdpath = wdpath;
   free(wdpath);
   return fWdpath;
}

//______________________________________________________________________________
const char *TWinNTSystem::HomeDirectory(const Char_t *userName)
{
   // Return the user's home directory.

   return WinNTHomedirectory(userName);
}

//---- Paths & Files -----------------------------------------------------------

//______________________________________________________________________________
const char *TWinNTSystem::DirName(const char *pathname)
{
   // Return the directory name in pathname. DirName of c:/user/root is /user.
   // It creates output with 'new char []' operator. Returned string has to
   // be deleted.

  // Delete old buffer
  if (fDirNameBuffer) {
    // delete [] fDirNameBuffer;
    fDirNameBuffer = 0;
  }
//*-*
//*-* Create a buffer to keep the path name
//*-*
   if (pathname) {
     if (strchr(pathname, '/') || strchr(pathname, '\\'))
     {
       char *rslash = strrchr(pathname, '/');
       char *bslash = strrchr(pathname, '\\');
       char *r = max(rslash, bslash);
       const char *ptr = pathname;
       while (ptr <= r) {
         if (r == ":")
         {
           // Windows path may contain a drive letter
           // For NTFS ":" may be a "stream" delimiter as well
           pathname =  ptr + 1;
           break;
         }
         ptr++;
       }
       int len =  r - pathname;
       if (len>0)
       {
         fDirNameBuffer = new char[len+1];
         memcpy(fDirNameBuffer,pathname,len);
         fDirNameBuffer[len] = 0;
       }
     }
   }
   if (!fDirNameBuffer) {
        fDirNameBuffer = new char[1];
       *fDirNameBuffer = '\0'; // Set the empty default response
   }
   return fDirNameBuffer;
}

//______________________________________________________________________________
const char TWinNTSystem::DriveName(const char *pathname)
{
   ////////////////////////////////////////////////////////////////////////////
   // Return the drive letter in pathname. DriveName of 'c:/user/root' is 'c'//
   //   Input:                                                               //
   //      pathname - the string containing file name                        //
   //   Return:                                                              //
   //     = Letter presenting the drive letter in the file name              //
   //     = The current drive if the pathname has no drive assigment         //
   //     = 0 if pathname is an empty string  or uses UNC syntax             //
   //   Note:                                                                //
   //      It doesn't chech whether pathname presents the 'real filename     //
   //      This subroutine looks for 'single letter' is follows with a ':'   //
   ////////////////////////////////////////////////////////////////////////////
   if (!pathname)    return 0;
   if (!pathname[0]) return 0;

   const char *lpchar;
   lpchar = pathname;
//*-*  Skip blanks
   while(*lpchar == ' ') lpchar++;
   if (isalpha((int)*lpchar) && *(lpchar+1) == ':')
       return *lpchar;
//*-* Test UNC syntax
   if ( (*lpchar == '\\' || *lpchar == '/' ) &&
        (*(lpchar+1) == '\\' || *(lpchar+1) == '/')
      ) return 0;
//*-* return the current drive
   return DriveName(WorkingDirectory());
}

//______________________________________________________________________________
Bool_t TWinNTSystem::IsAbsoluteFileName(const char *dir)
{
   // Return true if dir is an absolute pathname.

   if (dir) {
      int idx = 0;
      if (strchr(dir,':')) idx = 2;
      return  (dir[idx] == '/' || dir[idx] == '\\');
   }
   return kFALSE;
}

//______________________________________________________________________________
const char *TWinNTSystem::UnixPathName(const char *name)
{
   // Convert a pathname to a unix pathname. E.g. form \user\root to /user/root.

   // General rules for applications creating names for directories and files or
   // processing names supplied by the user include the following:
   //
   //  �  Use any character in the current code page for a name, but do not use
   //     a path separator, a character in the range 0 through 31, or any character
   //     explicitly disallowed by the file system. A name can contain characters
   //     in the extended character set (128-255).
   //  �  Use the backslash (\), the forward slash (/), or both to separate
   //     components in a path. No other character is acceptable as a path separator.
   //  �  Use a period (.) as a directory component in a path to represent the
   //     current directory.
   //  �  Use two consecutive periods (..) as a directory component in a path to
   //     represent the parent of the current directory.
   //  �  Use a period (.) to separate components in a directory name or filename.
   //  �  Do not use the following characters in directory names or filenames, because
   //     they are reserved for Windows:
   //                      < > : " / \ |
   //  �  Do not use reserved words, such as aux, con, and prn, as filenames or
   //     directory names.
   //  �  Process a path as a null-terminated string. The maximum length for a path
   //     is given by MAX_PATH.
   //  �  Do not assume case sensitivity. Consider names such as OSCAR, Oscar, and
   //     oscar to be the same.


   char *CurrentChar = (char *)name;
   while (*CurrentChar != '\0') {
     if (*CurrentChar == '\\') *CurrentChar = '/';
     CurrentChar++;
   }
   return name;

}

//______________________________________________________________________________
Bool_t TWinNTSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the WinNT access(2) function.

   if (::_access(path, mode) == 0)
      return kFALSE;
   fLastErrorString = sys_errlist[GetErrno()];
   return kTRUE;
}

//______________________________________________________________________________
char *TWinNTSystem::ConcatFileName(const char *dir, const char *name)
{
   // Concatenate a directory and a file name. Returned string must be
   // deleted by user.

   Int_t ldir  = dir  ? strlen(dir) : 0;
   Int_t lname = name ? strlen(name) : 0;

   if (!lname) return StrDup(dir);

   char *buf = new char[ldir+lname+2];

   if (ldir) {
//*-*  Test whether the last symbol of the directory is a separator
          char last = dir[ldir-1];
      if (last == '/' || last == '\\' || last == ':')
         sprintf(buf, "%s%s", dir, name);
      else
         sprintf(buf, "%s/%s", dir, name);
   } else
      sprintf(buf, "/%s", name);
   return buf;
}

//______________________________________________________________________________
void TWinNTSystem::Rename(const char *f, const char *t)
{
   // Rename a file.

   ::rename(f, t);
   fLastErrorString = sys_errlist[GetErrno()];
}

//______________________________________________________________________________
int TWinNTSystem::GetPathInfo(const char *path, Long_t *id, Long_t *size,
                             Long_t *flags, Long_t *modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is (statbuf.st_dev << 24) + statbuf.st_ino
   // Size    is the file size
   // Flags   is file type: bit 1 set executable, bit 2 set directory,
   //                       bit 3 set regular file
   // Modtime is modification time

   return WinNTFilestat(path, id,  size, flags, modtime);
}

//______________________________________________________________________________
int TWinNTSystem::Link(const char *from, const char *to)
{
   // Create a link from file1 to file2.

//   return ::link(from, to);
        return 0;
}

//______________________________________________________________________________
int TWinNTSystem::Unlink(const char *name)
{
   // Unlink, i.e. remove, a file or directory.

   struct stat finfo;

   if (stat(name, &finfo) < 0)
      return -1;

//#ifdef __SC__
//   if (S_ISDIR(finfo.st_mode))
//#else
   if (finfo.st_mode & S_IFDIR)
//#endif
      return ::_rmdir(name);
   else
      return ::_unlink(name);
}

//______________________________________________________________________________
int TWinNTSystem::SetNonBlock(int fd)
{
   // Make descriptor fd non-blocking.

   WinNTNonblock(fd);
   return 0;
}

//---- expand the metacharacters as in the shell -------------------------------

// expand the metacharacters as in the shell

static char
   *shellMeta      = "~*[]{}?$%",
   *shellStuff     = "(){}<>\"'",
   shellEscape     = '\\';

//______________________________________________________________________________
Bool_t TWinNTSystem::ExpandPathName(TString &patbuf0)
{
   // Expand a pathname getting rid of special shell characaters like ~.$, etc.

   const char *patbuf = (const char *)patbuf0;
   const char *hd, *p;
   char   *cmd = 0;
   char    stuffedPat[kMAXPATHLEN], name[70];
   char  *q;
   int    ch, i;

   // skip leading blanks
   while (*patbuf == ' ')
      patbuf++;

   // any shell meta characters ?
   for (p = patbuf; *p; p++)
      if (strchr(shellMeta, *p))
         goto needshell;

   return kFALSE;

needshell:


   // escape shell quote characters
   //  EscChar(patbuf, stuffedPat, sizeof(stuffedPat), shellStuff, shellEscape);
     patbuf0 = ExpandFileName(patbuf0.Data());
     Int_t lbuf = ExpandEnvironmentStrings(
                                 patbuf0.Data(), // pointer to string with environment variables
                                 cmd,            // pointer to string with expanded environment variables
                                 0               // maximum characters in expanded string
                              );
      if (lbuf > 0) {
          cmd = new Char_t[lbuf+1];
          ExpandEnvironmentStrings(
                                 patbuf0.Data(), // pointer to string with environment variables
                                 cmd,            // pointer to string with expanded environment variables
                                 lbuf            // maximum characters in expanded string
                              );
          patbuf0 = cmd;
          return kFALSE;
      }

      return kTRUE;
}

//______________________________________________________________________________
char *TWinNTSystem::ExpandPathName(const char *path)
{
   // Expand a pathname getting rid of special shell characaters like ~.$, etc.
   // User must delete returned string.

   TString patbuf = path;
   if (ExpandPathName(patbuf))
      return 0;
   return StrDup(patbuf.Data());
}

//---- environment manipulation ------------------------------------------------

//______________________________________________________________________________
void TWinNTSystem::Setenv(const char *name, const char *value)
{
   // Set environment variable.

   ::_putenv(Form("%s=%s", name, value));
}

//______________________________________________________________________________
const char *TWinNTSystem::Getenv(const char *name)
{
   // Get environment variable.

  const char *env = ::getenv(name);
  if (!env) {
     if (::_stricmp(name,"home")==0 )
        env = HomeDirectory();
     else if (::_stricmp(name,"rootsys")==0 )
        env = gRootDir;
  }
  return env;
}

//---- Processes ---------------------------------------------------------------

//______________________________________________________________________________
int TWinNTSystem::Exec(const char *shellcmd)
{
  //*-* Execute a command.

   return ::system(shellcmd);
}

//______________________________________________________________________________
FILE *TWinNTSystem::OpenPipe(const char *command, const char *mode)
{
   // Open a pipe.

  return ::_popen(command, mode);
}

//______________________________________________________________________________
int TWinNTSystem::ClosePipe(FILE *pipe)
{
   // Close the pipe.
  return ::_pclose(pipe);
}

//______________________________________________________________________________
int TWinNTSystem::GetPid()
{
   // Get process id.

   return ::getpid();
}

//______________________________________________________________________________
HANDLE TWinNTSystem::GetProcess()
{
  // Get current process handle
  return fhProcess;
}
//______________________________________________________________________________
void TWinNTSystem::Exit(int code, Bool_t mode)
{
   // Exit the application.

   if (mode)
      ::exit(code);
   else
      ::_exit(code);
}

//______________________________________________________________________________
void TWinNTSystem::Abort(int)
{
   // Abort the application.

   ::abort();
}

//---- dynamic loading and linking ---------------------------------------------

//______________________________________________________________________________
const char *TWinNTSystem::GetDynamicPath()
{
   // Get shared library search path.

   static const char *dynpath = 0;

   if (dynpath == 0) {
      dynpath = gEnv->GetValue("Root.DynamicPath", (char*)0);
      if (dynpath == 0)
         dynpath = StrDup(Form("%s;%s/bin;%s,", gProgPath,gRootDir,gSystem->Getenv("PATH")));
   }
   return dynpath;
}

//______________________________________________________________________________
char *TWinNTSystem::Which(const char *search, const char *infile, EAccessMode mode)
{
   // Find location of file in a search path.
   // User must delete returned string.

   static char name[kMAXPATHLEN];
   char *lpFilePart = 0;
   char *found = 0;

//* Expand parameters

   char *exinfile = gSystem->ExpandPathName(infile);
//  Check whether this infile has the absolute path first
   if (IsAbsoluteFileName(exinfile) ) {
     found = exinfile;
   }
   else {
     char *exsearch = gSystem->ExpandPathName(search);

 //*-*  Check access

    if (SearchPath( exsearch,exinfile,NULL,kMAXPATHLEN,name,&lpFilePart)
                    && access(name, mode) == 0) {
        if (gEnv->GetValue("Root.ShowPath", 0))
           Printf("Which: %s = %s", infile, name);
        found =  StrDup(name);
    }
    delete [] exsearch;
    delete [] exinfile;
   }

   if (found  && AccessPathName(found, mode))
   {
     delete [] found;
     found = 0;
   }
   return found;
}

//______________________________________________________________________________
int TWinNTSystem::Load(const char *module, const char *entry, Bool_t system)
{
   // Load a shared library. On successful loading return 0.

#ifdef NOCINT
   int i = WinNTDynLoad(module);
   if (!entry || !strlen(entry)) return i;

   Func_t f = WinNTDynFindSymbol(module, entry);
   if (f) return 0;
   return 1;
#else
   return TSystem::Load(module, entry, system);
#endif
}


//______________________________________________________________________________
char *TWinNTSystem::DynamicPathName(const char *lib, Bool_t quiet)
{
   // Returns the path of a dynamic library (searches for library in the
   // dynamic library search path). If no file name extension is provided
   // it tries .DLL. Returned string must be deleted.

   char *name;

   int len = strlen(lib);
   if (len > 4 && (!stricmp(lib+len-4, ".dll")))
      name = gSystem->Which(GetDynamicPath(), lib, kReadPermission);
   else {
      name = Form("%s.dll", lib);
      name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
   }

   if (!name && !quiet)
      Error("DynamicPathName",
      "dll does not exist or wrong file extension (.dll)", lib);

#if 0
    if (!lib || !strlen(lib)) return 0;
    char *name = 0;

   // Append an extention if any

   if (!strchr(lib, '.')) {
      name = Form("%s.dll", lib);
      name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
   }
   else {
       int len = strlen(lib);
       if (len > 4 && !(strnicmp(lib+len-4, ".dll",4)))
       {
           name = gSystem->Which(GetDynamicPath(), lib, kReadPermission);
       }
       else {
           ::Error("TWinNTSystem::DynamicPathName",
               "DLL library must have extension: .dll", lib);
           name = 0;
       }
   }
#endif
#if 0
   if (name)
   {
     char driveletter = DriveName(name);
     char *dirname    = (char *)DirName(name);
     delete [] name;  // allocated by Which()
     if (driveletter) {
        name = Form("%c:%s",driveletter,dirname);
        delete [] dirname;
        return StrDup(name);
     } else
        return dirname;
   }
   return 0;
#endif
   return name;
}

//______________________________________________________________________________
Func_t TWinNTSystem::DynFindSymbol(const char *module, const char *entry)
{
#ifdef NOCINT
   return WinNTDynFindSymbol(module,entry);
#else
   return TSystem::DynFindSymbol("*", entry);
#endif
}

//______________________________________________________________________________
void TWinNTSystem::Unload(const char *module)
{
   // Unload a shared library.

#ifdef NOCINT
   WinNTDynUnload(module);
#else
   if (module) { }
   // should call CINT unload file here, but does not work for sl's yet.
   Warning("Unload", "CINT does not support unloading shared libs");
#endif
}

//______________________________________________________________________________
void TWinNTSystem::ListSymbols(const char *module, const char *regexp)
{
   // List symbols in a shared library.

   WinNTDynListSymbols(module, regexp);
}

//______________________________________________________________________________
void TWinNTSystem::ListLibraries(const char *regexp)
{
   // List all loaded shared libraries.

   //WinNTDynListLibs(regexp);
   TSystem::ListLibraries(regexp);
}

//______________________________________________________________________________
const char *TWinNTSystem::GetLibraries(const char *regexp, const char *options)
{
   // Return a space separated list of loaded shared libraries.
   // This list is of a format suitable for a linker, i.e it may contain
   // -Lpathname and/or -lNameOfLib.
   // Option can be any of:
   //   S: shared libraries loaded at the start of the executable, because
   //      they were specified on the link line.
   //   D: shared libraries dynamically loaded after the start of the program.
   //   L: list the .LIB rather than the .DLL (this is intended for linking)
   //      [This options is not the default] 
   
   TString libs( TSystem::GetLibraries( regexp, options ) );
   TString ntlibs;
   TString opt = options;

   if ( (opt.First('L')!=kNPOS) ) {
      TRegexp separator("[^ \\t\\s]+");
      TRegexp user_dll("*.dll", kTRUE);
      TRegexp user_lib("*.lib", kTRUE);
      TString s;
      Ssiz_t start, index, end;
      start = index = end = 0;
 
      while ((start < libs.Length()) && (index != kNPOS)) {
	index = libs.Index(separator,&end,start);
	if (index >= 0) {
	  s = libs(index,end);
	  if (s.Index(user_dll) != kNPOS) {
	    s.ReplaceAll(".dll",".lib");
	    if ( GetPathInfo( s, 0, 0, 0, 0 ) != 0 ) {
	      s.Replace( 0, s.Last('/')+1, 0, 0);
	      s.Replace( 0, s.Last('\\')+1, 0, 0);
	    }
	  } else if (s.Index(user_lib) != kNPOS) {
	    if ( GetPathInfo( s, 0, 0, 0, 0 ) != 0 ) {
	      s.Replace( 0, s.Last('/')+1, 0, 0);
	      s.Replace( 0, s.Last('\\')+1, 0, 0);
	    }	    
	  }
	  if (!fListLibs.IsNull())
	    ntlibs.Append(" ");
	  ntlibs.Append(s);
	}
	start += end+1;
      }
   } else 
     ntlibs = libs;
   
   fListLibs = ntlibs;
   return fListLibs;
}


//---- Time & Date -------------------------------------------------------------

//______________________________________________________________________________
void TWinNTSystem::AddTimer(TTimer *ti)
{
    if (ti)
    {
        TSystem::AddTimer(ti);
 //       if (ti->IsAsync())
        {
          if (!fWin32Timer) fWin32Timer = new TWin32Timer;
          fWin32Timer->CreateTimer(ti);
          if (!ti->GetTimerID())
                RemoveTimer(ti);
        }
    }
}
//______________________________________________________________________________
TTimer *TWinNTSystem::RemoveTimer(TTimer *ti)
{
    if (ti && fWin32Timer ) {
       fWin32Timer->KillTimer(ti);
       return TSystem::RemoveTimer(ti);
    }
    return 0;
}

//______________________________________________________________________________
Bool_t TWinNTSystem::DispatchSynchTimers()
{
   // Handle and dispatch timers. If mode = kTRUE dispatch synchronous
   // timers else a-synchronous timers.
   if (!fTimers) return kFALSE;

   fInsideNotify = kTRUE;

   TOrdCollectionIter it((TOrdCollection*)fTimers);
   TTimer *t;
   Bool_t  timedout = kFALSE;

   while ((t = (TTimer *) it.Next()))
      if (t->IsSync()) {
         TTime now = Now();
         now += TTime(kItimerResolution);
         if (t->CheckTimer(now)) timedout = kTRUE;
      }
   fInsideNotify = kFALSE;
   return timedout;
}

const Double_t gTicks = 1.0e-7;
//______________________________________________________________________________
Double_t TWinNTSystem::GetRealTime(){
#if defined(R__MAC)
   return(Double_t)clock() / gTicks;
#elif defined(R__UNIX)
   struct tms cpt;
   return (Double_t)times(&cpt) / gTicks;
#elif defined(R__VMS)
  return(Double_t)clock()/gTicks;
#elif defined(WIN32)
  union     {FILETIME ftFileTime;
             __int64  ftInt64;
            } ftRealTime; // time the process has spent in kernel mode
  SYSTEMTIME st;
  GetSystemTime(&st);
  SystemTimeToFileTime(&st,&ftRealTime.ftFileTime);
  return (Double_t)ftRealTime.ftInt64 * gTicks;
#endif
}

//______________________________________________________________________________
Double_t TWinNTSystem::GetCPUTime(){
#if defined(R__MAC)
   return(Double_t)clock() / gTicks;
#elif defined(R__UNIX)
   struct tms cpt;
   times(&cpt);
   return (Double_t)(cpt.tms_utime+cpt.tms_stime) / gTicks;
#elif defined(R__VMS)
   return(Double_t)clock()/gTicks;
#elif defined(WIN32)

  OSVERSIONINFO OsVersionInfo;

//*-*         Value                      Platform
//*-*  ----------------------------------------------------
//*-*  VER_PLATFORM_WIN32s          Win32s on Windows 3.1
//*-*  VER_PLATFORM_WIN32_WINDOWS       Win32 on Windows 95
//*-*  VER_PLATFORM_WIN32_NT            Windows NT
//*-*
  OsVersionInfo.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&OsVersionInfo);
  if (OsVersionInfo.dwPlatformId == VER_PLATFORM_WIN32_NT) {
    DWORD       ret;
    FILETIME    ftCreate,       // when the process was created
                ftExit;         // when the process exited

    union     {FILETIME ftFileTime;
               __int64  ftInt64;
              } ftKernel; // time the process has spent in kernel mode

    union     {FILETIME ftFileTime;
               __int64  ftInt64;
              } ftUser;   // time the process has spent in user mode

    HANDLE hProcess = GetCurrentProcess();
    ret = GetProcessTimes (hProcess, &ftCreate, &ftExit,
                                     &ftKernel.ftFileTime,
                                     &ftUser.ftFileTime);
    if (ret != TRUE){
      ret = GetLastError ();
      ::Error ("GetCPUTime", " Error on GetProcessTimes 0x%lx", (int)ret);
    }

    /*
     * Process times are returned in a 64-bit structure, as the number of
     * 100 nanosecond ticks since 1 January 1601.  User mode and kernel mode
     * times for this process are in separate 64-bit structures.
     * To convert to floating point seconds, we will:
     *
     *          Convert sum of high 32-bit quantities to 64-bit int
     */

      return (Double_t) (ftKernel.ftInt64 + ftUser.ftInt64) * gTicks;
  }
  else
      return GetRealTime();

#endif
}

//______________________________________________________________________________
TTime TWinNTSystem::Now()
{
   // Return current time.

   return Long_t(GetRealTime()*1000.0);
 // WinNTNow();
}

#if 0 //rdm

//______________________________________________________________________________
void TWinNTSystem::AddTimer(TTimer *ti)
{
   // Add timer to list of system timers.

   TSystem::AddTimer(ti);
   if (!fInsideNotify)  // we are not inside notify
      if (ti->IsAsync() && (ti == fAsyncTimers))
         WinNTSetitimer(NextTimeout(fAsyncTimers));
}

//______________________________________________________________________________
Bool_t TWinNTSystem::RemoveTimer(TTimer *ti)
{
   // Remove timer from list of system timers.

   Bool_t rc = TSystem::RemoveTimer(ti);
   if (ti->IsAsync() && fAsyncTimers == 0)
      WinNTSetitimer(-1);
   return rc;
}
#endif
//______________________________________________________________________________
void TWinNTSystem::Sleep(UInt_t milliSec)
{
   // Sleep milliSec milli seconds.
//*-* The Sleep function suspends the execution of the CURRENT THREAD for
//*-* a specified interval.
   ::Sleep(milliSec);
}

//---- RPC ---------------------------------------------------------------------
//______________________________________________________________________________
int TWinNTSystem::GetServiceByName(const char *servicename)
{
   // Get port # of internet service.

   struct servent *sp;

   if ((sp = getservbyname(servicename, kProtocolName)) == 0) {
      Error("GetServiceByName", "no service \"%s\" with protocol \"%s\"\n",
              servicename, kProtocolName);
      return -1;
   }
   return ntohs(sp->s_port);
}

//______________________________________________________________________________
char *TWinNTSystem::GetServiceByPort(int port)
{

   // Get name of internet service.

   struct servent *sp;

   if ((sp = getservbyport(port, kProtocolName)) == 0) {
      //::Error("GetServiceByPort", "no service \"%d\" with protocol \"%s\"",
      //        port, kProtocolName);
      return Form("%d", port);
   }
   return sp->s_name;
}
//______________________________________________________________________________
TInetAddress TWinNTSystem::GetHostByName(const char *hostname)
{
   // Get Internet Protocol (IP) address of host.

   struct hostent *host_ptr;
   struct in_addr  ad;
   const char     *host;
   int             type;
   UInt_t          addr;    // good for 4 byte addresses

   if ((addr = inet_addr(hostname)) != INADDR_NONE) {
      type = AF_INET;
      if ((host_ptr = gethostbyaddr((const char *)&addr,
                                    sizeof(addr), AF_INET)))
         host = host_ptr->h_name;
      else
         host = "UnNamedHost";
   } else if ((host_ptr = gethostbyname(hostname))) {
      // Check the address type for an internet host
      if (host_ptr->h_addrtype != AF_INET) {
         Error("GetHostByName", "%s is not an internet host\n", hostname);
         return TInetAddress();
      }
      memcpy(&addr, host_ptr->h_addr, host_ptr->h_length);
      host = host_ptr->h_name;
      type = host_ptr->h_addrtype;
   } else {
      if (gDebug > 0) Error("GetHostByName", "unknown host %s", hostname);
      return TInetAddress(hostname, 0, -1);
   }

   return TInetAddress(host, ntohl(addr), type);
}

//______________________________________________________________________________
TInetAddress TWinNTSystem::GetPeerName(int socket)
{
   // Get Internet Protocol (IP) address of remote host and port #.
   SOCKET sock = socket;
   struct sockaddr_in addr;
#if defined(R__AIX) && defined(_AIX41)
   size_t len = sizeof(addr);
#else
   int len = sizeof(addr);
#endif

   if (getpeername(sock, (struct sockaddr *)&addr, &len) == SOCKET_ERROR) {
      SysError("GetPeerName", "getpeername");
      return TInetAddress();
   }

   struct hostent *host_ptr;
   const char *hostname;
   int         family;
   UInt_t      iaddr;

   if ((host_ptr = gethostbyaddr((const char *)&addr.sin_addr,
                                 sizeof(addr.sin_addr), AF_INET))) {
      memcpy(&iaddr, host_ptr->h_addr, host_ptr->h_length);
      hostname = host_ptr->h_name;
      family   = host_ptr->h_addrtype;
   } else {
      memcpy(&iaddr, &addr.sin_addr, sizeof(addr.sin_addr));
      hostname = "????";
      family   = AF_INET;
   }

   return TInetAddress(hostname, ntohl(iaddr), family, ntohs(addr.sin_port));
}
//______________________________________________________________________________
TInetAddress TWinNTSystem::GetSockName(int socket)
{
   // Get Internet Protocol (IP) address of host and port #.

   SOCKET sock = socket;
   struct sockaddr_in addr;
#if defined(R__AIX) && defined(_AIX41)
   size_t len = sizeof(addr);
#else
   int len = sizeof(addr);
#endif

   if (getsockname(sock, (struct sockaddr *)&addr, &len) == SOCKET_ERROR) {
      SysError("GetSockName", "getsockname");
      return TInetAddress();
   }

   struct hostent *host_ptr;
   const char *hostname;
   int         family;
   UInt_t      iaddr;

   if ((host_ptr = gethostbyaddr((const char *)&addr.sin_addr,
                                 sizeof(addr.sin_addr), AF_INET))) {
      memcpy(&iaddr, host_ptr->h_addr, host_ptr->h_length);
      hostname = host_ptr->h_name;
      family   = host_ptr->h_addrtype;
   } else {
      memcpy(&iaddr, &addr.sin_addr, sizeof(addr.sin_addr));
      hostname = "????";
      family   = AF_INET;
   }

   return TInetAddress(hostname, ntohl(iaddr), family, ntohs(addr.sin_port));

}
//______________________________________________________________________________
int          TWinNTSystem::AnnounceUnixService(int port, int backlog)
{
   // Announce unix domain service.
   return WinNTWinNTService(port, backlog);
 }
//______________________________________________________________________________
void      TWinNTSystem::CloseConnection(int socket, Bool_t force)
{
   // Close socket.

   if (socket == -1) return;
   SOCKET sock = socket;

   if (force)
      ::shutdown(sock, 2);

   while (::closesocket(sock) == SOCKET_ERROR && WSAGetLastError() == WSAEINTR)
   ResetErrno();
}
//______________________________________________________________________________
int TWinNTSystem::RecvBuf(int sock, void *buf, int length)
{
   // Receive a buffer headed by a length indicator. Lenght is the size of
   // the buffer. Returns the number of bytes received in buf or -1 in
   // case of error.

   Int_t header;

   if (WinNTRecv(sock, &header, sizeof(header), 0) > 0) {
      int count = ntohl(header);

      if (count > length) {
         Error("RecvBuf", "record header exceeds buffer size");
         return -1;
      } else if (count > 0) {
         if (WinNTRecv(sock, buf, count, 0) < 0) {
            Error("RecvBuf", "cannot receive buffer");
            return -1;
         }
      }
      return count;
   }
   return -1;
}

//______________________________________________________________________________
int TWinNTSystem::SendBuf(int sock, const void *buf, int length)
{
   // Send a buffer headed by a length indicator. Returns length of sent buffer
   // or -1 in case of error.

   Int_t header = htonl(length);

   if (WinNTSend(sock, &header, sizeof(header), 0) < 0) {
      Error("SendBuf", "cannot send header");
      return -1;
   }
   if (length > 0) {
      if (WinNTSend(sock, buf, length, 0) < 0) {
         Error("SendBuf", "cannot send buffer");
         return -1;
      }
   }
   return length;
}

//______________________________________________________________________________
int TWinNTSystem::RecvRaw(int sock, void *buf, int length, int opt)
{
   // Receive exactly length bytes into buffer. Use opt to receive out-of-band
   // data or to have a peek at what is in the buffer (see TSocket). Buffer
   // must be able to store at least lenght bytes. Returns the number of
   // bytes received (can be 0 if other side of connection was closed) or -1
   // in case of error, -2 in case of MSG_OOB and errno == EWOULDBLOCK, -3
   // in case of MSG_OOB and errno == EINVAL and -4 in case of kNoBlock and
   // errno == EWOULDBLOCK.

   int flag;

   switch (opt) {
   case kDefault:
      flag = 0;
      break;
   case kOob:
      flag = MSG_OOB;
      break;
   case kPeek:
      flag = MSG_PEEK;
      break;
   default:
      flag = 0;
      break;
   }

   int n;
   if ((n = WinNTRecv(sock, buf, length, flag)) <= 0) {
      if (n == -1)
         Error("RecvRaw", "cannot receive buffer");
      return n;
   }
   return length;
}

//______________________________________________________________________________
int TWinNTSystem::SendRaw(int sock, const void *buf, int length, int opt)
{
   // Send exactly length bytes from buffer. Use opt to send out-of-band
   // data (see TSocket). Returns the number of bytes sent or -1 in case of
   // error.

   int flag;

   switch (opt) {
   case kDefault:
      flag = 0;
      break;
   case kOob:
      flag = MSG_OOB;
      break;
   case kPeek:            // receive only option (see RecvRaw)
   default:
      flag = 0;
      break;
   }

   if (WinNTSend(sock, buf, length, flag) < 0) {
      Error("SendRaw", "cannot send buffer");
      return -1;
   }
   return length;
}

//______________________________________________________________________________
int  TWinNTSystem::SetSockOpt(int socket, int opt, int value)
{
   // Set socket option.

   u_long val = value;
   if (socket == -1) return -1;
   SOCKET sock = socket;

   switch (opt) {
   case kSendBuffer:
      if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(SO_SNDBUF)");
         return -1;
      }
      break;
   case kRecvBuffer:
      if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(SO_RCVBUF)");
         return -1;
      }
      break;
   case kOobInline:
      if (setsockopt(sock, SOL_SOCKET, SO_OOBINLINE, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(SO_OOBINLINE)");
         return -1;
      }
      break;
   case kKeepAlive:
      if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(SO_KEEPALIVE)");
         return -1;
      }
      break;
   case kReuseAddr:
      if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(SO_REUSEADDR)");
         return -1;
      }
      break;
   case kNoDelay:
      if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&val, sizeof(val)) == SOCKET_ERROR) {
         SysError("SetSockOpt", "setsockopt(TCP_NODELAY)");
         return -1;
      }
      break;
//*-*  Must be checked
   case kNoBlock:
      if (ioctlsocket(sock, FIONBIO, &val) == SOCKET_ERROR) {
         SysError("SetSockOpt", "ioctl(FIONBIO)");
         return -1;
      }
      break;
#if 0
   case kProcessGroup:
      if (ioctl(sock, SIOCSPGRP, &val) == -1) {
         SysError("SetSockOpt", "ioctl(SIOCSPGRP)");
         return -1;
      }
      break;
#endif
   kAtMark:       // read-only option (see GetSockOpt)
   kBytesToRead:  // read-only option
   default:
      Error("SetSockOpt", "illegal option (%d)", opt);
      return -1;
      break;
   }
   return 0;
}
//______________________________________________________________________________
int TWinNTSystem::GetSockOpt(int socket, int opt, int *val)
{
   // Get socket option.
   if (socket == -1) return -1;
   SOCKET sock = socket;

   int optlen = sizeof(*val);

   switch (opt) {
   case kSendBuffer:
      if (getsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(SO_SNDBUF)");
         return -1;
      }
      break;
   case kRecvBuffer:
      if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(SO_RCVBUF)");
         return -1;
      }
      break;
   case kOobInline:
      if (getsockopt(sock, SOL_SOCKET, SO_OOBINLINE, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(SO_OOBINLINE)");
         return -1;
      }
      break;
   case kKeepAlive:
      if (getsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(SO_KEEPALIVE)");
         return -1;
      }
      break;
   case kReuseAddr:
      if (getsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(SO_REUSEADDR)");
         return -1;
      }
      break;
   case kNoDelay:
      if (getsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)val, &optlen) == SOCKET_ERROR) {
         SysError("GetSockOpt", "getsockopt(TCP_NODELAY)");
         return -1;
      }
      break;
   case kNoBlock:
       {
           int flg = 0;
//*-*  Get file status flags and access modes
//      if ((flg = fcntl(sock, F_GETFL, 0)) == SOCKET_ERROR) {NVALID_SOCKET
           if (sock == INVALID_SOCKET)
               SysError("GetSockOpt", "INVALID_SOCKET");
           return -1;
           *val = flg; //  & O_NDELAY;  It is not been defined for WIN32
       }
      break;
#if 0
   case kProcessGroup:
      if (ioctlsocket(sock, SIOCGPGRP, (u_long*)val) == SOCKET_ERROR) {
         SysError("GetSockOpt", "ioctl(SIOCGPGRP)");
         return -1;
      }
      break;
#endif
   case kAtMark:
      if (ioctlsocket(sock, SIOCATMARK, (u_long*)val) == SOCKET_ERROR) {
         SysError("GetSockOpt", "ioctl(SIOCATMARK)");
         return -1;
      }
      break;
   case kBytesToRead:
      if (ioctlsocket(sock, FIONREAD, (u_long*)val) == SOCKET_ERROR) {
         SysError("GetSockOpt", "ioctl(FIONREAD)");
         return -1;
      }
      break;
   default:
      Error("GetSockOpt", "illegal option (%d)", opt);
      *val = 0;
      return -1;
      break;
   }
   return 0;
}

//______________________________________________________________________________
int TWinNTSystem::ConnectService(const char *servername, int port)
{
   // Connect to service servicename on server servername.
   if (!strcmp(servername, "unix"))
   {
       printf(" Error don't know how to do UnixUnixConnact under WIN32 \n");
       return -1;
       // return UnixUnixConnect(port);
   }
   return WinNTTcpConnect(servername, port);
}

//______________________________________________________________________________
int TWinNTSystem::OpenConnection(const char *server, int port)
{
   // Open a connection to a service on a server. Try 3 times with an
   // interval of 1 second.

   for (int i = 0; i < 3; i++) {
      int fd = ConnectService(server, port);
      if (fd >= 0)
         return fd;
      Sleep(1000);
   }
   return -1;
}

//______________________________________________________________________________
int TWinNTSystem::AnnounceTcpService(int port, Bool_t reuse, int backlog)
{
   // Start TCP/IP service.

   return WinNTTcpService(port, reuse, backlog);

}


//______________________________________________________________________________
int TWinNTSystem::AcceptConnection(int socket)
{
   // Accept a connection. In case of an error return -1. In case
   // non-blocking I/O is enabled and no connections are available
   // return -2.

   int soc = -1;
   SOCKET sock = socket;

   while ((soc = ::accept(sock, 0, 0)) == INVALID_SOCKET && WSAGetLastError() == WSAEINTR)
      ResetErrno();

   if (soc == -1) {
      if (WSAGetLastError() == WSAEWOULDBLOCK)
         return -2;
      else {
         SysError("AcceptConnection", "accept");
         return -1;
      }
   }

   return soc;

}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Static Protected WinNT Interface functions.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//---- signals -----------------------------------------------------------------

static struct signal_map {
   int code;
   SigHandler_t handler;
   char *signame;
} signal_map[kMAXSIGNALS] = {   // the order of the signals should be identical
//   SIGBUS,   0, "bus error",    // to the one in SysEvtHandler.h
   SIGSEGV,  0, "segmentation violation",
//   SIGSYS,   0, "bad argument to system call",
//   SIGPIPE,  0, "write on a pipe with no one to read it",
   SIGILL,   0, "illegal instruction",
//   SIGQUIT,  0, "quit",
   SIGINT,   0, "interrupt",
//   SIGWINCH, 0, "window size change",
//   SIGALRM,  0, "alarm clock",
//   SIGCHLD,  0, "death of a child",
//   SIGURG,   0, "urgent data arrived on an I/O channel",
   SIGFPE,   0, "floating point exception"
//   SIGTERM,  0, "termination signal",
//   SIGUSR1,  0, "user-defined signal 1",
//   SIGUSR2,  0, "user-defined signal 2"
};

//______________________________________________________________________________
static void sighandler(int sig)
{
   // Call the signal handler associated with the signal.

   for (int i= 0; i < kMAXSIGNALS; i++) {
      if (signal_map[i].code == sig) {
         (*signal_map[i].handler)((ESignals)i);
         return;
      }
   }
}

//______________________________________________________________________________
void TWinNTSystem::WinNTSignal(ESignals sig, SigHandler_t handler)
{
   // Set a signal handler for a signal.

#ifndef WIN32
if (signal_map[sig].handler != handler) {
      struct sigaction sigact;

      signal_map[sig].handler = handler;

      sigact.sa_handler = sighandler;
      sigemptyset(&sigact.sa_mask);
#if defined(SA_INTERRUPT)       // SunOS
      sigact.sa_flags = SA_INTERRUPT;
#else
      sigact.sa_flags = 0;
#endif
      if (sigaction(signal_map[sig].code, &sigact, NULL) < 0)
         ::SysError("TWinNTSystem::WinNTSignal", "sigaction");
   }
#endif
}

//______________________________________________________________________________
char *TWinNTSystem::WinNTSigname(ESignals sig)
{
   // Return the signal name associated with a signal.

   return signal_map[sig].signame;
}

//---- time --------------------------------------------------------------------

//______________________________________________________________________________
long TWinNTSystem::WinNTNow()
{

   // Get current time in milliseconds since 0:00 Jan 1 1995.

   SYSTEMTIME  t; // SYSTEMTIME structure to receive the current system date and time.
   GetSystemTime(&t);
   return (t.wMinute*60 + t.wSecond) * 1000 + t.wMilliseconds;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTSetitimer(TTimer *ti)
{
   // Set interval timer to time-out in ms milliseconds.

#ifndef WIN32
   struct itimerval itv;
   itv.it_interval.tv_sec = itv.it_interval.tv_usec = 0;
   itv.it_value.tv_sec = itv.it_value.tv_usec = 0;
   if (ms >= 0) {
      itv.it_value.tv_sec  = ms / 1000;
      itv.it_value.tv_usec = (ms % 1000) * 1000;
   }
   return setitimer(ITIMER_REAL, &itv, 0);
#endif
        return 0;
}

//---- file descriptors --------------------------------------------------------

//______________________________________________________________________________
int TWinNTSystem::WinNTSelect(UInt_t nfds, TFdSet *readready, TFdSet *writeready,
                            Long_t timeout)
{
   // Wait for events on the file descriptors specified in the readready and
   // writeready masks or for timeout (in milliseconds) to occur.

//*-*  nfds  !!!!
//*-*  This argument is ignored and included only for the sake of compatibility.

   int retcode;

   if (timeout >= 0) {
      struct timeval tv;
      tv.tv_sec  = timeout / 1000;
      tv.tv_usec = (timeout % 1000) * 1000;
#if (defined(R__HPUX) && !defined(_XPG4_EXTENDED)) || defined(R__AIX)
      retcode = select(nfds, readready->GetBits(), writeready->GetBits(), 0, &tv);
#else
      retcode = select(nfds, (fd_set*)readready->GetBits(), (fd_set*)writeready->GetBits(), 0, &tv);
#endif
   } else {
#if (defined(R__HPUX) && !defined(_XPG4_EXTENDED)) || defined(R__AIX)
      retcode = select(nfds, readready->GetBits(), writeready->GetBits(), 0, 0);
#else
      retcode = select(nfds, (fd_set*)readready->GetBits(), (fd_set*)writeready->GetBits(), 0, 0);
#endif
   }
   if (retcode == SOCKET_ERROR) {
      if (WSAGetLastError() == WSAEINTR) {
         ResetErrno();  // errno is not self reseting
         return -2;
      }
      if (WSAGetLastError() == EBADF)
         return -3;
      return -1;
   }
   return retcode;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTNonblock(int fd)
{
   // Make a descriptor non-blocking.

   int val;

   if (fd < 0)
      return -1;

//   if ((val = fcntl(fd, F_GETFL, 0)) < 0) {
//      ::SysError("TWinNTSystem::WinNTNonblock", "fcntl F_GETFL");
//      return val;
//   }
//#if 1 || defined(R__SUN)
//   val |= O_NDELAY;
//#else
//   val |= O_NONBLOCK;
//#endif
//   if ((val = fcntl(fd, F_SETFL, val)) < 0) {
//     // ::SysError("TWinNTSystem::WinNTNonblock:", "fcntl F_SETFL");
//      return val;
//   }
   return 0;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTIoctl(int fd, int code, void *vp)
{
   // Wrapper for ioctl function.

   int rc;
   if (fd < 0)
      return -1;
   return -1;
#if 0
//   rc = ioctl(fd, code, vp);
   if (rc == -1)
      ::SysError("TWinNTSystem::WinNTIoctl", "ioctl");
   return rc;
#endif
}

//---- directories -------------------------------------------------------------


//______________________________________________________________________________
const char *TWinNTSystem::WinNTHomedirectory(const char *name)
{
   // Returns the user's home directory.

   static char mydir[kMAXPATHLEN]="./";
#ifndef WIN32
   static char path[kMAXPATHLEN], mydir[kMAXPATHLEN];
   struct passwd *pw;

   if (name) {
//      pw = getpwnam(name);
//        LookupAccountName();
//      if (pw) {
//         strncpy(path, pw->pw_dir, kMAXPATHLEN);
         return path;
      }
   } else {
      if (mydir[0])
         return mydir;
      pw = getpwuid(getuid());
      if (pw) {
         strncpy(mydir, pw->pw_dir, kMAXPATHLEN);
         return mydir;
      }
   }
#endif
   const char *h = 0;
   if (!(h = ::getenv("home"))) h = ::getenv("HOME");

   if (h) strcpy(mydir,h);
   else {
      // for Windows NT HOME might be defined as either $(HOMESHARE)/$(HOMEPATH)
      //                                         or     $(HOMEDRIVE)/$(HOMEPATH)
     h = ::getenv("HOMESHARE");
     if (!h)  h = ::getenv("HOMEDRIVE");
     if (h) {
         strcpy(mydir,h);
         h=::getenv("HOMEPATH");
         if(h) strcat(mydir,h);
     }
   }
   return mydir;
}


//---- files -------------------------------------------------------------------

//______________________________________________________________________________
int TWinNTSystem::WinNTFilestat(const char *path, Long_t *id, Long_t *size,
                              Long_t *flags, Long_t *modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is (statbuf.st_dev << 24) + statbuf.st_ino
   // Size    is the file size
   // Flags   is file type: bit 1 set executable, bit 2 set directory,
   //                       bit 3 set regular file
   // Modtime is modification time

   struct stat statbuf;
   if (id)      *id = 0;
   if (size)    *size = 0;
   if (flags)   *flags = 0;
   if (modtime) *modtime = 0;
   if (path != 0 && stat(path, &statbuf) >= 0) {
      if (id)
         *id = (statbuf.st_dev << 24) + statbuf.st_ino;
      if (size)
         *size = statbuf.st_size;
      if (modtime)
         *modtime = statbuf.st_mtime;
      if (flags) {
         if (statbuf.st_mode & ((S_IEXEC)|(S_IEXEC>>3)|(S_IEXEC>>6)))
            *flags |= 1;
         if ((statbuf.st_mode & S_IFMT) == S_IFDIR)
            *flags |= 2;
         if ((statbuf.st_mode & S_IFMT) != S_IFREG &&
             (statbuf.st_mode & S_IFMT) != S_IFDIR)
            *flags |= 4;
      }
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTWaitchild()
{
   // Wait till child is finished.

#ifdef R__AIX
   return -1;
#elif !defined(WIN32)
#  ifdef R__HPUX
   int status;
#  else
   union wait status;
#  endif
   return (int) wait3(&status, WNOHANG, 0);
#endif
        return 0;
}

//---- RPC -------------------------------------------------------------------
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-* Error codes set by the Windows Sockets implementation are not made available
//*-* via the errno variable. Additionally, for the getXbyY class of functions,
//*-* error codes are NOT made available via the h_errno variable. Instead, error
//*-* codes are accessed by using the WSAGetLastError . This function is provided
//*-* in Windows Sockets as a precursor (and eventually an alias) for the Win32
//*-* function GetLastError. This is intended to provide a reliable way for a thread
//*-* in a multithreaded process to obtain per-thread error information.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//______________________________________________________________________________
int TWinNTSystem::WinNTTcpConnect(const char *hostname, int port)
{
   // Open a TCP/IP connection to server and connect to a service (i.e. port).
   // Is called via the TSocket constructor.
   short  sport;
   struct servent *sp;

   if ((sp = getservbyport(port, kProtocolName)))
      sport = sp->s_port;
   else
      sport = htons(port);

   TInetAddress addr = gSystem->GetHostByName(hostname);
   if (!addr.IsValid()) return -1;
   UInt_t adr = htonl(addr.GetAddress());

   struct sockaddr_in server;
   memset(&server, 0, sizeof(server));
   memcpy(&server.sin_addr, &adr, sizeof(adr));
   server.sin_family = addr.GetFamily();
   server.sin_port   = sport;

   // Create socket
   SOCKET sock;
   if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
      ::SysError("TWinNTSystem::WinNTConnectTcp", "socket");
      return -1;
   }

   if (connect(sock, (struct sockaddr*) &server, sizeof(server)) == INVALID_SOCKET) {
      //::SysError("TWinNTSystem::UnixConnectTcp", "connect");
      closesocket(sock);
      return -1;
   }
   return (int) sock;
}

#if 0
//______________________________________________________________________________
int TWinNTSystem::WinNTWinNTConnect(int port)
{
   // Connect to a Unix domain socket.

   int sock;
   char buf[100];
   struct sockaddr_un unserver;

   sprintf(buf, "%s/%d", kServerPath, port);

   unserver.sun_family = AF_UNIX;
   strcpy(unserver.sun_path, buf);

   // Open socket
   if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      ::SysError("UnixUnixConnect", "socket");
      return -1;
   }

   if (connect(sock, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2) < 0) {
      // ::SysError("TWinNTSystem::UnixUnixConnect", "connect");
      close(sock);
      return -1;
   }
   return sock;
}
#endif

//______________________________________________________________________________
int TWinNTSystem::WinNTTcpService(int port, Bool_t reuse, int backlog)
{
   // Open a socket, bind to it and start listening for TCP/IP connections
   // on the port. Returns socket fd or -1 if socket() failed, -2 if bind() failed
   // or -3 if listen() failed.

   short  sport;
   struct servent *sp;

   if ((sp = getservbyport(port, kProtocolName)))
      sport = sp->s_port;
   else
      sport = htons(port);

   // Create tcp socket
   SOCKET sock;
   if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      ::SysError("TWinNTSystem::WinNTTcpService", "socket");
      return -1;
   }

   if (reuse)
      gSystem->SetSockOpt((int)sock, kReuseAddr, 1);

   struct sockaddr_in inserver;
   memset(&inserver, 0, sizeof(inserver));
   inserver.sin_family = AF_INET;
   inserver.sin_addr.s_addr = htonl(INADDR_ANY);
   inserver.sin_port = sport;

   // Bind socket
   if (bind(sock, (struct sockaddr*) &inserver, sizeof(inserver)) == SOCKET_ERROR) {
      ::SysError("TWinNTSystem::WinNTTcpService", "bind");
      return -2;
   }

   // Start accepting connections
   if (listen(sock, backlog)==SOCKET_ERROR) {
      ::SysError("TWinNTSystem::WinNTTcpService", "listen");
      return -3;
   }

   return (int)sock;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTWinNTService(int port, int backlog)
{
   // Open a socket, bind to it and start listening for Unix domain connections
   // to it. Returns socket fd or -1.

   SOCKET sock;
#if 0
   struct sockaddr_un unserver;
   int oldumask;

   memset(&unserver, 0, sizeof(unserver));
   unserver.sun_family = AF_UNIX;

   // Assure that socket directory exists
   oldumask = umask(0);
   ::mkdir(kServerPath, 0777);
   umask(oldumask);
   sprintf(unserver.sun_path, "%s/%d", kServerPath, port);

   // Remove old socket
   unlink(unserver.sun_path);
#endif
   // Create socket
   if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) == INVALID_SOCKET) {
      ::SysError("TWinNTSystem::WinNTWinNTService", "socket");
      return -1;
   }
#if 0
   //*-* Winsocket defines the sockaddr_in type sockaddr

   if (bind(sock, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2)) {
      ::SysError("TWinNTSystem::WinNTWinNTService", "bind");
      return -1;
   }
#endif

   // Start accepting connections
   if (listen(sock, backlog)) {
      ::SysError("TWinNTSystem::WinNTWInNTService", "listen");
      return -1;
   }

   return (int) sock;
}
//______________________________________________________________________________
int TWinNTSystem::WinNTRecv(int socket, void *buffer, int length, int flag)
{
   // Receive exactly length bytes into buffer. Returns number of bytes
   // received. Returns -1 in case of error, -2 in case of MSG_OOB
   // and errno == EWOULDBLOCK, -3 in case of MSG_OOB and errno == EINVAL
   // and -4 in case of kNonBlock and errno == EWOULDBLOCK.


   if (socket == -1) return -1;
   SOCKET sock = socket;

   int nrecv, n;
   char *buf = (char *)buffer;

   for (n = 0; n < length; n += nrecv) {
      if ((nrecv = recv(sock, buf+n, length-n, flag)) <= 0) {
         if (nrecv == 0)
            break;        // EOF
         if (flag == MSG_OOB) {
            if (WSAGetLastError() == WSAEWOULDBLOCK)
               return -2;
            else if (WSAGetLastError() == WSAEINVAL)
               return -3;
         }
         if (WSAGetLastError() == WSAEWOULDBLOCK)
            return -4;
         else {
            ::SysError("TWinNTSystem::WinNTRecv", "recv");
            return -1;
         }
      }
   }
   return n;
}

//______________________________________________________________________________
int TWinNTSystem::WinNTSend(int socket, const void *buffer, int length, int flag)
{
   // Send exactly length bytes from buffer.

   if (socket < 0) return -1;
   SOCKET sock = socket;

   int nsent, n;
   const char *buf = (const char *)buffer;

   for (n = 0; n < length; n += nsent) {
      if ((nsent = send(sock, buf+n, length-n, flag)) < 0) {
         ::SysError("TWinNTSystem::WinNTSend", "send");
         return -1;
      }
   }
   return n;
}


//---- Dynamic Loading ---------------------------------------------------------

#ifdef R__HPUX
//______________________________________________________________________________
static const char *DynamicPathName(const char *lib)
{
   // Returns the path of a shared library (searches for library in the
   // shared library search path). If no file name extension is provided
   // it first tries .so, sl and then .dl. Static utility function.

   const char *name = lib;

   if (!strchr(name, '.')) {
      name = Form("%s.dll", lib);
      name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
   } else {

      int len = strlen(name);
      if (len > 4 && (!strcmp(name+len-4, ".dll"))
         name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
      else {
         ::Error("TUnixSystem::DynamicPathName",
                 "shared library must have extension: .dll", lib);
         name = 0;
      }
   }
   return name;
}

//______________________________________________________________________________
static shl_t FindDynLib(const char *lib)
{
   // Returns the handle to a loaded shared library. Returns 0 when library
   // not loaded.

   const char *path;

   if (path = DynamicPathName(lib)) {
      // find handle of shared library using its name
      struct shl_descriptor *desc;
      int index = 0;
      while (shl_get(index++, &desc) == 0)
         if (!strcmp(path, desc->filename))
            return desc->handle;
   }
   return 0;
}
#endif

//______________________________________________________________________________
int TWinNTSystem::WinNTDynLoad(const char *lib)
{
   // Load a shared library. Returns 0 on successful loading.

#ifdef R__HPUX
   const char *path;

   if (path = DynamicPathName(lib)) {
//    shl_t handle = cxxshl_load(path, BIND_DEFERRED, 0L);
      shl_t handle = cxxshl_load(path, BIND_IMMEDIATE | BIND_NONFATAL, 0L);
      if (handle) return 0;
   }
   return 1;
#endif
        return 0;
}

//______________________________________________________________________________
Func_t TWinNTSystem::WinNTDynFindSymbol(const char *lib, const char *entry)
{
   // Finds and returns a function pointer to a symbol in the shared library.
   // Returns 0 when symbol not found.

#ifdef R__HPUX
   shl_t handle;

   if (handle = FindDynLib(lib)) {
      Func_t addr;
      if (shl_findsym(&handle, entry, TYPE_PROCEDURE, addr) == -1)
         ::SysError("TWinNTSystem::WinNTDynFindSymbol", "shl_findsym");
      return addr;
   }
   return 0;
#endif

   // Assume always found
   return (Func_t)1;
}

//______________________________________________________________________________
void TWinNTSystem::WinNTDynListSymbols(const char *lib, const char *regexp)
{
   // List symbols in a shared library. One can use wildcards to list only
   // the intresting symbols.

#ifdef R__HPUX
   shl_t handle;

   if (handle = FindDynLib(lib)) {
      struct shl_symbol *symbols;
      int nsym = shl_getsymbols(handle, TYPE_PROCEDURE,
                                EXPORT_SYMBOLS|NO_VALUES, (void *(*)())malloc,
                                &symbols);
      if (nsym != -1) {
         if (nsym > 0) {
            int cnt = 0;
            TRegexp *re = 0;
            if (regexp && strlen(regexp)) re = new TRegexp(regexp, kTRUE);
            Printf("");
            Printf("Functions exported by library %s", DynamicPathName(lib));
            Printf("=========================================================");
            for (int i = 0; i < nsym; i++)
               if (symbols[i].type == TYPE_PROCEDURE) {
                  cnt++;
                  char *dsym = cplus_demangle(symbols[i].name,
                                              DMGL_PARAMS|DMGL_ANSI|DMGL_ARM);
                  if (re) {
                     TString s = dsym;
                     if (s.Index(*re) != kNPOS) Printf("%s", dsym);
                  } else
                     Printf("%s", dsym);
                  free(dsym);
               }
            Printf("---------------------------------------------------------");
            Printf("%d exported functions", cnt);
            Printf("=========================================================");
            delete re;
         }
         free(symbols);
      }
   }
#endif
}

//______________________________________________________________________________
void TWinNTSystem::WinNTDynListLibs(const char *lib)
{
   // List all loaded shared libraries.

#ifdef R__HPUX
   TRegexp *re = 0;
   if (lib && strlen(lib)) re = new TRegexp(lib, kTRUE);
   struct shl_descriptor *desc;
   int index = 0;

   Printf("");
   Printf("Loaded shared libraries");
   Printf("=======================");

   while (shl_get(index++, &desc) == 0)
      if (re) {
         TString s = desc->filename;
         if (s.Index(*re) != kNPOS) Printf("%s", desc->filename);
      } else
         Printf("%s", desc->filename);
   Printf("-----------------------");
   Printf("%d libraries loaded", index-1);
   Printf("=======================");
   delete re;
#endif
}

//______________________________________________________________________________
void TWinNTSystem::WinNTDynUnload(const char *lib)
{
   // Unload a shared library.

#ifdef R__HPUX
   shl_t handle;

   if (handle = FindDynLib(lib))
      if (cxxshl_unload(handle) == -1)
         ::SysError("TWinNTSystem::WinNTDynUnLoad", "could not unload library %s", lib);
#endif
}
