// @(#)root/vms:$Name:  $:$Id: TVmsSystem.cxx,v 1.4 2001/01/23 19:01:55 rdm Exp $
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
// TVmsSystem                                                          //
//                                                                      //
// Class providing an interface to the UNIX Operating System.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVmsSystem.h"
#include "TROOT.h"
#include "TError.h"
#include "TMath.h"
#include "TOrdCollection.h"
#include "TRegexp.h"
#include "TException.h"
#include "Demangle.h"
#include "TEnv.h"
#include "TSocket.h"
#include "Getline.h"
#include "TTime.h"

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <direntdef.h>
#include <sys/stat.h>
#include <setjmp.h>
#include <signal.h>
#include <pwd.h>
#include <errno.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#   include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <net/soioctl.h>
#include <ioctl.h>
#include <dirent.h>
#include <cxxl.hxx> //#      include <cxxdl.h>
#include <wait.h>
#include <iodef.h>
#include <descrip.h>
#include <stdio.h>
#include <starlet.h>
#include <libdef.h>
#include <lib$routines.h>
    // print stack trace (HP undocumented internal call)
extern "C" void U_STACK_TRACE();

//#if defined(R__ALPHACXX)
extern "C" int inet_aton(const char *, struct in_addr *) { return 0; }


#include "G__ci.h"

const char *kServerPath     = "/tmp";
const char *kProtocolName   = "tcp";

// X11 input event handler which is set in TMotifApplication.
TSysEvtHandler *gXDisplay = 0;


//______________________________________________________________________________
static void SigHandler(ESignals sig)
{
   // Unix signal handler.

   if (gSystem)
      ((TVmsSystem*)gSystem)->DispatchSignals(sig);
}


ClassImp(TVmsSystem)

//______________________________________________________________________________
TVmsSystem::TVmsSystem() : TSystem("Vms", "Vms System")
{ }

//______________________________________________________________________________
TVmsSystem::~TVmsSystem()
{ }

//______________________________________________________________________________
Bool_t TVmsSystem::Init()
{
   // Initialize vms system interface.

   if (TSystem::Init())
      return kTRUE;

   //--- install default handlers
   VmsSignal(kSigChild,                 SigHandler);
   VmsSignal(kSigBus,                   SigHandler);
   VmsSignal(kSigSegmentationViolation, SigHandler);
   VmsSignal(kSigIllegalInstruction,    SigHandler);
   VmsSignal(kSigSystem,                SigHandler);
   VmsSignal(kSigPipe,                  SigHandler);
   VmsSignal(kSigAlarm,                 SigHandler);
   VmsSignal(kSigUrgent,                SigHandler);
   VmsSignal(kSigFloatingException,     SigHandler);
   VmsSignal(kSigWindowChanged,         SigHandler);

   gRootDir = Getenv("ROOTSYS");
   if (gRootDir == 0)
      gRootDir= "/usr/local/root";

   return kFALSE;
}

//---- Misc --------------------------------------------------------------------

//______________________________________________________________________________
void TVmsSystem::SetProgname(const char *name)
{
   // Set the application name (from command line, argv[0]) and copy it in
   // gProgName. Copy the application pathname in gProgPath.

   if (name && strlen(name) > 0) {
      gProgName = StrDup(BaseName(name));
  //  char *w   = Which(Getenv("PATH"), gProgName);
  //  gProgPath = StrDup(DirName(w));
      gProgPath = StrDup(DirName(gProgName));
//    if(w) delete [] w;

   }
}


//______________________________________________________________________________
const char *TVmsSystem::GetError()
{
   // Return system error string.
   return strerror(GetErrno());

}

//______________________________________________________________________________
const char *TVmsSystem::HostName()
{
   // Return the system's host name.

   if (fHostname == "") {
      char hn[64];
#if defined(R__SOLARIS)
      sysinfo(SI_HOSTNAME, hn, sizeof(hn));
#else
      gethostname(hn, sizeof(hn));
#endif
      fHostname = hn;
   }
   return (const char *)fHostname;
}

//---- EventLoop ---------------------------------------------------------------

//______________________________________________________________________________
void TVmsSystem::AddFileHandler(TFileHandler *h)
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
TFileHandler *TVmsSystem::RemoveFileHandler(TFileHandler *h)
{
   // Remove a file handler from the list of file handlers.

   TFileHandler *oh = TSystem::RemoveFileHandler(h);
   if (oh) {       // found
      TFileHandler *th;
      TIter next(fFileHandler);
      fMaxrfd = 0;
      fMaxwfd = 0;
      fReadmask.Zero();
      fWritemask.Zero();
      while ((th = (TFileHandler *) next())) {
         int fd = th->GetFd();
         if (th->HasReadInterest()) {
            fReadmask.Set(fd);
            fMaxrfd = TMath::Max(fMaxrfd, fd);
         }
         if (th->HasWriteInterest()) {
            fWritemask.Set(fd);
            fMaxwfd = TMath::Max(fMaxwfd, fd);
         }
      }
   }
   return oh;
}

//______________________________________________________________________________
void TVmsSystem::AddSignalHandler(TSignalHandler *h)
{
   // Add a signal handler to list of system signal handlers.

   TSystem::AddSignalHandler(h);
   VmsSignal(h->GetSignal(), SigHandler);
}

//______________________________________________________________________________
TSignalHandler *TVmsSystem::RemoveSignalHandler(TSignalHandler *h)
{
   // Remove a signal handler from list of signal handlers.

   // if last handler of specific signal need to reset sighandler to default

   return TSystem::RemoveSignalHandler(h);
}

//______________________________________________________________________________
void TVmsSystem::IgnoreInterrupt(Bool_t ignore)
{
   //Ignore the interrupt signal if ignore == kTRUE else restore previous
   // behaviour. Typically call ignore interrupt before writing to disk.
   //*** Since sigaction is not defined in VMS6.2 and in WINNT this function
   //*** is empty, I also left it empty.

}

//______________________________________________________________________________
void TVmsSystem::DispatchOneEvent()
{
   // Dispatch a single event.
//gROOT->GetApplication()->HandleTermInput();

   while (1) {
      // first handle any X11 events
      if (gXDisplay && gXDisplay->Notify())
         return;

      // check for file descriptors ready for reading/writing
      if (fNfd > 0 && fFileHandler->GetSize() > 0) {
         TFileHandler *fh;
         TOrdCollectionIter it((TOrdCollection*)fFileHandler);

         while ((fh = (TFileHandler*) it.Next())) {
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
      fNfd = VmsSelect(mxfd, &fReadready, &fWriteready, NextTimeOut(kTRUE));
      if (fNfd < 0 && fNfd != -2) {
         int fd, rc;
         TFdSet t;
         for (fd = 0; fd < mxfd; fd++) {
            t.Set(fd);
            if (fReadmask.IsSet(fd)) {
               rc = VmsSelect(fd+1, &t, 0, 0);
               if (rc < 0 && rc != -2) {
                  fprintf(stderr, "select: read error on %d\n", fd);
                  fReadmask.Clr(fd);
               }
            }
            if (fWritemask.IsSet(fd)) {
               rc = VmsSelect(fd+1, 0, &t, 0);
               if (rc < 0 && rc != -2) {
                  fprintf(stderr, "select: write error on %d\n", fd);
                  fWritemask.Clr(fd);
               }
            }
            t.Clr(fd);
         }
      }
  }
}

//______________________________________________________________________________
void TVmsSystem::Sleep(UInt_t milliSec)
{
   // Sleep milliSec milliseconds.

   struct timeval tv;

   tv.tv_sec  = milliSec / 1000;
   tv.tv_usec = (milliSec % 1000) * 1000;

   select(0, 0, 0, 0, &tv);
}

//---- handling of system events -----------------------------------------------

//______________________________________________________________________________
void TVmsSystem::DispatchSignals(ESignals sig)
{
   // Handle and dispatch signals.

   switch (sig) {
   case kSigAlarm:
      DispatchTimers(kFALSE);
      break;
   case kSigChild:
      CheckChilds();
      return;
   case kSigBus:
   case kSigSegmentationViolation:
   case kSigIllegalInstruction:
   case kSigFloatingException:
      Printf("\n *** Break *** %s", VmsSigname(sig));
      StackTrace();
      if (TROOT::Initialized())
         Throw(sig);
      Abort(-1);
      break;
   case kSigSystem:
   case kSigPipe:
      Printf("\n *** Break *** %s", VmsSigname(sig));
      break;
   case kSigWindowChanged:
      Gl_windowchanged();
      break;
   default:
      fSignals.Set(sig);
      fSigcnt++;
      break;
   }

   // check a-synchronous signals
   if (fSigcnt && fSignalHandler->GetSize() > 0)
      CheckSignals(kFALSE);
}

//______________________________________________________________________________
Bool_t TVmsSystem::CheckSignals(Bool_t sync)
{
   // Check if some signals were raised and call their Notify() member.

   TSignalHandler *sh;
   {
      TOrdCollectionIter it((TOrdCollection*)fSignalHandler);

      while ((sh = (TSignalHandler*)it.Next())) {
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
void TVmsSystem::CheckChilds()
{
   // Check if childs have finished.

#if 0  //rdm
   int pid;
   while ((pid = VmsWaitchild()) > 0) {
      TIter next(zombieHandler);
      register VmsPtty *pty;
      while (pty = (VmsPtty*) next())
         if (pty->GetPid() == pid) {
            zombieHandler->RemovePtr(pty);
            pty->DiedNotify();
         }
   }
#endif
}

//---- Directories -------------------------------------------------------------

//______________________________________________________________________________
int TVmsSystem::MakeDirectory(const char *name)
{
   // Make a Unix file system directory. Returns 0 in case of success and
   // -1 if the directory could not be created.

   return VmsMakedir(name);
}

//______________________________________________________________________________
void *TVmsSystem::OpenDirectory(const char *name)
{
   // Open a Unix file system directory. Returns 0 if directory does not exist.

   return VmsOpendir(name);
}

//______________________________________________________________________________
void TVmsSystem::FreeDirectory(void *dirp)
{
   // Close a Unix file system directory.

   //if (dirp)
     // ::closedir((DIR*)dirp); Closedir is not defined in VMS6.2
}

//______________________________________________________________________________
const char *TVmsSystem::GetDirEntry(void *dirp)
{
   // Get next Unix file system directory entry. Returns 0 if no more entries.

   if (dirp)
      return VmsGetdirentry(dirp);
   return 0;
}

//______________________________________________________________________________
Bool_t TVmsSystem::ChangeDirectory(const char *path)
{
   // Change directory. Returns kTRUE in case of success, kFALSE otherwise.

   Bool_t ret = (Bool_t) (::chdir(path) == 0);
   if (fWdpath != "")
      fWdpath = "";   // invalidate path cache
   return ret;
}

//______________________________________________________________________________
const char *TVmsSystem::WorkingDirectory()
{
   // Return working directory.

   if (fWdpath != "")
      return fWdpath.Data();

   static char cwd[kMAXPATHLEN];
   if (getcwd(cwd, kMAXPATHLEN) == 0) {
      fWdpath = "/";
      Error("WorkingDirectory", "getcwd() failed");
   }
   fWdpath = cwd;
   return fWdpath.Data();
}

//______________________________________________________________________________
const char *TVmsSystem::HomeDirectory()
{
   // Return the user's home directory.

   return VmsHomedirectory(0);
}

//______________________________________________________________________________
char *TVmsSystem::ConcatFileName(const char *dir, const char *name)
{
   // Concatenate a directory and a file name. Returned string must be
   // deleted by user.

   if (name == 0 || strlen(name) <= 0 || strcmp(name, ".") == 0)
      return StrDup(dir);

   char buf[kMAXPATHLEN];
   if (dir && (strcmp(dir, "/") != 0)) {
      if (dir[strlen(dir)-1] == '/')
         sprintf(buf, "%s%s", dir, name);
      else
         sprintf(buf, "%s/%s", dir, name);
   } else
      sprintf(buf, "/%s", name);

   return StrDup(buf);
}

//---- Paths & Files -----------------------------------------------------------

//______________________________________________________________________________
Bool_t TVmsSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the Unix access(2) function.

   if (::access(path, mode) == 0)
      return kFALSE;
   fLastErrorString = GetError();
   return kTRUE;
}

//______________________________________________________________________________
void TVmsSystem::Rename(const char *f, const char *t)
{
   // Rename a file.

   ::rename(f, t);
   fLastErrorString = GetError();
}

//______________________________________________________________________________
int TVmsSystem::GetPathInfo(const char *path, unsigned short *id, Long_t *size,
                             Long_t *flags, Long_t *modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is (statbuf.st_dev << 24) + statbuf.st_ino
   // Size    is the file size
   // Flags   is file type: bit 1 set executable, bit 2 set directory,
   //                       bit 3 set regular file
   // Modtime is modification time

   return VmsFilestat(path, id,  size, flags, modtime);
}

//______________________________________________________________________________
int TVmsSystem::Link(const char *from, const char *to)
{
   // Create a link from file1 to file2. Returns 0 when successful,
   // -1 in case of failure.
   //***IN WINNT it returns a 0 without doing anything.

   //return ::link(from, to);
   return 0;
}

//______________________________________________________________________________
//***Symlink does not exist in VMS.  This functions doesn't exist in WINNT.
//int TVmsSystem::Symlink(const char *from, const char *to)

//______________________________________________________________________________
int TVmsSystem::Unlink(const char *name)
{
   // Unlink, i.e. remove, a file or directory.

   struct stat finfo;

   if (stat(name, &finfo) < 0)
      return -1;

   if (S_ISDIR(finfo.st_mode))
      return ::rmdir(name);
   else
      return ::remove(name);  //unlink(name); Not in VMS6.2
}

//---- expand the metacharacters as in the shell -------------------------------

// expand the metacharacters as in the shell

static const char
   *shellMeta      = "~*[]{}?$",
   *shellStuff     = "(){}<>\"'",
   shellEscape     = '\\';

//______________________________________________________________________________
Bool_t TVmsSystem::ExpandPathName(TString &patbuf0)
{
   // Expand a pathname getting rid of special shell characters like ~.$, etc.

   const char *patbuf = (const char *)patbuf0;
   const char *hd, *p;
   char   cmd[kMAXPATHLEN], stuffedPat[kMAXPATHLEN], name[70];
   char  *q;
   FILE  *pf;
   int    ch;

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
   EscChar(patbuf, stuffedPat, sizeof(stuffedPat), (char*)shellStuff, shellEscape);

#ifdef R__HPUX
   strcpy(cmd, "/bin/echo ");
#else
   strcpy(cmd, "echo ");
#endif

   // emulate csh -> popen executes sh
   if (stuffedPat[0] == '~') {
      if (stuffedPat[1] != '\0' && stuffedPat[1] != '/') {
         // extract user name
         for (p = &stuffedPat[1], q = name; *p && *p !='/';)
            *q++ = *p++;
         *q = '\0';
         hd = VmsHomedirectory(name);
         if (hd == 0)
            strcat(cmd, stuffedPat);
         else {
            strcat(cmd, hd);
            strcat(cmd, p);
         }
      } else {
         hd = VmsHomedirectory(0);
         if (hd == 0) {
            fLastErrorString = GetError();
            return kTRUE;
         }
         strcat(cmd, hd);
         strcat(cmd, &stuffedPat[1]);
      }
   } else
      strcat(cmd, stuffedPat);

//   if ((pf = ::popen(&cmd[0], "r")) == NULL) {
//      fLastErrorString = GetError();
//      return kTRUE;
//   }

   // read first argument
   patbuf0 = "";
   int cnt = 0;
#if defined(R__ALPHACXX)
again:
#endif
   for (ch = fgetc(pf); ch != EOF && ch != ' ' && ch != '\n'; ch = fgetc(pf)) {
      patbuf0.Append(ch);
      cnt++;
   }
#if defined(R__ALPHACXX)
   // Work around bug in Alpha/OSF popen
   if (cnt == 0 && ch == EOF) goto again;
#endif

   // skip rest of pipe
   while (ch != EOF) {

      ch = fgetc(pf);
      if (ch == ' ' || ch == '\t') {
         fLastErrorString = "expression ambigous";
//         ::pclose(pf);
         return kTRUE;
      }
   }

//   ::pclose(pf);

   return kFALSE;
}

//______________________________________________________________________________
char *TVmsSystem::ExpandPathName(const char *path)
{
   // Expand a pathname getting rid of special shell characaters like ~.$, etc.
   // User must delete returned string.

   TString patbuf = path;
   if (ExpandPathName(patbuf))
      return 0;
   return StrDup(patbuf.Data());
}

//______________________________________________________________________________
int TVmsSystem::Umask(Int_t mask)
{
   // Set the process file creation mode mask.

   return ::umask(mask);
}

//______________________________________________________________________________
char *TVmsSystem::Which(const char *search, const char *file, EAccessMode mode)
{
   // Find location of file in a search path.
   // User must delete returned string.

   char name[kMAXPATHLEN], temp[kMAXPATHLEN];
   const char *ptr;
   char *next, *exname, *ffile;
   char s2[]=":";

ffile = StrDup(file);

if (search != 0){
  if (strchr(file,']')){
    exname = StrDup(file);
    return exname;
    }
  else if(strrchr(file,':')){
    exname = StrDup(strtok(ffile,s2));
    return exname;
    }
  else {
   if (file == 0){
     exname = StrDup(search);
     return exname;
     }
     strcpy(temp,search);
     strcat(temp,file);
     exname = StrDup(temp);
     if (exname && access(exname,mode) == 0) {
       if (gEnv->GetValue("Root.ShowPath",0))
          Printf("Which: %s = %s", file,exname);
       return exname;
       }
   delete [] exname;
   return 0;
   }
}

#if 0
   if (strchr(file, '/')) {
      if (file[0] != '/' && file[0] != '$' && file[0] != '~') {
         char tmp[kMAXPATHLEN];
         strcpy(tmp, file);
         strcpy(name, gSystem->WorkingDirectory());
         strcat(name, "/");
         strcat(name, tmp);
      } else
         strcpy(name, file);

      exname = gSystem->ExpandPathName(name);
      if (exname && access(exname, mode) == 0) {
         if (gEnv->GetValue("Root.ShowPath", 0))
            Printf("Which: %s = %s", file, exname);
         return exname;
      }
      delete [] exname;
      return 0;
   }

   if (search == 0)
      search = ".";

   for (ptr = search; *ptr;) {
      for (next = name; *ptr && *ptr != ':'; )
         *next++ = *ptr++;
      *next = '\0';

      if (name[0] != '/' && name[0] != '$' && name[0] != '~') {
         char tmp[kMAXPATHLEN];
         strcpy(tmp, name);
         strcpy(name, gSystem->WorkingDirectory());
         strcat(name, "/");
         strcat(name, tmp);
      }

      if (*(name + strlen(name) - 1) != '/')
         strcat(name, "/");
      strcat(name, file);

      exname = gSystem->ExpandPathName(name);
      if (exname && access(exname, mode) == 0) {
         if (gEnv->GetValue("Root.ShowPath", 0))
            Printf("Which: %s = %s", file, exname);
         return exname;
      }
      delete [] exname;
      if (*ptr)
         ptr++;
   }
   return 0;
#endif
}

//---- environment manipulation ------------------------------------------------

//______________________________________________________________________________
void TVmsSystem::Setenv(const char *name, const char *value)
{
   // Set environment variable. The string passed will be owned by
   // the environment and can not be reused till a "name" is set
   // again. The solution below will lose the space for the string
   // in that case, but if this functions is not called thousands
   // of times that should not be a problem.

   char *s = new char [strlen(name)+strlen(value) + 2];
   sprintf(s, "%s=%s", name, value);
   printf("Setenv:: You cannot setenv in VMS6.2");
  // ::setenv(s);
  //***setenv not in VMS6.2
}

//______________________________________________________________________________
const char *TVmsSystem::Getenv(const char *name)
{
   // Get environment variable.

   return ::getenv(name);
}

//---- Processes ---------------------------------------------------------------

//______________________________________________________________________________
int TVmsSystem::Exec(const char *shellcmd)
{
   // Execute a command.

   return ::system(shellcmd);
}

//______________________________________________________________________________

//______________________________________________________________________________
int TVmsSystem::GetPid()
{
   // Get process id.

   return ::getpid();
}

//______________________________________________________________________________
void TVmsSystem::Exit(int code, Bool_t mode)
{
   // Exit the application.

   if (mode)
      ::exit(code);
   else
      ::_exit(code);
}

//______________________________________________________________________________
void TVmsSystem::Abort(int)
{
   // Abort the application.

   ::abort();
}

//______________________________________________________________________________
void TVmsSystem::StackTrace()
{
   // Print a stack trace.

#ifdef R__HPUX
      Printf("");
      U_STACK_TRACE();
      Printf("");
#endif
}

//---- System Logging ----------------------------------------------------------

//______________________________________________________________________________
//***Openlog is defined in syslog.h.  It does not exist in Vms.
//***WINNT does not have this function.
//void TVmsSystem::Openlog(const char *name, Int_t options, ELogFacility facility)

//______________________________________________________________________________
//***Syslog is not defined in VMS.  Function not in WINNT.
//void TVmsSystem::Syslog(ELogLevel level, const char *mess)

//______________________________________________________________________________
//***CloseLog not defined in VMS.  Function not in WINNT.
//void TVmsSystem::Closelog()

//---- dynamic loading and linking ---------------------------------------------

//______________________________________________________________________________
Func_t TVmsSystem::DynFindSymbol(const char *module, const char *entry)
{
   return VmsDynFindSymbol(module,entry);
}

//______________________________________________________________________________
int TVmsSystem::Load(const char *module, const char *entry)
{
   // Load a shared library. Returns 0 on successful loading, 1 in
   // case lib was already loaded and -1 in case lib does not exist
   // or in case of error.

   int i = VmsDynLoad(module);
   if (!entry || !strlen(entry)) return i;

   Func_t f = VmsDynFindSymbol(module, entry);
   if (f) return 0;
   return -1;
}

//______________________________________________________________________________
void TVmsSystem::Unload(const char *module)
{
   // Unload a shared library.

   VmsDynUnload(module);
}

//______________________________________________________________________________
void TVmsSystem::ListSymbols(const char *module, const char *regexp)
{
   // List symbols in a shared library.

   VmsDynListSymbols(module, regexp);
}

//______________________________________________________________________________
void TVmsSystem::ListLibraries(const char *regexp)
{
   // List all loaded shared libraries.

   VmsDynListLibs(regexp);
}

//---- Time & Date -------------------------------------------------------------

//______________________________________________________________________________
TTime TVmsSystem::Now()
{
   // Return current time.

   return VmsNow();
}

//______________________________________________________________________________
Bool_t TVmsSystem::DispatchTimers(Bool_t mode)
{
   // Handle and dispatch timers. If mode = kTRUE dispatch synchronous
   // timers else a-synchronous timers.

   if (!fTimers) return kFALSE;

   fInsideNotify = kTRUE;

   TOrdCollectionIter it((TOrdCollection*)fTimers);
   TTimer *t;
   Bool_t  timedout = kFALSE;

   while ((t = (TTimer *) it.Next())) {
      Long_t now = VmsNow()+kItimerResolution;
      if (mode && t->IsSync()) {
         if (t->CheckTimer(now))
            timedout = kTRUE;
      } else if (!mode && t->IsAsync()) {
         if (t->CheckTimer(now)) {
            VmsSetitimer(NextTimeOut(kFALSE));
            timedout = kTRUE;
         }
      }
   }
   fInsideNotify = kFALSE;
   return timedout;
}

//______________________________________________________________________________
void TVmsSystem::AddTimer(TTimer *ti)
{
   // Add timer to list of system timers.

   TSystem::AddTimer(ti);
   if (!fInsideNotify)  // we are not inside notify
      if (ti->IsAsync())
         VmsSetitimer(NextTimeOut(kFALSE));
}

//______________________________________________________________________________
TTimer *TVmsSystem::RemoveTimer(TTimer *ti)
{
   // Remove timer from list of system timers.

   TTimer *t = TSystem::RemoveTimer(ti);
   if (ti->IsAsync())
      VmsSetitimer(NextTimeOut(kFALSE));
   return t;
}

//---- RPC ---------------------------------------------------------------------

//______________________________________________________________________________
TInetAddress TVmsSystem::GetHostByName(const char *hostname)
{
   // Get Internet Protocol (IP) address of host.

   struct hostent *host_ptr;
   struct in_addr  ad;
   const char     *host;
   int             type;
   UInt_t          addr;    // good for 4 byte addresses

   if (inet_aton(hostname, &ad)) {
      memcpy(&addr, &ad.s_addr, sizeof(ad.s_addr));
      if ((host_ptr = gethostbyaddr((const char *)&ad.s_addr,
                                    sizeof(ad.s_addr), AF_INET)))
         host = host_ptr->h_name;
      else
         host = hostname;
      type = AF_INET;
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
      Error("GetHostByName", "unknown host %s", hostname);
      return TInetAddress();
   }

   return TInetAddress(host, ntohl(addr), type);
}

//______________________________________________________________________________
TInetAddress TVmsSystem::GetSockName(int sock)
{
   // Get Internet Protocol (IP) address of host and port #.

   struct sockaddr_in addr;
#if defined(R__AIX) && defined(_AIX41)
   size_t len = sizeof(addr);
#else
   unsigned int len = sizeof(addr);
#endif

   if (getsockname(sock, (struct sockaddr *)&addr, &len) == -1) {
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
TInetAddress TVmsSystem::GetPeerName(int sock)
{
   // Get Internet Protocol (IP) address of remote host and port #.

   struct sockaddr_in addr;
#if defined(R__AIX) && defined(_AIX41)
   size_t len = sizeof(addr);
#else
   unsigned int len = sizeof(addr); //change from int to unsigned int
#endif

   if (getpeername(sock, (struct sockaddr *)&addr, &len) == -1) {
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
int TVmsSystem::GetServiceByName(const char *servicename)
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
char *TVmsSystem::GetServiceByPort(int port)
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
int TVmsSystem::ConnectService(const char *servername, int port,
                               int tcpwindowsize)
{
   // Connect to service servicename on server servername.

   if (!strcmp(servername, "vms"))
      return VmsVmsConnect(port);
   return VmsTcpConnect(servername, port, tcpwindowsize);
}

//______________________________________________________________________________
int TVmsSystem::OpenConnection(const char *server, int port, int tcpwindowsize)
{
   // Open a connection to a service on a server. Try 3 times with an
   // interval of 1 second.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Is called via the TSocket constructor.

   for (int i = 0; i < 3; i++) {
      int fd = ConnectService(server, port, tcpwindowsize);
      if (fd >= 0)
         return fd;
      sleep(1);
   }
   return -1;
}

//______________________________________________________________________________
int TVmsSystem::AnnounceTcpService(int port, Bool_t reuse, int backlog,
                                   int tcpwindowsize)
{
   // Announce TCP/IP service.
   // Open a socket, bind to it and start listening for TCP/IP connections
   // on the port. If reuse is true reuse the address, backlog specifies
   // how many sockets can be waiting to be accepted.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns socket fd or -1 if socket() failed, -2 if bind() failed
   // or -3 if listen() failed.

   return VmsTcpService(port, reuse, backlog, tcpwindowsize);
}

//______________________________________________________________________________
int TVmsSystem::AnnounceVmsService(int port, int backlog)
{
   // Announce unix domain service.

   return VmsVmsService(port, backlog);
}

//______________________________________________________________________________
int TVmsSystem::AcceptConnection(int sock)
{
   // Accept a connection. In case of an error return -1. In case
   // non-blocking I/O is enabled and no connections are available
   // return -2.

   int soc = -1;

   while ((soc = ::accept(sock, 0, 0)) == -1 && GetErrno() == EINTR)
      ResetErrno();

   if (soc == -1) {
      if (GetErrno() == EWOULDBLOCK)
         return -2;
      else {
         SysError("AcceptConnection", "accept");
         return -1;
      }
   }

   return soc;
}

//______________________________________________________________________________
void TVmsSystem::CloseConnection(int sock, Bool_t force)
{
   // Close socket.

   if (sock < 0) return;

   if (force)
      ::shutdown(sock, 2);

   while (::close(sock) == -1 && GetErrno() == EINTR)
      ResetErrno();
}

//______________________________________________________________________________
int TVmsSystem::RecvBuf(int sock, void *buf, int length)
{
   // Receive a buffer headed by a length indicator. Lenght is the size of
   // the buffer. Returns the number of bytes received in buf or -1 in
   // case of error.

   Int_t header;

   if (VmsRecv(sock, &header, sizeof(header), 0) > 0) {
      int count = ntohl(header);

      if (count > length) {
         Error("RecvBuf", "record header exceeds buffer size");
         return -1;
      } else if (count > 0) {
         if (VmsRecv(sock, buf, count, 0) < 0) {
            Error("RecvBuf", "cannot receive buffer");
            return -1;
         }
      }
      return count;
   }
   return -1;
}

//______________________________________________________________________________
int TVmsSystem::SendBuf(int sock, const void *buf, int length)
{
   // Send a buffer headed by a length indicator. Returns length of sent buffer
   // or -1 in case of error.

   Int_t header = htonl(length);

   if (VmsSend(sock, &header, sizeof(header), 0) < 0) {
      Error("SendBuf", "cannot send header");
      return -1;
   }
   if (length > 0) {
      if (VmsSend(sock, buf, length, 0) < 0) {
         Error("SendBuf", "cannot send buffer");
         return -1;
      }
   }
   return length;
}

//______________________________________________________________________________
int TVmsSystem::RecvRaw(int sock, void *buf, int length, int opt)
{
   // Receive exactly length bytes into buffer. Use opt to receive out-of-band
   // data or to have a peek at what is in the buffer (see TSocket). Buffer
   // must be able to store at least lenght bytes. Returns the number of
   // bytes received (can be 0 if other side of connection was closed) or -1
   // in case of error, -2 in case of MSG_OOB and errno == EWOULDBLOCK, -3
   // in case of MSG_OOB and errno == EINVAL and -4 in case of kNonBlock and
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
   case kDontBlock:
      flag = -1;
      break;
   default:
      flag = 0;
      break;
   }

   int n;
   if ((n = VmsRecv(sock, buf, length, flag)) <= 0) {
      if (n == -1 && GetErrno() != EINTR)
         Error("RecvRaw", "cannot receive buffer");
      return n;
   }
   return n;
}

//______________________________________________________________________________
int TVmsSystem::SendRaw(int sock, const void *buf, int length, int opt)
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
   case kDontBlock:
      flag = -1;
      break;
   case kPeek:            // receive only option (see RecvRaw)
   default:
      flag = 0;
      break;
   }

   int n;
   if ((n = VmsSend(sock, buf, length, flag)) < 0) {
      Error("SendRaw", "cannot send buffer");
      return -1;
   }
   return n;
}

//______________________________________________________________________________
int TVmsSystem::SetSockOpt(int sock, int opt, int val)
{
   // Set socket option.

   if (sock < 0) return -1;

   switch (opt) {
   case kSendBuffer:
      if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(SO_SNDBUF)");
         return -1;
      }
      break;
   case kRecvBuffer:
      if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(SO_RCVBUF)");
         return -1;
      }
      break;
   case kOobInline:
      if (setsockopt(sock, SOL_SOCKET, SO_OOBINLINE, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(SO_OOBINLINE)");
         return -1;
      }
      break;
   case kKeepAlive:
      if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(SO_KEEPALIVE)");
         return -1;
      }
      break;
   case kReuseAddr:
      if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(SO_REUSEADDR)");
         return -1;
      }
      break;
   case kNoDelay:
      if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&val, sizeof(val)) == -1) {
         SysError("SetSockOpt", "setsockopt(TCP_NODELAY)");
         return -1;
      }
      break;
   case kNoBlock:
     //***ioctl does not exist in VMS6.2
     // if (ioctl(sock, FIONBIO, (char*)&val) == -1) {
     //    SysError("SetSockOpt", "ioctl(FIONBIO)");
     //    return -1;
     // }
      printf("Caution::Commented Out ioctl().");
      break;
   case kProcessGroup:
     //***ioctl does not exist in VMS6.2
     // if (ioctl(sock, SIOCSPGRP, (char*)&val) == -1) {
     //    SysError("SetSockOpt", "ioctl(SIOCSPGRP)");
     //    return -1;
     // }
     printf("Caution::Commented Out ioctl().");
      break;
   case kAtMark:       // read-only option (see GetSockOpt)
   case kBytesToRead:  // read-only option
   default:
      Error("SetSockOpt", "illegal option (%d)", opt);
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
int TVmsSystem::GetSockOpt(int sock, int opt, int *val)
{
   // Get socket option.

   if (sock < 0) return -1;

   unsigned int optlen = sizeof(*val);

   switch (opt) {
   case kSendBuffer:
      if (getsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(SO_SNDBUF)");
         return -1;
      }
      break;
   case kRecvBuffer:
      if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(SO_RCVBUF)");
         return -1;
      }
      break;
   case kOobInline:
      if (getsockopt(sock, SOL_SOCKET, SO_OOBINLINE, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(SO_OOBINLINE)");
         return -1;
      }
      break;
   case kKeepAlive:
      if (getsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(SO_KEEPALIVE)");
         return -1;
      }
      break;
   case kReuseAddr:
      if (getsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(SO_REUSEADDR)");
         return -1;
      }
      break;
   case kNoDelay:
      if (getsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)val, &optlen) == -1) {
         SysError("GetSockOpt", "getsockopt(TCP_NODELAY)");
         return -1;
      }
      break;
   case kNoBlock:
      //Must be checked somehow!!
      int flg;
       printf("GetSockOpt::This is not being check right now!!");
     // if ((flg = fcntl(sock, F_GETFL, 0)) == -1) {
     //    SysError("GetSockOpt", "fcntl(F_GETFL)");
     //    return -1;
     // }
      *val = flg & O_NDELAY;
      break;
   case kProcessGroup:
     // if (ioctl(sock, SIOCGPGRP, (char*)val) == -1) {
     //    SysError("GetSockOpt", "ioctl(SIOCGPGRP)");
     //    return -1;
     // }
     printf("Caution::Commented Out ioctl().");
      break;
   case kAtMark:
     // if (ioctl(sock, SIOCATMARK, (char*)val) == -1) {
     //    SysError("GetSockOpt", "ioctl(SIOCATMARK)");
     //    return -1;
     // }
     printf("Caution::Commented Out ioctl().");
      break;
   case kBytesToRead:
    //  if (ioctl(sock, FIONREAD, (char*)val) == -1) {
     //    SysError("GetSockOpt", "ioctl(FIONREAD)");
     //    return -1;
     // }
     printf("Caution::Commented Out ioctl().");
      break;
   default:
      Error("GetSockOpt", "illegal option (%d)", opt);
      *val = 0;
      return -1;
   }
   return 0;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Static Protected Unix Interface functions.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//---- signals -----------------------------------------------------------------
//In VMS SIGWINCH, SIGCHLD and SIGURG is not defined.  These values are from
//Unix.

#define SIGWINCH 20
#define SIGCHLD 18
#define SIGURG 0x2

static struct signal_map {
   int code;
   SigHandler_t handler;
   const char *signame;
} gSignalMap[kMAXSIGNALS] = {    // the order of the signals should be identical
   { SIGBUS,   0, "bus error" }, // to the one in SysEvtHandler.h
   { SIGSEGV,  0, "segmentation violation" },
   { SIGSYS,   0, "bad argument to system call" },
   { SIGPIPE,  0, "write on a pipe with no one to read it" },
   { SIGILL,   0, "illegal instruction" },
   { SIGQUIT,  0, "quit" },
   { SIGINT,   0, "interrupt" },
   { SIGWINCH, 0, "window size change" },
   { SIGALRM,  0, "alarm clock" },
   { SIGCHLD,  0, "death of a child" },
   { SIGURG,   0, "urgent data arrived on an I/O channel" },
   { SIGFPE,   0, "floating point exception" },
   { SIGTERM,  0, "termination signal" },
   { SIGUSR1,  0, "user-defined signal 1" },
   { SIGUSR2,  0, "user-defined signal 2" }
};

//______________________________________________________________________________
static void sighandler(int sig)
{
   // Call the signal handler associated with the signal.

   for (int i= 0; i < kMAXSIGNALS; i++) {
      if (gSignalMap[i].code == sig) {
         (*gSignalMap[i].handler)((ESignals)i);
         return;
      }
   }
}

//______________________________________________________________________________
void TVmsSystem::VmsSignal(ESignals sig, SigHandler_t handler)
{
//***NEED to FIX LATER:: Define Sigaction
printf("VmsSignal::This will not execute anything.  Sigaction not defined.\n");
 // Set a signal handler for a signal.

//   if (gSignalMap[sig].handler != handler) {
//      struct sigaction sigact;

//      gSignalMap[sig].handler = handler;

//#if defined(R__SOLARIS) && !defined(R__I386)
//      sigact.sa_handler = (void (*)())sighandler;
//#elif defined(R__SGI)
//      sigact.sa_handler = (void (*)(...))sighandler;
//#else
//      sigact.sa_handler = sighandler;
//#endif
//      sigemptyset(&sigact.sa_mask);
//#if defined(SA_INTERRUPT)       // SunOS
//      sigact.sa_flags = SA_INTERRUPT;
//#else
//     sigact.sa_flags = 0;
//#endif
//      if (sigaction(gSignalMap[sig].code, &sigact, 0) < 0)
//         ::SysError("TVmsSystem::VmsSignal", "sigaction");
//   }
}

//______________________________________________________________________________
const char *TVmsSystem::VmsSigname(ESignals sig)
{
   // Return the signal name associated with a signal.

   return gSignalMap[sig].signame;
}

//---- time --------------------------------------------------------------------

//______________________________________________________________________________
Long_t TVmsSystem::VmsNow()
{
   // Get current time in milliseconds since 0:00 Jan 1 1995.

   static time_t jan95 = 0;
   if (!jan95) {
      struct tm tp;
      tp.tm_year  = 95;
      tp.tm_mon   = 0;
      tp.tm_mday  = 1;
      tp.tm_hour  = 0;
      tp.tm_min   = 0;
      tp.tm_sec   = 0;
      tp.tm_isdst = -1;

      jan95 = mktime(&tp);
      if ((int)jan95 == -1) {
         ::SysError("TVmsSystem::VmsNow", "error converting 950001 0:00 to time_t");
         return 0;
      }
   }

   struct timeval t;
  //***Find something that can replace gettimeofday.
 // gettimeofday(&t, 0);
   return (t.tv_sec-(Long_t)jan95)*1000 + t.tv_usec/1000;
}

//______________________________________________________________________________
int TVmsSystem::VmsSetitimer(Long_t ms)
{
   // Set interval timer to time-out in ms milliseconds.
   //****NEED to FIX.  Define Setitimer.
   struct itimerval itv;
   itv.it_interval.tv_sec = 0;
   itv.it_value.tv_usec   = 0;
   itv.it_interval.tv_sec = 0;
   itv.it_value.tv_usec   = 0;
   if (ms >= 0) {
      itv.it_value.tv_sec  = Int_t(ms / 1000);
      itv.it_value.tv_usec = (ms % 1000) * 1000;
   }
printf("VmsSetitimer::Will not return setitimer. Not defined in VMS.");
return 0; //dummy return
//   return setitimer(ITIMER_REAL, &itv, 0);
}

//---- file descriptors --------------------------------------------------------

//______________________________________________________________________________
int TVmsSystem::VmsSelect(UInt_t nfds, TFdSet *readready, TFdSet *writeready,
                            Long_t timeout)
{
   // Wait for events on the file descriptors specified in the readready and
   // writeready masks or for timeout (in milliseconds) to occur.

   int retcode;
   int ikb_hitflag = 0;

if (readready->IsSet(0)) {
      ikb_hitflag =1;
      }

   if (timeout >= 0) {
      struct timeval tv;
      tv.tv_sec  = Int_t(timeout / 1000);
      tv.tv_usec = (timeout % 1000) * 1000;
      retcode = select(nfds, (fd_set*)readready->GetBits(),(fd_set*)writeready->GetBits(), 0, &tv);
   } else {
      retcode = select(nfds, (fd_set*)readready->GetBits(), (fd_set*)writeready->GetBits(), 0, 0);
      if (ikb_hitflag == 1) {
        if(Kbhit()) {
          readready->Set(0);
          retcode++;
          }
        else {
          readready->Clr(0);
          }
      }
    }

   if (retcode == -1) {
      if (GetErrno() == EINTR) {
         ResetErrno();  // errno is not self reseting
         return -2;
      }
      if (GetErrno() == EBADF)
         return -3;
      return -1;
   }
   return retcode;
}

int TVmsSystem::Kbhit()
{

register status;
static short channel = 0, iosb[4];
static $DESCRIPTOR (terminal,"TT:");
static struct {
                unsigned short number;
                unsigned char first;
                unsigned char filler[5];
              } typahdbuff;

if (!channel)
  {
    status = sys$assign(&terminal,&channel,0,0);
    if (!(status &1)) lib$stop(status);
  }

status = sys$qiow(0,channel,IO$_SENSEMODE | IO$M_TYPEAHDCNT,iosb, 0,0,
&typahdbuff,sizeof(typahdbuff),0,0,0,0);

if (!(status & 1)) lib$stop(status);
if (!(iosb[0] & 1)) lib$stop(iosb[0]);

if (typahdbuff.number > 0)
  return (1);
else
  return (0);
}

//---- directories -------------------------------------------------------------

//______________________________________________________________________________
const char *TVmsSystem::VmsHomedirectory(const char *name)
{
//***NEED TO FIX LATER!
printf("VmsHomedirectory::This will not return the current directory. ");
 // Returns the user's home directory.

//   static char path[kMAXPATHLEN], mydir[kMAXPATHLEN];
//   struct passwd *pw;

//   if (name) {
//get      pw = getpwnam(name);
//      if (pw) {
//         strncpy(path, pw->pw_dir, kMAXPATHLEN);
//         return path;
//      }
//   } else {
//      if (mydir[0])
//         return mydir;
//      pw = getpwuid(getuid());
//      if (pw) {
//         strncpy(mydir, pw->pw_dir, kMAXPATHLEN);
//         return mydir;
//      }
//  }
   return 0;
}

//______________________________________________________________________________
int TVmsSystem::VmsMakedir(const char *dir)
{
   // Make a Vms file system directory. Returns 0 in case of success and
   // -1 if the directory could not be created.

   return ::mkdir(dir, 0755);
}

//______________________________________________________________________________
void *TVmsSystem::VmsOpendir(const char *dir)
{
   // Open a directory.

   struct stat finfo;

   if (stat(dir, &finfo) < 0)
      return 0;

   if (!S_ISDIR(finfo.st_mode))
      return 0;

   printf("This will not opendir().  Is not defined in 6.2");
   return 0;//(void*) opendir(dir);

}

#   define REAL_DIR_ENTRY(dp) (dp->d_ino != 0)


//______________________________________________________________________________
//Not in VMS but this is how they defined it in Unix.

typedef struct
    {
    int dd_fd;
    int dd_loc;
    int dd_size;
    char *dd_buf;
    } DIR;

const char *TVmsSystem::VmsGetdirentry(void *dirp1)
{
   // Returns the next directory entry.

printf("This will not execute anything.  6.2 does not have readdir.");
//   DIR *dirp = (DIR*)dirp1;
//#if defined(R__SUN) || defined(R__SGI) || defined(R__AIX) || defined(R__HPUX) || \
//    defined(R__LINUX) || defined(R__SOLARIS) || defined(R__ALPHA)
//   struct dirent *dp;
//#else
//   struct direct *dp;
//#endif

//   if (dirp) {
//      for (;;) {
//         dp = readdir(dirp);
//         if (dp == 0)
//            return 0;
//         if (REAL_DIR_ENTRY(dp))
//            return dp->d_name;
//      }
//   }
   return 0;
}

//---- files -------------------------------------------------------------------

//______________________________________________________________________________
//***Changed id from Long_t* to short*.  In Vms statbuf.st_dev is defined to be
//*** a short.
int TVmsSystem::VmsFilestat(const char *path, unsigned short *id, Long_t *size,
                              Long_t *flags, Long_t *modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is (statbuf.st_dev << 24) + statbuf.st_ino
   // Size    is the file size
   // Flags   is file type: bit 0 set executable, bit 1 set directory,
   //                       bit 2 set regular file
   // Modtime is modification time
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   struct stat statbuf;

   if (path != 0 && stat(path, &statbuf) >= 0) {
      if (id)
         *id = (*(statbuf.st_dev) << 24) + *statbuf.st_ino;
      if (size)
         *size = statbuf.st_size;
      if (modtime)
         *modtime = statbuf.st_mtime;
      if (flags) {
         *flags = 0;
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
//***waitpid not defined in VMS6.2
//***One of the return statements in WINNT is zero.
int TVmsSystem::VmsWaitchild()
{
   // Wait till child is finished.

   //int status;
   //return (int) waitpid(0, &status, WNOHANG);
   printf("VmsWaitchild::Always will return 0.");
   return 0;
}

//---- RPC -------------------------------------------------------------------

//______________________________________________________________________________
int TVmsSystem::VmsTcpConnect(const char *hostname, int port, int tcpwindowsize)
{
   // Open a TCP/IP connection to server and connect to a service (i.e. port).
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
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
   int sock;
   if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      ::SysError("TVmsSystem::VmsConnectTcp", "socket");
      return -1;
   }

   if (tcpwindowsize > 0) {
      gSystem->SetSockOpt(sock, kRecvBuffer, tcpwindowsize);
      gSystem->SetSockOpt(sock, kSendBuffer, tcpwindowsize);
   }

   if (connect(sock, (struct sockaddr*) &server, sizeof(server)) < 0) {
      //::SysError("TVmsSystem::VmsConnectTcp", "connect");
      close(sock);
      return -1;
   }
   return sock;
}

//VMS does not define sockaddr_un.  This struct is from unix.

struct sockaddr_un{
        short sun_family; //AF_UNIX
        char sun_path[108] ; //path name (gag)
};

//______________________________________________________________________________
int TVmsSystem::VmsVmsConnect(int port)
{
   // Connect to a Vms domain socket.

   int sock;
   char buf[100];
   struct sockaddr_un unserver;

   sprintf(buf, "%s/%d", kServerPath, port);

   unserver.sun_family = AF_UNIX;
   strcpy(unserver.sun_path, buf);

   // Open socket
   if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      ::SysError("TVmsSystem::VmsVmsConnect", "socket");
      return -1;
   }

   if (connect(sock, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2) < 0) {
      // ::SysError("TVmsSystem::VmsVmsConnect", "connect");
      close(sock);
      return -1;
   }
   return sock;
}

//______________________________________________________________________________
int TVmsSystem::VmsTcpService(int port, Bool_t reuse, int backlog,
                              int tcpwindowsize)
{
   // Open a socket, bind to it and start listening for TCP/IP connections
   // on the port. If reuse is true reuse the address, backlog specifies
   // how many sockets can be waiting to be accepted. If port is 0 a port
   // scan will be done to find a free port. This option is mutual exlusive
   // with the reuse option.
   // Use tcpwindowsize to specify the size of the receive buffer, it has
   // to be specified here to make sure the window scale option is set (for
   // tcpwindowsize > 65KB and for platforms supporting window scaling).
   // Returns socket fd or -1 if socket() failed, -2 if bind() failed
   // or -3 if listen() failed.

   const short kSOCKET_MINPORT = 5000, kSOCKET_MAXPORT = 15000;
   short  sport, tryport = kSOCKET_MINPORT;
   struct servent *sp;

   if (port == 0 && reuse) {
      ::Error("TVmsSystem::VmsTcpService", "cannot do a port scan while reuse is true");
      return -1;
   }

   if ((sp = getservbyport(port, kProtocolName)))
      sport = sp->s_port;
   else
      sport = htons(port);

   // Create tcp socket
   int sock;
   if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      ::SysError("TVmsSystem::VmsTcpService", "socket");
      return -1;
   }

   if (reuse)
      gSystem->SetSockOpt(sock, kReuseAddr, 1);

   if (tcpwindowsize > 0) {
      gSystem->SetSockOpt(sock, kRecvBuffer, tcpwindowsize);
      gSystem->SetSockOpt(sock, kSendBuffer, tcpwindowsize);
   }

   struct sockaddr_in inserver;
   memset(&inserver, 0, sizeof(inserver));
   inserver.sin_family = AF_INET;
   inserver.sin_addr.s_addr = htonl(INADDR_ANY);
   inserver.sin_port = sport;

   // Bind socket
   if (port > 0) {
      for (int retry = 20; bind(sock, (struct sockaddr*) &inserver, sizeof(inserver)); retry--) {
         if (retry <= 0) {
            ::SysError("TVmsSystem::VmsTcpService", "bind");
            return -2;
         }
         sleep(10);
      }
   } else {
      int bret;
      do {
         inserver.sin_port = htons(tryport++);
         bret = bind(sock, (struct sockaddr*) &inserver, sizeof(inserver));
      } while (bret < 0 && GetErrno() == EADDRINUSE && tryport < kSOCKET_MAXPORT);
      if (bret < 0) {
         ::SysError("TVmsSystem::VmsTcpService", "bind (port scan)");
         return -2;
      }
   }

   // Start accepting connections
   if (listen(sock, backlog)) {
      ::SysError("TVmsSystem::VmsTcpService", "listen");
      return -1;
   }

   return sock;
}

//______________________________________________________________________________
int TVmsSystem::VmsVmsService(int port, int backlog)
{
   // Open a socket, bind to it and start listening for Unix domain connections
   // to it. Returns socket fd or -1.

   struct sockaddr_un unserver;
   int sock, oldumask;

   memset(&unserver, 0, sizeof(unserver));
   unserver.sun_family = AF_UNIX;

   // Assure that socket directory exists
   oldumask = umask(0);
   ::mkdir(kServerPath, 0777);
   umask(oldumask);
   sprintf(unserver.sun_path, "%s/%d", kServerPath, port);

   // Remove old socket
   //**Unlink is not defined in VMS6.2.
   //unlink(unserver.sun_path);
     remove(unserver.sun_path);

   // Create socket
   if ((sock = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      ::SysError("TVmsSystem::VmsVmsService", "socket");
      return -1;
   }

   if (bind(sock, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2)) {
      ::SysError("TVmsSystem::VmsVmsService", "bind");
      return -1;
   }

   // Start accepting connections
   if (listen(sock, backlog)) {
      ::SysError("TVmsSystem::VmsVmsService", "listen");
      return -1;
   }

   return sock;
}

//______________________________________________________________________________
int TVmsSystem::VmsRecv(int sock, void *buffer, int length, int flag)
{
   // Receive exactly length bytes into buffer. Returns number of bytes
   // received. Returns -1 in case of error, -2 in case of MSG_OOB
   // and errno == EWOULDBLOCK, -3 in case of MSG_OOB and errno == EINVAL
   // and -4 in case of kNonBlock and errno == EWOULDBLOCK.

   ResetErrno();

   if (sock < 0) return -1;

   int once = 0;
   if (flag == -1) {
      flag = 0;
      once = 1;
   }

   int n, nrecv = 0;
   char *buf = (char *)buffer;

   for (n = 0; n < length; n += nrecv) {
      if ((nrecv = recv(sock, buf+n, length-n, flag)) <= 0) {
         if (nrecv == 0)
            break;        // EOF
         if (flag == MSG_OOB) {
            if (GetErrno() == EWOULDBLOCK)
               return -2;
            else if (GetErrno() == EINVAL)
               return -3;
         }
         if (GetErrno() == EWOULDBLOCK)
            return -4;
         else {
            if (GetErrno() != EINTR)
               ::SysError("TVmsSystem::VmsRecv", "recv");
            return -1;
         }
      }
      if (once)
         return nrecv;
   }
   return n;
}

//______________________________________________________________________________
int TVmsSystem::VmsSend(int sock, const void *buffer, int length, int flag)
{
   // Send exactly length bytes from buffer. Returns -1 in case of error,
   // otherwise number of sent bytes.

   if (sock < 0) return -1;

   int once = 0;
   if (flag == -1) {
      flag = 0;
      once = 1;
   }

   int n, nsent = 0;
   const char *buf = (const char *)buffer;

   for (n = 0; n < length; n += nsent) {
      if ((nsent = send(sock, buf+n, length-n, flag)) <= 0) {
         ::SysError("TVmsSystem::VmsSend", "send");
         return nsent;
      }
      if (once)
         return nsent;
   }
   return n;
}

//---- Dynamic Loading ---------------------------------------------------------

//______________________________________________________________________________
const char *TVmsSystem::GetDynamicPath()
{
   // Get shared library search path. Static utility function.

   static const char *dynpath = 0;

   if (dynpath == 0) {
      dynpath = gEnv->GetValue("Root.DynamicPath", (char*)0);
      if (dynpath == 0)
//      if (strrchr(gRootDir,']'))
//        *strrchr(gRootDir,']') = '.';
        dynpath = StrDup(Form("%s[lib]",gRootDir));
   }
   return dynpath;
}

//______________________________________________________________________________
char *TVmsSystem::DynamicPathName(const char *lib)
{
   // Returns the path of a shared library (searches for library in the
   // shared library search path). If no file name extension is provided
   // it first tries .so, .sl and then .dl. Returned string must be deleted.

   char *name;

   if (!strchr(lib, '.')) {
      name = Form("%s.so", lib);
      name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
      if (!name) {
         name = Form("%s.sl", lib);
         name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
         if (!name) {
             name = Form("%s.dl", lib);
             name = gSystem->Which(GetDynamicPath(), name, kReadPermission);
         }
      }

   } else {

      int len = strlen(lib);
      if (len > 3 && (!strcmp(lib+len-3, ".sl") ||
                      !strcmp(lib+len-3, ".dl") ||
                      !strcmp(lib+len-3, ".so")))
          name = Form("%s[000000]%s",Getenv("ROOTSYS"),lib);	//temporary
         //name = gSystem->Which(GetDynamicPath(), lib, kReadPermission);
      else {
         Error("DynamicPathName",
               "shared library must have extension: .so, .sl or .dl", lib);
         name = 0;
      }
   }
   return name;
}

//______________________________________________________________________________
void *TVmsSystem::FindDynLib(const char *lib)
{
   // Returns the handle to a loaded shared library. Returns 0 when library
   // not loaded.

#ifdef R__HPUX
   const char *path;

   if ((path = gSystem->DynamicPathName(lib))) {
      // find handle of shared library using its name
      struct shl_descriptor *desc;
      int index = 0;
      while (shl_get(index++, &desc) == 0)
         if (!strcmp(path, desc->filename))
            return desc->handle;
   }
#endif

   if (lib) { }  // avoid warning, use lib

   return 0;
}

//______________________________________________________________________________
int TVmsSystem::VmsDynLoad(const char *lib)
{
   // Load a shared library. Returns 0 on successful loading, 1 in
   // case lib was already loaded and -1 in case lib does not exist
   // or in case of error.

   const char *path;

   if ((path = gSystem->DynamicPathName(lib))) {
#if defined(R__HPUX) && defined(NOCINT)
      shl_t handle = cxxshl_load(path, BIND_IMMEDIATE | BIND_NONFATAL, 0L);
      if (handle > 0) return 0;
#else
      return G__loadfile((char *)path);
#endif
   }
   return -1;
}

//______________________________________________________________________________
Func_t TVmsSystem::VmsDynFindSymbol(const char *lib, const char *entry)
{
   // Finds and returns a function pointer to a symbol in the shared library.
   // Returns 0 when symbol not found.

#ifdef R__HPUX
   shl_t handle;

   if (handle = (shl_t)FindDynLib(lib)) {
      Func_t addr = 0;
      if (shl_findsym(&handle, entry, TYPE_PROCEDURE, addr) == -1)
         ::SysError("TVmsSystem::VmsDynFindSymbol", "shl_findsym");
      return addr;
   }
   return 0;
#else
   if (lib || entry) { }

   // Always assume symbol found
   return (Func_t)1;
#endif
}

//______________________________________________________________________________
void TVmsSystem::VmsDynListSymbols(const char *lib, const char *regexp)
{
   // List symbols in a shared library. One can use wildcards to list only
   // the intresting symbols.

#ifdef R__HPUX
   shl_t handle;

   if (handle = (shl_t)FindDynLib(lib)) {
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
            Printf("Functions exported by library %s", gSystem->DynamicPathName(lib));
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
   if (lib || regexp) { }
}

//______________________________________________________________________________
void TVmsSystem::VmsDynListLibs(const char *lib)
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
   if (lib) { }
}

//______________________________________________________________________________
void TVmsSystem::VmsDynUnload(const char *lib)
{
   // Unload a shared library.

#if defined(R__HPUX) && defined (NOCINT)
   shl_t handle;

   if (handle = (shl_t)FindDynLib(lib))
      if (cxxshl_unload(handle) == -1)
         ::SysError("TVmsSystem::VmsDynUnload", "could not unload library %s", lib);
#else
   if (lib) { }
   // should call CINT unload file here, but does not work for sl's yet.
   ::Warning("TVmsSystem::VmsDynUnload", "CINT does not support unloading shared libs");
#endif
}

void TVmsSystem::SetDisplay()
{//***May want to add later

}


