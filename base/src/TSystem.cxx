// @(#)root/base:$Name:  $:$Id: TSystem.cxx,v 1.8 2000/12/06 07:17:43 brun Exp $
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
// TSystem                                                              //
//                                                                      //
// Abstract base class defining a generic interface to the underlying   //
// Operating System.                                                    //
// This is not an ABC in the strict sense of the (C++) word. For        //
// every member function their is an implementation (often not more     //
// than a call to AbstractMethod() which prints a warning saying        //
// that the method should be overridden in a derived class), which      //
// allows a simple partial implementation for new OS'es.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <errno.h>
#include <fstream.h>

#include "TSystem.h"
#include "TApplication.h"
#include "TException.h"
#include "TSysEvtHandler.h"
#include "TROOT.h"
#include "TBrowser.h"
#include "TString.h"
#include "TOrdCollection.h"
#include "TCint.h"
#include "TRegexp.h"
#include "TTimer.h"
#include "TObjString.h"

#include "compiledata.h"


const char *gSystemName;
const char *gRootDir;
const char *gProgName;
const char *gRootName;
const char *gProgPath;

TSystem  *gSystem;
TSysEvtHandler *gXDisplay = 0;  // Display server event handler, set in TGClient

ClassImp(TProcessEventTimer)

//______________________________________________________________________________
TProcessEventTimer::TProcessEventTimer(Long_t delay) : TTimer(delay, kFALSE)
{
   // Create async event processor timer. Delay is in milliseconds.

   gROOT->SetInterrupt(kFALSE);
   TurnOn();
}

//______________________________________________________________________________
Bool_t TProcessEventTimer::ProcessEvents()
{
   // Process events if timer did time out. Returns kTRUE if intterrupt
   // flag is set (by hitting a key in the canvas or selecting the
   // Interrupt menu item in canvas or some other action).

   if (fTimeout) {
      if (gSystem->ProcessEvents()) {
         Remove();
         return kTRUE;
      } else {
         Reset();
         return kFALSE;
      }
   }
   return kFALSE;
}



ClassImp(TSystem)

//______________________________________________________________________________
TSystem::TSystem(const char *name, const char *title) : TNamed(name, title)
{
   // Create a new OS interface.

   fOnExitList    = 0;
   fSignalHandler = 0;
   fFileHandler   = 0;
   fTimers        = 0;
   fCompiled      = 0;
}

//______________________________________________________________________________
TSystem::~TSystem()
{
   // Delete the OS interface.

   if (fOnExitList) {
      fOnExitList->Delete();
      SafeDelete(fOnExitList);
   }

   if (fSignalHandler) {
      fSignalHandler->Delete();
      SafeDelete(fSignalHandler);
   }

   if (fFileHandler) {
      fFileHandler->Delete();
      SafeDelete(fFileHandler);
   }

   if (fTimers) {
      fTimers->Delete();
      SafeDelete(fTimers);
   }

   if (fCompiled) {
      fCompiled->Delete();
      SafeDelete(fCompiled);
   }

   if (gSystem == this)
      gSystem = 0;
}

//______________________________________________________________________________
Bool_t TSystem::Init()
{
   // Initialize the OS interface. Copy the OS name (i.e. Unix) and the
   // ROOT name to gSystemName and gRootName, respectively.

   fNfd    = 0;
   fMaxrfd = 0;
   fMaxwfd = 0;
   fReadmask.Zero();
   fWritemask.Zero();

   fSigcnt = 0;
   fLevel  = 0;

   fSignalHandler = new TOrdCollection;
   fFileHandler   = new TOrdCollection;
   fTimers        = new TOrdCollection;

   fIncludePath   = INCLUDEPATH;
   fLinkedLibs    = LINKEDLIBS;
   fSoExt         = SOEXT;
   fObjExt        = OBJEXT;
   fMakeSharedLib = MAKESHAREDLIB;
   fMakeExe       = MAKEEXE;
   fCompiled      = new TOrdCollection;

   gSystemName = StrDup(fName.Data());
   gRootName   = StrDup(gROOT->GetName());

   if (!fName.CompareTo("Generic")) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TSystem::SetProgname(const char *name)
{
   // Set the application name (from command line, argv[0]) and copy it in
   // gProgName.

   gProgName = StrDup(name);
}

//______________________________________________________________________________
void TSystem::SetDisplay()
{
   // Set DISPLAY environment variable based on utmp entry. Only for UNIX.
}

//______________________________________________________________________________
const char *TSystem::GetError()
{
   // Return system error string.

   return Form("errno: %d", GetErrno());
}

//______________________________________________________________________________
Int_t TSystem::GetErrno()
{
   // Static function returning system error number.

#ifdef R__SOLARIS_CC50
   return ::errno;
#else
   return errno;
#endif
}

//______________________________________________________________________________
void TSystem::ResetErrno()
{
   // Static function resetting system error number.

#ifdef R__SOLARIS_CC50
   ::errno = 0;
#else
   errno = 0;
#endif
}

//______________________________________________________________________________
void TSystem::RemoveOnExit(TObject *obj)
{
   // Objects that should be deleted on exit of the OS interface.

   if (fOnExitList == 0)
      fOnExitList = new TOrdCollection;
   if (fOnExitList->FindObject(obj) == 0)
      fOnExitList->Add(obj);
}

//______________________________________________________________________________
const char *TSystem::HostName()
{
   // Return the system's host name.

   return "Local host";
}

//---- EventLoop ---------------------------------------------------------------

//______________________________________________________________________________
void TSystem::Run()
{
   // System event loop.

   fInControl = kTRUE;
   fDone      = kFALSE;

#ifdef R__EH
   try {
#endif
      RETRY {
         while (!fDone) {
            gApplication->StartIdleing();
            InnerLoop();
            gApplication->StopIdleing();
         }
      } ENDTRY;
#ifdef R__EH
   }
   catch (const char *str) {
      printf("%s\n", str);
   }
   // handle every exception
   catch (...) {
      Warning("Run", "handle uncaugth exception, terminating\n");
   }
#endif

   fInControl = kFALSE;
}

//______________________________________________________________________________
void TSystem::ExitLoop()
{
   // Exit from event loop.

   fDone = kTRUE;
}

//______________________________________________________________________________
void TSystem::InnerLoop()
{
   // Inner event loop.

   fLevel++;
   DispatchOneEvent();
   fLevel--;
}

//______________________________________________________________________________
Bool_t TSystem::ProcessEvents()
{
   // Process pending events (GUI, timers, sockets). Returns the result of
   // TROOT::IsInterrupted(). The interrupt flag (TROOT::SetInterrupt())
   // can be set during the handling of the events. This mechanism allows
   // macros running in tight calculating loops to be interrupted by some
   // GUI event (depending on the interval with which this method is
   // called). For example hitting ctrl-c in a canvas will set the
   // interrupt flag.

   gROOT->SetInterrupt(kFALSE);

   if (!gROOT->IsBatch())
      DispatchOneEvent(kTRUE);

   return gROOT->IsInterrupted();
}

//______________________________________________________________________________
void TSystem::DispatchOneEvent(Bool_t)
{
   // Dispatch a single event.

   AbstractMethod("DispatchOneEvent");
}

//______________________________________________________________________________
void TSystem::Sleep(UInt_t)
{
   // Sleep milliSec milli seconds.

   AbstractMethod("Sleep");
}


//---- handling of system events -----------------------------------------------
//______________________________________________________________________________
TTime TSystem::Now()
{
   // Return current time.

   return TTime(0);
}

//______________________________________________________________________________
void TSystem::AddTimer(TTimer *ti)
{
   // Add timer to list of system timers.

   if (ti && fTimers && (fTimers->FindObject(ti) == 0))
      fTimers->Add(ti);
}

//______________________________________________________________________________
TTimer *TSystem::RemoveTimer(TTimer *ti)
{
   // Remove timer from list of system timers. Returns removed timer or 0
   // if timer was not active.

   if (fTimers) {
      TTimer *tr = (TTimer*) fTimers->Remove(ti);
      return tr;
   }
   return 0;
}

//______________________________________________________________________________
Long_t TSystem::NextTimeOut(Bool_t mode)
{
   // Time when next timer of mode (synchronous=kTRUE or
   // asynchronous=kFALSE) will time-out (in ms).

   if (!fTimers) return -1;

   TIter next(fTimers);
   TTimer *t;
   Long_t  tt, timeout = -1, tnow = Now();

   while ((t = (TTimer *)next())) {
      if (t->IsSync() == mode) {
         tt = (long)t->GetAbsTime() - tnow;
         if (tt < 0) tt = 0;
         if (timeout == -1) timeout = tt;
         if (tt < timeout) timeout = tt;
      }
   }
   return timeout;
}

//______________________________________________________________________________
void TSystem::AddSignalHandler(TSignalHandler *h)
{
   // Add a signal handler to list of system signal handlers.

   if (h && fSignalHandler && (fSignalHandler->FindObject(h) == 0))
      fSignalHandler->Add(h);
}

//______________________________________________________________________________
TSignalHandler *TSystem::RemoveSignalHandler(TSignalHandler *h)
{
   // Remove a signal handler from list of signal handlers.

   if (fSignalHandler) {
      fSignalHandler->Remove(h);
      return h;
   }
   return 0;
}

//______________________________________________________________________________
void TSystem::AddFileHandler(TFileHandler *h)
{
   // Add a file handler to the list of system file handlers.

   if (h && fFileHandler && (fFileHandler->FindObject(h) == 0))
      fFileHandler->Add(h);
}

//______________________________________________________________________________
TFileHandler *TSystem::RemoveFileHandler(TFileHandler *h)
{
   // Remove a file handler from the list of file handlers.

   if (fFileHandler) {
      fFileHandler->Remove(h);
      return h;
   }
   return 0;
}

//______________________________________________________________________________
void TSystem::IgnoreInterrupt(Bool_t)
{
   // Ignore the interrupt signal if ignore == kTRUE else restore previous
   // behaviour. Typically call ignore interrupt before writing to disk.

   AbstractMethod("IgnoreInterrupt");
}

//---- Processes ---------------------------------------------------------------

//______________________________________________________________________________
int TSystem::Exec(const char*)
{
   // Execute a command.

   AbstractMethod("Exec");
   return -1;
}

//______________________________________________________________________________
FILE *TSystem::OpenPipe(const char*, const char*)
{
   // Open a pipe.

   AbstractMethod("OpenPipe");
   return 0;
}

//______________________________________________________________________________
int TSystem::ClosePipe(FILE*)
{
   // Close the pipe.

   AbstractMethod("ClosePipe");
   return -1;
}

//______________________________________________________________________________
int TSystem::GetPid()
{
   // Get process id.

   AbstractMethod("GetPid");
   return -1;
}

//______________________________________________________________________________
void TSystem::Exit(int, Bool_t)
{
   // Exit the application.

   AbstractMethod("Exit");
}

//______________________________________________________________________________
void TSystem::Abort(int)
{
   // Abort the application.

   AbstractMethod("Abort");
}

//______________________________________________________________________________
void TSystem::StackTrace()
{
   // Print a stack trace.

   AbstractMethod("StackTrace");
}


//---- Directories -------------------------------------------------------------

//______________________________________________________________________________
int TSystem::MakeDirectory(const char*)
{
   // Make a directory. Returns 0 in case of success and
   // -1 if the directory could not be created.

   AbstractMethod("MakeDirectory");
   return 0;
}

//______________________________________________________________________________
void *TSystem::OpenDirectory(const char*)
{
   // Open a directory. Returns 0 if directory does not exist.

   AbstractMethod("OpenDirectory");
   return 0;
}

//______________________________________________________________________________
void TSystem::FreeDirectory(void*)
{
   // Free a directory.

   AbstractMethod("FreeDirectory");
}

//______________________________________________________________________________
const char *TSystem::GetDirEntry(void*)
{
   // Get a directory entry. Returns 0 if no more entries.

   AbstractMethod("GetDirEntry");
   return 0;
}

//______________________________________________________________________________
Bool_t TSystem::ChangeDirectory(const char*)
{
   // Change directory.

   AbstractMethod("ChangeDirectory");
   return kFALSE;
}

//______________________________________________________________________________
const char *TSystem::WorkingDirectory()
{
   // Return working directory.

   return 0;
}

//______________________________________________________________________________
const char *TSystem::HomeDirectory(const Char_t*)
{
   // Return the user's home directory.

   return 0;
}


//---- Paths & Files -----------------------------------------------------------

//______________________________________________________________________________
const char *TSystem::BaseName(const char *name)
{
   // Base name of a file name. Base name of /user/root is root.

   if (name) {
      if (name[0] == '/' && name[1] == '\0')
         return name;
      char *cp;
      if ((cp = (char*)strrchr(name, '/')))
         return ++cp;
      return name;
   }
   Error("BaseName", "name = 0");
   return 0;
}

//______________________________________________________________________________
Bool_t TSystem::IsAbsoluteFileName(const char *dir)
{
   // Return true if dir is an absolute pathname.

   if (dir)
      return dir[0] == '/';
   return kFALSE;
}

//______________________________________________________________________________
const char *TSystem::DirName(const char *pathname)
{
   // Return the directory name in pathname. DirName of /user/root is /user.

   if (pathname && strchr(pathname, '/')) {
      static char buf[1000];
      strcpy(buf, pathname);
      char *r = strrchr(buf, '/');
      if (r != buf)
         *r = '\0';
      else
         *(r+1) = '\0';
      return buf;
   }
   return WorkingDirectory();
}

//______________________________________________________________________________
const char *TSystem::UnixPathName(const char *name)
{
   // Convert from a Unix pathname to a local pathname. E.g. from /user/root to \user\root.

   return name;
}

//______________________________________________________________________________
char *TSystem::ConcatFileName(const char *, const char *)
{
   // Concatenate a directory and a file name.

   AbstractMethod("ConcatFileName");
   return 0;
}


//---- Paths & Files -----------------------------------------------------------


//______________________________________________________________________________
const char *TSystem::ExpandFileName(const char *fname)
{
   // Expand a pathname getting rid of special shell characters like ~.$, etc.
   // For Unix/Win32 compatibility use $(XXX) instead of $XXX when using
   // environment variables in a pathname. If compatibility is not an issue
   // you can use on Unix directly $XXX. This is a protected function called
   // from the OS specific system classes, like TUnixSystem and TWinNTSystem.

   const int   kBufSize = 1024;
   int         n, ier, iter, lx, ncopy;
   char       *inp, *out, *c, *b, *e, *x, *t, buff[kBufSize*3];
   const char *p;
   static char xname[kBufSize];

   iter = 0; xname[0] = 0; inp = buff + kBufSize; out = inp + kBufSize;
   inp[-1] = ' '; inp[0] = 0; out[-1] = ' ';
   c = (char *)fname + strspn(fname, " \t\f\r");
   if (isalnum(c[0])) { strcpy(inp, WorkingDirectory()); strcat(inp, "/"); } // add $cwd

   strcat(inp, c);

again:
   iter++; c = inp; ier = 0;
   x = out; x[0] = 0;

   for ( ; (c[0]) && c[0] != ' ' ; c++) {

      p = 0; e = 0;
      if (c[0] == '~' && c[1] == '/') { // ~/ case
         p = HomeDirectory(); e = c + 1; if (!p) ier++;
      }
      if (p) {                         // we have smth to copy
         strcpy(x, p); x += strlen(p); c = e-1; continue;
      }

      p = 0;
      if (c[0] == '~' && c[1] != '/') { // ~user case
         n = strcspn(c+1, "/ "); buff[0] = 0; strncat(buff, c+1, n);
         p = HomeDirectory(buff); e = c+1+n;
      }
      if (p) {                          // we have smth to copy
         strcpy(x,p); x += strlen(p); c = e-1; continue;
      }

      p = 0;
      if (c[0] == '.' && c[1] == '/' && c[-1] == ' ') { // $cwd
         p = strcpy(buff, WorkingDirectory()); e = c + 1; if (!p) ier++;
      }

      if (p) {                          // we have smth to copy */
         strcpy(x,p); x += strlen(p); c = e-1; continue;
      }

      if (c[0] != '$') {                // not $, simple copy
         x++[0] = c[0];
      } else {                          // we have a $
         b = c+1;
         if (c[1] == '(') b++;
         if (c[1] == '{') b++;
         for (e = b; isalnum(e[0]) || e[0] == '_'; e++) ;
         buff[0] = 0; strncat(buff, b, e-b);
         p = Getenv(buff);
         if (!p) {                      // too bad, try UPPER case
            for (t = buff; (t[0] = toupper(t[0])); t++) ;
            p = Getenv(buff);
         }
         if (!p) {                      // too bad, try Lower case
            for (t = buff; (t[0] = tolower(t[0])); t++) ;
            p = Getenv(buff);
         }
         if (!p && !strcmp(buff, "cwd")) { // it is $cwd
            p = strcpy(buff, WorkingDirectory());
         }
         if (!p) {                      // too bad, nothing can help
           ier++; x++[0] = c[0];
         } else {                       // It is OK, copy result
           strcpy(x,p); x += strlen(p); c = (b==c+1) ? e-1 : e;
         }
      }
   }

   x[0] = 0; lx = x - out;
   if (ier && iter < 3) { strcpy(inp,out); goto again; }
   ncopy = (lx >= kBufSize) ? kBufSize-1 : lx;
   xname[0] = 0; strncat(xname,out,ncopy);

   if (ier || ncopy != lx)
      Error("ExpandFileName", "input: %s, output: %s", fname, xname);

   return xname;
}

//______________________________________________________________________________
Bool_t TSystem::ExpandPathName(TString&)
{
   // Expand a pathname getting rid of special shell characaters like ~.$, etc.
   // For Unix/Win32 compatibility use $(XXX) instead of $XXX when using
   // environment variables in a pathname. If compatibility is not an issue
   // you can use on Unix directly $XXX.

   return kFALSE;
}

//______________________________________________________________________________
char *TSystem::ExpandPathName(const char *)
{
   // Expand a pathname getting rid of special shell characaters like ~.$, etc.
   // For Unix/Win32 compatibility use $(XXX) instead of $XXX when using
   // environment variables in a pathname. If compatibility is not an issue
   // you can use on Unix directly $XXX. The user must delete returned string.

   return 0;
}

//______________________________________________________________________________
Bool_t TSystem::AccessPathName(const char*, EAccessMode)
{
   // Returns FALSE if one can access a file using the specified access mode.

   return kFALSE;
}

//______________________________________________________________________________
void TSystem::Rename(const char *, const char *)
{
   // Rename a file.

   AbstractMethod("Rename");
}

//______________________________________________________________________________
int TSystem::Link(const char *, const char *)
{
   // Create a link from file1 to file2.

   AbstractMethod("Link");
   return -1;
}

//______________________________________________________________________________
int TSystem::Symlink(const char *, const char *)
{
   // Create a symbolic link from file1 to file2.

   AbstractMethod("Symlink");
   return -1;
}

//______________________________________________________________________________
int TSystem::Unlink(const char *)
{
   // Unlink, i.e. remove, a file.

   AbstractMethod("Unlink");
   return -1;
}

//______________________________________________________________________________
int TSystem::GetPathInfo(const char*, Long_t*, Long_t*, Long_t*, Long_t*)
{
   // Get info about a file: id, size, flags, modification time.

   AbstractMethod("GetPathInfo");
   return -1;
}

//______________________________________________________________________________
int TSystem::Umask(Int_t)
{
   // Set the process file creation mode mask.

   AbstractMethod("Umask");
   return -1;
}

//______________________________________________________________________________
char *TSystem::Which(const char *, const char *, EAccessMode)
{
   // Find location of file in a search path. User must delete returned string.

   AbstractMethod("Which");
   return 0;
}

//---- environment manipulation ------------------------------------------------

//______________________________________________________________________________
void TSystem::Setenv(const char*, const char*)
{
   // Set environment variable.

   AbstractMethod("Setenv");
}

//______________________________________________________________________________
void TSystem::Unsetenv(const char *name)
{
   // Unset environment variable.
   Setenv(name, 0);
}

//______________________________________________________________________________
const char *TSystem::Getenv(const char*)
{
   // Get environment variable.

   AbstractMethod("Getenv");
   return 0;
}

//---- System Logging ----------------------------------------------------------

//______________________________________________________________________________
void TSystem::Openlog(const char *, Int_t, ELogFacility)
{
   // Open connection to system log daemon. For the use of the options and
   // facility see the Unix openlog man page.

   AbstractMethod("Openlog");
}

//______________________________________________________________________________
void TSystem::Syslog(ELogLevel, const char *)
{
   // Send mess to syslog daemon. Level is the logging level and mess the
   // message that will be written on the log.

   AbstractMethod("Syslog");
}

//______________________________________________________________________________
void TSystem::Closelog()
{
   // Close connection to system log daemon.

   AbstractMethod("Closelog");
}

//---- Dynamic Loading ---------------------------------------------------------

//______________________________________________________________________________
int TSystem::Load(const char *module, const char *entry, Bool_t system)
{
   // Load a shared library. Returns 0 on successful loading, 1 in
   // case lib was already loaded and -1 in case lib does not exist
   // or in case of error. When entry is specified the loaded lib is
   // search for this entry point (return -1 when entry does not exist,
   // 0 otherwise). When the system flag is kTRUE, the library is consisdered
   // a permanent systen library that should not be unloaded during the
   // course of the session.

#ifdef NOCINT
   AbstractMethod("Load");
   return 0;
#else
   char *path;
   int i = -1;
   if ((path = DynamicPathName(module))) {
      if (!system)
         i = G__loadfile(path);
      else
         i = G__loadsystemfile(path);
      delete [] path;
   }

   if (!entry || !strlen(entry)) return i;

   Func_t f = DynFindSymbol(module, entry);
   if (f) return 0;
   return -1;
#endif

}
//______________________________________________________________________________
char *TSystem::DynamicPathName(const char *, Bool_t)
{
   AbstractMethod("DynamicPathName");
   return 0;
}
//______________________________________________________________________________
Func_t TSystem::DynFindSymbol(const char * /*lib*/, const char *entry)
{
   // Find specific entry point in specified library. Specify "*" for lib
   // to search in all libraries.

#ifdef NOCINT
   AbstractMethod("DynFindSymbol");
   return 0;
#else
   return G__findsym(entry);
#endif
}

//______________________________________________________________________________
void TSystem::Unload(const char *)
{
   // Unload a shared library.

   AbstractMethod("UnLoad");
}

//______________________________________________________________________________
void TSystem::ListSymbols(const char *, const char *)
{
   // List symbols in a shared library.

   AbstractMethod("ListSymbols");
}

//______________________________________________________________________________
void TSystem::ListLibraries(const char *regexp)
{
   // List all loaded shared libraries.

   TString libs = GetLibraries(regexp);
   TRegexp separator("[^ \\t\\s]+");
   TString s;
   Ssiz_t start = 0, index = 0, end = 0;
   int i = 0;

   Printf("");
   Printf("Loaded shared libraries");
   Printf("=======================");

   while ((start < libs.Length()) && (index != kNPOS)) {
      index = libs.Index(separator, &end, start);
      if (index >= 0) {
         s = libs(index, end);
         if (s.BeginsWith("-")) {
            if (s.BeginsWith("-l")) {
               Printf("%s", s.Data());
               i++;
            }
         } else {
            Printf("%s", s.Data());
            i++;
         }
      }
      start += end+1;
   }

   Printf("-----------------------");
   Printf("%d libraries loaded", i);
   Printf("=======================");
}

//______________________________________________________________________________
const char *TSystem::GetLibraries(const char *regexp, const char *options)
{
   // Return a space separated list of loaded shared libraries.
   // This list is of a format suitable for a linker, i.e it may contain
   // -Lpathname and/or -lNameOfLib.
   // Option can be any of:
   //   S: shared libraries loaded at the start of the executable, because
   //      they were specified on the link line.
   //   D: shared libraries dynamically loaded after the start of the program.

   fListLibs = "";
   TString libs = "";
   TString opt = options;
   if ((opt.Length()==0) || (opt.First('D')!=kNPOS))
      libs += gInterpreter->GetSharedLibs();

   if ((opt.Length()==0) || (opt.First('S')!=kNPOS)) {
      if (!libs.IsNull()) libs.Append(" ");
      libs.Append(fLinkedLibs);
   }

   // Select according to regexp
   if (regexp!=0 && strlen(regexp)!=0) {
      TRegexp separator("[^ \\t\\s]+");
      TRegexp user_re(regexp, kTRUE);
      TString s;
      Ssiz_t start, index, end;
      start = index = end = 0;

      while ((start < libs.Length()) && (index != kNPOS)) {
         index = libs.Index(separator,&end,start);
         if (index >= 0) {
            s = libs(index,end);
            if (s.Index(user_re) != kNPOS) {
               if (!fListLibs.IsNull())
                  fListLibs.Append(" ");
               fListLibs.Append(s);
            }
         }
         start += end+1;
      }
   } else
      fListLibs = libs;

   return fListLibs;
}

//---- RPC ---------------------------------------------------------------------

//______________________________________________________________________________
TInetAddress TSystem::GetHostByName(const char *)
{
   // Get Internet Protocol (IP) address of host.

   AbstractMethod("GetHostByName");
   return TInetAddress();
}

//______________________________________________________________________________
TInetAddress TSystem::GetPeerName(int)
{
   // Get Internet Protocol (IP) address of remote host and port #.

   AbstractMethod("GetPeerName");
   return TInetAddress();
}

//______________________________________________________________________________
TInetAddress TSystem::GetSockName(int)
{
   // Get Internet Protocol (IP) address of host and port #.

   AbstractMethod("GetSockName");
   return TInetAddress();
}

//______________________________________________________________________________
int TSystem::GetServiceByName(const char *)
{
   // Get port # of internet service.

   AbstractMethod("GetServiceByName");
   return -1;
}

//______________________________________________________________________________
char *TSystem::GetServiceByPort(int)
{
   // Get name of internet service.
   AbstractMethod("GetServiceByPort");
   return 0;
}

//______________________________________________________________________________
int TSystem::OpenConnection(const char*, int)
{
   // Open a connection to another host.

   AbstractMethod("OpenConnection");
   return -1;
}

//______________________________________________________________________________
int TSystem::AnnounceTcpService(int, Bool_t, int)
{
   // Announce TCP/IP service.

   AbstractMethod("AnnounceTcpService");
   return -1;
}

//______________________________________________________________________________
int TSystem::AnnounceUnixService(int, int)
{
   // Announce unix domain service.

   AbstractMethod("AnnounceUnixService");
   return -1;
}

//______________________________________________________________________________
int TSystem::AcceptConnection(int)
{
   // Accept a connection.

   AbstractMethod("AcceptConnection");
   return -1;
}

//______________________________________________________________________________
void TSystem::CloseConnection(int, Bool_t)
{
   // Close socket connection.

   AbstractMethod("CloseConnection");
}

//______________________________________________________________________________
int TSystem::RecvRaw(int, void *, int, int)
{
   // Receive exactly length bytes into buffer. Use opt to receive out-of-band
   // data or to have a peek at what is in the buffer (see TSocket).

   AbstractMethod("RecvRaw");
   return -1;
}

//______________________________________________________________________________
int TSystem::SendRaw(int, const void *, int, int)
{
   // Send exactly length bytes from buffer. Use opt to send out-of-band
   // data (see TSocket).

   AbstractMethod("SendRaw");
   return -1;
}

//______________________________________________________________________________
int TSystem::RecvBuf(int, void *, int)
{
   // Receive a buffer headed by a length indicator.

   AbstractMethod("RecvBuf");
   return -1;
}

//______________________________________________________________________________
int TSystem::SendBuf(int, const void *, int)
{
   // Send a buffer headed by a length indicator.

   AbstractMethod("SendBuf");
   return -1;
}

//______________________________________________________________________________
int TSystem::SetSockOpt(int, int, int)
{
   // Set socket option.

   AbstractMethod("SetSockOpt");
   return -1;
}

//______________________________________________________________________________
int TSystem::GetSockOpt(int, int, int*)
{
   // Get socket option.

   AbstractMethod("GetSockOpt");
   return -1;
}


//---- Script Compiler ---------------------------------------------------------

//______________________________________________________________________________
int TSystem::CompileMacro(const char *filename, Option_t * opt)
{
  // This method compiles and loads a shared library containing
  // the code from the file "filename".
  //
  // The possible options are:
  //     k : keep the shared library after the session end.
  //     f : force recompilation.
  //
  // It uses the directive fMakeSharedLibs to create a shared library.
  // If loading the shared library fails, it tries to output a list of missing
  // symbols by creating an executable (on some platforms like OSF, this does
  // not HAVE to be an executable) containing the script. It uses the
  // directive fMakeExe to do so.
  // For both directives, before passing them to TSystem::Exec, it expands the
  // variables $SourceFiles, $SharedLib, $LibName, $IncludePath, $LinkedLibs,
  // $ExeName and $ObjectFiles. See SetMakeSharedLib() for more information on
  // those variables.
  //
  // This method is used to implement the following feature:
  //
  // Synopsis:
  //
  // The purpose of this addition is to allow the user to use an external
  // compiler to create a shared library from its C++ macro (scripts).
  // Currently in order to execute a script, a user has to type at the root
  // prompt
  //
  //  .X myfunc.C(arg1,arg2)
  //
  // We allow him to type:
  //
  //  .X myfunc.C++(arg1,arg2)
  // or
  //  .X myfunc.C+(arg1,arg2)
  //
  // In which case an external compiler will be called to create a shared
  // library.  This shared library will then be loaded and the function
  // myfunc will be called with the two arguments.  With '++' the shared library
  // is always recompiled.  With '+' the shared library is recompiled only
  // if it does not exist yet or the macro file is newer than the shared
  // library.
  //
  // HOWEVER, CompileMacro currently does NOT detect if any of the file that
  // are included by the macro file has been changed!!!
  //
  // Of course the + and ++ notation is supported in similar way for .x and .L.
  //
  // Through the function TSystem::SetMakeSharedLib(), the user will be able to
  // indicate, with shell commands, how to build a shared library (a good
  // default will be provided). The most common change, namely where to find
  // header files, will be available through the function
  // TSystem::SetIncludePath().
  // A good default will be provided so that a typical user session should be at
  // most:
  //
  // root[1] gSystem->SetIncludePath("-I$ROOTSYS/include
  // -I$HOME/mypackage/include");
  // root[2] .x myfunc.C++(10,20);
  //
  // The user may sometimes try to compile a script before it has loaded all the
  // needed shared libraries.  In this case we want to be helpfull and output a
  // list of the unresolved symbols. So if the loading of the created shared
  // library fails, we will try to build a executable that contains the
  // script. The linker should then output a list of missing symbols.
  //
  // To support this we provide a TSystem::SetMakeExe() function, that sets the
  // directive telling how to create an executable. The loader will need
  // to be informed of all the libraries available. The information about
  // the libraries that has been loaded by .L and TSystem::Load() is accesible
  // to the script compiler. However, the information about
  // the libraries that have been selected at link time by the application
  // builder (like the root libraries for root.exe) are not available and need
  // to be explictly listed in fLinkedLibs (either by default or by a call to
  // TSystem::SetLinkedLibs()).
  //
  // To simplify customization we could also add to the .rootrc support for the
  // variables
  //
  // Unix.*.Root.IncludePath:     -I$ROOTSYS/include
  // WinNT.*.Root.IncludePath:    -I%ROOTSYS%/include
  //
  // Unix.*.Root.LinkedLibs:      -L$ROOTSYS/lib -lBase ....
  // WinNT.*.Root.LinkedLibs:     %ROOTSYS%/lib/*.lib msvcrt.lib ....
  //
  // And also support for MakeSharedLibs() and MakeExe().
  //
  // (the ... have to be replaced by the actual values and are here only to
  // shorten this comment).

  // ======= Analyze the options
  Bool_t keep = kFALSE;
  Bool_t recompile = kFALSE;
  if (opt) {
     keep = (strchr(opt,'k')!=0);
     recompile = (strchr(opt,'f')!=0);
  }
  // if non-zero, build_loc indicates where to build the shared library.
  TString build_loc = "";

  // ======= Get the right file names for the dictionnary and the shared library
  TString library = filename;
  ExpandFileName( library );

  TString file_location( DirName( library ) );
  // so far we do not distinguish
  if (build_loc.Length()==0) build_loc = file_location;

  Ssiz_t dot_pos = library.Last('.');
  TString extension = library;
  extension.Replace( 0, dot_pos+1, 0 , 0);
  TString filename_noext = library;
  filename_noext.Remove( dot_pos );

  // Extension of shared library is platform dependent!!
  library.Replace( dot_pos, library.Length()-dot_pos, 
                   TString("_") + extension + "." + fSoExt );

  TString libname ( BaseName( filename_noext ) );
  libname.Append("_").Append(extension);

  // ======= Check if the library need to loaded or compiled
  if ( gInterpreter->IsLoaded(filename) ) {
     // the script has already been loaded in interpreted mode
     // Let's warn the user and unload it.

     cerr << "script has already been loaded in interpreted mode" << endl;
     cerr << "Unloading " << filename << " and compiling it" << endl;

     if ( G__unloadfile( (char*) filename ) != 0 ) {
       // We can not unload it.
       return(G__LOADFILE_FAILURE);
     }
  }

  Bool_t modified = kFALSE;
  if ( !recompile ) {
     Long_t lib_time, script_time;
     if ( GetPathInfo( library, 0, 0, 0, &lib_time ) == 0 ) {

        // If the time library is older than the script we
        // are going to recompile !
        GetPathInfo( filename, 0, 0, 0, &script_time );
        modified = ( lib_time <= script_time );
        recompile = modified;
     } else {
        recompile = kTRUE;
     }
  }

  if ( gInterpreter->IsLoaded(library)
       || strlen(GetLibraries(library,"D")) != 0 ) {
     // The library has already been built and loaded.

     if (modified)
        cerr << "Modified ";
     else
        cerr << "Unmodified ";
     cerr << "script has already been compiled and loaded. " << endl;
     if ( !recompile ) {
        return G__LOADFILE_SUCCESS;
     } else {
#ifdef R__KCC
        cerr << "Shared library can not be updated (when using the KCC compiler)!" 
             << endl;
        return G__LOADFILE_DUPLICATE;
#else
        // the following is not working in KCC because it seems that dlclose
        // does not properly get rid of the object.  It WILL provoke a
        // core dump at termination.

        cerr << "It will be regenerated and reloaded!" << endl;
        if ( G__unloadfile( (char*) library.Data() ) != 0 ) {
          // The library is being used. We can not unload it.
          return(G__LOADFILE_FAILURE);
        }
        Unlink(library);
#endif
     }

  }
  if (!recompile) {
    // The library already exist, let's just load it.
    return !gSystem->Load(library);
  }

  cerr << "Creating shared library " << library << endl;

  // ======= Select the dictionary name
  TString dict = BaseName( tmpnam(0) );
  // do a basename to remove /var/tmp

  // the file name end up in the file produced
  // by rootcint as a variable name so all character need to be valid!
  dict.ReplaceAll( "-","_" );
  if ( dict.Last('.')!=dict.Length()-1 ) dict.Append(".");
  dict.Prepend( build_loc + "/" ); 
  TString dicth = dict;
  TString dictObj = dict;
  dict += extension;
  dicth += "h";
  dictObj += fObjExt;

  // ======= Generate a linkdef file

  TString linkdef = tmpnam(0);
  linkdef += "linkdef.h";
  ofstream linkdefFile( linkdef, ios::out );
  linkdefFile << "// File Automatically generated by the ROOT Script Compiler " 
              << endl;
  linkdefFile << endl;
  linkdefFile << "#ifdef __CINT__" << endl;
  linkdefFile << endl;
  linkdefFile << "#pragma link off all globals;" << endl;
  linkdefFile << "#pragma link off all classes;" << endl;
  linkdefFile << "#pragma link off all functions;" << endl;
  linkdefFile << "#pragma link C++ nestedclasses;" << endl;
  linkdefFile << "#pragma link C++ nestedtypedefs;" << endl;
  linkdefFile << endl;

  // We want to look for a header file that has the same name as the macro
  // First lets get the include directory list in the dir1:dir2:dir3 format
  TString incPath = GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
  incPath.Append(":").Prepend(" ");
  incPath.ReplaceAll(" -I",":");       // of form :dir1 :dir2:dir3
  while ( incPath.Index(" :") != -1 ) {
    incPath.ReplaceAll(" :",":");
  }
  incPath.Prepend(file_location+":.:");
  if (gDebug>5) cout << "Looking for header in:" << endl << incPath << endl;
  const char * extensions[] = { ".h", ".hh", ".hpp", ".hxx",  ".hPP", ".hXX" };
  for ( int i = 0; i < 6; i++ ) {
    char * name;
    TString lookup = filename_noext;
    lookup.Append(extensions[i]);
    name = Which(incPath,lookup);
    if (name) {
      linkdefFile << "#pragma link C++ defined_in "<<name<<";"<< endl;
      delete name;
    }
  }
  linkdefFile << "#pragma link C++ defined_in "<<filename << ";" << endl;

  linkdefFile << endl;
  linkdefFile << "#endif" << endl;
  linkdefFile.close();

  // ======= Generate the three command lines

  TString rcint = "rootcint -f ";
  rcint.Append(dict).Append(" -c -p ").Append(GetIncludePath()).Append(" ");
  rcint.Append(filename).Append(" ").Append(linkdef);

  TString cmd = fMakeSharedLib;
  // Replace(cmd,filename,library);
  // NOTE: may want to add code to allow for \$SourceFiles to not be
  //       interpreted
  // we do not add filename because it is already include in dicth !
  // dict.Append(" ").Append(filename);
  cmd.ReplaceAll("$SourceFiles",dict);
  cmd.ReplaceAll("$ObjectFiles",dictObj);
  cmd.ReplaceAll("$IncludePath",TString(GetIncludePath()) + " -I" + build_loc);
  cmd.ReplaceAll("$SharedLib",library);
  cmd.ReplaceAll("$LinkedLibs",GetLibraries("","SDL"));
  cmd.ReplaceAll("$LibName",libname);
  cmd.ReplaceAll("$BuildDir",build_loc);

  TString testcmd = fMakeExe;
  TString fakeMain = tmpnam(0);
  fakeMain += extension;
  ofstream fakeMainFile( fakeMain, ios::out );
  fakeMainFile << "// File Automatically generated by the ROOT Script Compiler " 
               << endl;
  fakeMainFile << "int main(char*argc,char**argvv) {};" << endl;
  fakeMainFile.close();
  // We could append this fake main routine to the compilation line.
  // But in this case compiler may output the name of the dictionary file
  // and of the fakeMain file while it compiles it. (this would be useless
  // confusing output).
  // We could also the fake main routine to the end of the dictionnary file
  // however compilation would fail if a main is already there
  // (like stress.cxx)
  // dict.Append(" ").Append(fakeMain);
  TString exec = tmpnam(0);
  testcmd.ReplaceAll("$SourceFiles",dict);
  testcmd.ReplaceAll("$ObjectFiles",dictObj);
  testcmd.ReplaceAll("$IncludePath",GetIncludePath());
  testcmd.ReplaceAll("$ExeName",exec);
  testcmd.ReplaceAll("$LinkedLibs",GetLibraries("","SDL"));
  testcmd.ReplaceAll("$BuildDir",build_loc);
  // ======= Run the build

  if (gDebug>3) {
     cout << "Creating the dictionary files." << endl;
     if (gDebug>4) cout << rcint << endl;
  }
  int result = !gSystem->Exec(rcint);

  if (gDebug>3) {
     cout << "Compiling the dictionary and script files." << endl;
     if (gDebug>4) cout << cmd << endl;
  }
  if (result) result = !gSystem->Exec( cmd );

  if ( result ) {

    if (!keep) fCompiled->Add(new TObjString( library ));

#ifndef NOCINT
    // This is intended to force a failure if not all symbols needed
    // by the library are present.
    G__Set_RTLD_NOW();
#endif
    if (gDebug>3) cout << "Loading the shared library." << endl;    
    result = !gSystem->Load(library);
#ifndef NOCINT
    G__Set_RTLD_LAZY();
#endif

    if ( !result ) {
      if (gDebug>3) {
         cout << "Testing for missing symbols:" << endl; 
         if (gDebug>4) cout << testcmd << endl;
      }
      gSystem->Exec(testcmd);
      gSystem->Unlink( exec );
    }

  };

  if (gDebug<=5) {
     gSystem->Unlink( dict );
     gSystem->Unlink( dicth );
     gSystem->Unlink( dictObj );
     gSystem->Unlink( linkdef );
     gSystem->Unlink( fakeMain );
     gSystem->Unlink( exec );
  } 
  if (gDebug>6) {
     rcint.Prepend("echo ");
     cmd.Prepend("echo \" ").Append(" \" ");
     testcmd.Prepend("echo \" ").Append(" \" ");
     gSystem->Exec(rcint);
     gSystem->Exec( cmd );
     gSystem->Exec(testcmd);
  } 
  
  return result;
}

//______________________________________________________________________________
const char *TSystem::GetMakeSharedLib() const
{
   return fMakeSharedLib;
}

//______________________________________________________________________________
const char *TSystem::GetMakeExe() const
{
   return fMakeExe;
}

//______________________________________________________________________________
const char *TSystem::GetIncludePath()
{
   fListPaths = fIncludePath;
   fListPaths.Append(" ").Append(gInterpreter->GetIncludePath());
   return fListPaths;
}

//______________________________________________________________________________
const char *TSystem::GetLinkedLibs() const
{
   return fLinkedLibs;
}

//______________________________________________________________________________
const char *TSystem::GetSoExt() const
{
   return fSoExt;
}

//______________________________________________________________________________
const char *TSystem::GetObjExt() const
{
   return fObjExt;
}

//______________________________________________________________________________
void TSystem::SetMakeExe(const char *directives)
{
   // Directives has the same syntax as the argument of SetMakeSharedLib but is
   // used to create an executable. This creation is used as a means to output
   // a list of unresolved symbols, when loading a shared library has failed.
   // The required variable is $ExeName rather than $SharedLib, e.g.:
   // gSystem->SetMakeExe(
   // "g++ -Wall -fPIC $IncludePath $SourceFiles
   //  -o $ExeName $LinkedLibs -L/usr/X11R6/lib -lX11 -lm -ldl -rdynamic");

   fMakeExe = directives;
   // NOTE: add verification that the directives has the required variables
}

//______________________________________________________________________________
void TSystem::SetMakeSharedLib(const char *directives)
{
   // Directives should contain the description on how to compile and link a
   // shared lib. This description can be any valid shell command, including
   // the use of ';' to separate several instructions. However, shell specific
   // construct should be avoided. In particular this description can contain
   // environment variables, like $ROOTSYS (or %ROOTSYS% on windows).
   //
   // Five special variables will be expanded before execution:
   //   Variable name       Expands to
   //   -------------       ----------
   //   $SourceFiles        Name of source files to be compiled
   //   $SharedLib          Name of the shared library being created
   //   $LibName            Name of shared library without extension
   //   $BuildDir           Directory where the files will be created 
   //   $IncludePath        value of fIncludePath
   //   $LinkedLibs         value of fLinkedLibs
   //   $ObjectFiles        Name of source files to be compiler with
   //                       their extension changed to .o or .obj
   // e.g.:
   // gSystem->SetMakeSharedLib(
   // "KCC -n32 --strict $IncludePath -K0 -O0 -g $SourceFile
   //  --no_exceptions --signed_chars --display_error_number
   //  --diag_suppress 68 -o $SharedLib");
   //
   // gSystem->setMakeSharedLib(
   // "Cxx $IncludePath -c $SourceFile;
   //  ld  -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx -expect_unresolved
   //  -g0 -O1 -shared /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o
   //  -o $SharedLib $ObjectFile -lcxxstd -lcxx -lexc -lots -lc"
   //
   // gSystem->SetMakeSharedLib(
   // "$HOME/mygcc/bin/g++ -Wall -fPIC $IncludePath $SourceFile
   //  -shared -o $SharedLib");
   //
   // gSystem->SetMakeSharedLib(
   // "cl -DWIN32  -D_WIN32 -D_MT -D_DLL -MD /O2 /G5 /MD -DWIN32
   //  -DVISUAL_CPLUSPLUS -D_WINDOWS $IncludePath $SourceFile
   //  /link -PDB:NONE /NODEFAULTLIB /INCREMENTAL:NO /RELEASE /NOLOGO
   //  $LinkedLibs -entry:_DllMainCRTStartup@12 -dll /out:$SharedLib")

   fMakeSharedLib = directives;
   // NOTE: add verification that the directives has the required variables
}

//______________________________________________________________________________
void TSystem::SetIncludePath(const char *IncludePath)
{
   // IncludePath should contain the list of compiler flags to indicate where
   // to find user defined header files. It is used to expand $IncludePath in
   // the directives given to SetMakeSharedLib() and SetMakeExe(), e.g.:
   //    gSystem->SetInclude("-I$ROOTSYS/include -Imydirectory/include");
   // the default value of IncludePath on Unix is:
   //    "-I$ROOTSYS/include "
   // and on Windows:
   //    "/I%ROOTSYS%/include "

   fIncludePath = IncludePath;
}

//______________________________________________________________________________
void  TSystem::SetLinkedLibs(const char *LinkedLibs)
{
   // LinkedLibs should contain the library directory and list of libraries
   // needed to recreate the current executable. It is used to expand $LinkedLibs
   // in the directives given to SetMakeSharedLib() and SetMakeExe()
   // The default value on Unix is: root-config --glibs

   fLinkedLibs = LinkedLibs;
}

//______________________________________________________________________________
void TSystem::SetSoExt(const char *SoExt)
{
   // Set shared library extension, should be either .so, .sl, .a, .dll, etc.

   fSoExt = SoExt;
}

//______________________________________________________________________________
void TSystem::SetObjExt(const char *ObjExt)
{
   // Set object files extension, should be either .o, .obj, etc.

   fObjExt = ObjExt;
}

//______________________________________________________________________________
void TSystem::CleanCompiledMacros()
{
   // Remove the shared libs produced by the CompileMacro() function.

   TIter next(fCompiled);
   TObjString *lib;
   while ((lib = (TObjString*)next()))
      Unlink(lib->GetString().Data());
}
