// @(#)root/mac:$Name:  $:$Id: TMacSystem.h,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
// Author: Fons Rademakers   24/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMacSystem
#define ROOT_TMacSystem


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMacSystem                                                           //
//                                                                      //
// Class providing an interface to the Macintosh (MacOS) Operating      //
// System.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>

#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


class TMacSystem : public TSystem {

protected:
   TString     fHostname;           //Hostname
   RgnHandle   fMouseMovedRegion;   //Only used in DispatchOneEvent (like a static)
   Bool_t      fHasWaitNextEvent;   //Set in ctor, only used in TMacApplication::Run()
   TString     fWdPath;             //Current working directory
   MenuHandle  fAppleMenu, fFileMenu, fEditMenu, fClassMenu;

   void SetupMenu();

   // static functions providing semi-low level interface to raw MacOS
   static void VeryFirstInit();

public:
   TMacSystem();
   ~TMacSystem();

   //---- Misc
   Bool_t          Init();
   //void            SetProgname(const char *name);
   //const char     *GetError();
   const char     *HostName();

   //---- EventLoop
   //void            Run();
   //void            DispatchOneEvent();

   //---- Handling of system events
   //void            AddSignalHandler(TSignalHandler *sh);
   //TSignalHandler *RemoveSignalHandler(TSignalHandler *sh);
   //void            AddFileHandler(TFileHandler *fh);
   //TFileHandler   *RemoveFileHandler(TFileHandler *fh);

   //---- Processes
   //Int_t           Exec(const char *shellcmd);
   //FILE           *OpenPipe(const char *command, const char *mode);
   //int             ClosePipe(FILE *pipe);
   void            Exit(int code, Bool_t mode = kTRUE);
   //void            Wait(UInt_t duration);
   void            Abort(int code = 0);
   //int             GetPid();

   //---- Directories
   int             MakeDirectory(const char *name);
   void           *OpenDirectory(const char *name);
   void            FreeDirectory(void *dirp);
   const char     *GetDirEntry(void *dirp);
   Bool_t          ChangeDirectory(const char *path);
   const char     *WorkingDirectory();
   //const char     *HomeDirectory();

   //---- Paths & Files
   const char     *BaseName(const char *pathname);
   const char     *DirName(const char *pathname);
   char           *ConcatFileName(const char *dir, const char *name);
   Bool_t          IsAbsoluteFileName(const char *dir);
   //Bool_t          ExpandPathName(TString &path); // expand the metacharacters in buf as in the shell
   //const char     *ExpandPathName(const char *path);
   Bool_t          AccessPathName(const char *path, EAccessMode mode = kFileExists);
   //void            Rename(const char *from, const char *to);
   //int             Link(const char *from, const char *to);
   //int             Symlink(const char *from, const char *to);
   int             Unlink(const char *name);
   //int             SetNonBlock(int fd);
   int             GetPathInfo(const char *path, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   const char     *UnixPathName(const char *unixpathname);
   //const char     *Which(const char *search, const char *file, EAccessMode mode = kFileExists);

   //---- environment manipulation
   void            Setenv(const char *name, const char *value); // set environment variable name to value
   //void            Unsetenv(const char *name);  // remove environment variable
   const char     *Getenv(const char *env);

   //---- Time & Date
//   TTime           Now();
//   void            AddTimer(TTimer *t);
//   Bool_t          RemoveTimer(TTimer *T);

   //---- Dynamic Loading
   //int             Load(const char *module, const char *entry = "", Bool_t system = kFALSE);
   //void            Unload(const char *module);
   //void            ListSymbols(const char *module, const char *re = "");
   //void            ListLibraries(const char *regexp = "");

   //---- IAC
   //int             OpenConnection(const char *server, const char *service);
   //int             AnnounceTcpService(const char *service);
   //int             AnnounceUnixService(const char *server);
   //int             AcceptConnection(int sock);

   static Bool_t   fgDebug;

   ClassDef(TMacSystem,0)  //Interface to MacOS services
};

#endif
