// @(#)root/mac:$Name:  $:$Id: TMacSystem.cxx,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
// Author: Fons Rademakers   24/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMacSystem                                                           //
//                                                                      //
// Class providing an interface to the Macintosh Operating System.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMacSystem.h"
#include "TROOT.h"
#include "TError.h"
#include "TMath.h"
#include "TOrdCollection.h"
#include "TRegexp.h"
#include "TException.h"
#include "TEnv.h"
#include "root_prefix_mac.h"

#ifdef R__MWERKS
#  define SIOUX
#endif

#ifdef SIOUX
#  include "SIOUX.h"
#endif

#include <stdlib.h>
#include <stat.h>
#include <unistd.h>
#include <utsname.h>
#include <errno.h>

typedef union {
   DirInfo    d;
   FileParam  f;
   HFileInfo  hf;
} mpb;

#define ms2ticks(x) ((x)/17)
#define ticks2ms(x) ((x)*17)
#define mac_now() ticks2ms(*((long*)0x016a))
#define C2Pcpy(to,from) strncpy((char*)&to[1], from, to[0] = strlen(from))

// Difference in origin between Mac and Unix clocks
#define TIMEDIFF ((unsigned long) \
   (((1970-1904)*365 + (1970-1904)/4) * 24 * 3600))

// Macro to find out whether we can do HFS-only calls
#define FSFCBLen (*(short*) 0x3f6)
#define hfsrunning() (FSFCBLen > 0)

// Universal constants
const int  kMAXPATH   = 256;
const char kSEP       = ':';
const int  kMAXNAMLEN = 31;

Bool_t TMacSystem::fgDebug = kFALSE;


//______________________________________________________________________________
static int FileInfo(const char *path, mpb &abp)
{
   mpb pb;
   char ppath[kMAXPATH];
   int err;

   if (path == 0 || strlen(path) <= 0)
      return 1;

   C2Pcpy(ppath, path);

   if (TMacSystem::fgDebug)
      Printf("FileInfo(%s)", &ppath[1]);

   // setup the parameter block and make a synchronous PB call
   pb.d.ioCompletion = 0;
   pb.d.ioNamePtr    = (unsigned char*)ppath;
   pb.d.ioVRefNum    = 0;
   pb.d.ioFDirIndex  = 0;
   pb.hf.ioDirID     = 0L;
   if (hfsrunning())
      err = PBGetCatInfoSync((CInfoPBPtr)&pb);
   else
      err = PBGetFInfo((ParmBlkPtr)&pb, kFALSE);
   if (err != noErr) {
#ifndef R__MWERKS
      errno = EIO;
#endif
      return 1;
   }
   abp = pb;
   return 0;
}

//______________________________________________________________________________
static Bool_t IsDirectory(const char *path)
{
   mpb pb;
   if (FileInfo(path, pb))
      return kFALSE;

   if (pb.hf.ioFlAttrib & 16)
      return kTRUE;
   else
      return kFALSE;
}

//---- MacIter ----------------------------------------------------------------

class TMacIter {
public:
   virtual ~TMacIter() { }
   virtual const char *Next() = 0;
};

//---- MacDirIter -------------------------------------------------------------

class TMacDirIter : public TMacIter {
private:
   long  fDirId;
   int   fNextFile;
public:
   TMacDirIter(long id) : fDirId(id), fNextFile(1) { }
   ~TMacDirIter();
   const char *Next();
};

//______________________________________________________________________________
TMacDirIter::~TMacDirIter()
{
   if (hfsrunning()) {
      WDPBRec pb;
      pb.ioVRefNum = fDirId;
      PBCloseWD(&pb, kFALSE);
   }
}

//______________________________________________________________________________
const char *TMacDirIter::Next()
{
   mpb pb;

   short err;
   static char name[kMAXPATH];
   do {  // until file is visible
      name[0] = 0;
      pb.d.ioNamePtr = (unsigned char*)name;
      pb.d.ioVRefNum = fDirId;
      pb.d.ioFDirIndex = fNextFile++;
      pb.d.ioDrDirID = 0;
      if (hfsrunning())
         err = PBGetCatInfo((CInfoPBPtr)&pb, kFALSE);
      else
         err = PBGetFInfo((ParmBlkPtr)&pb, kFALSE);
      if (err != noErr)
         return 0;
   } while ((pb.hf.ioFlFndrInfo.fdFlags & 0x4000) > 0);

   p2cstr((unsigned char*)name);

   return name;
}

//---- MacVolumeIter ----------------------------------------------------------

class TMacVolumeIter : public TMacIter {
   int fNextVolume;
public:
   TMacVolumeIter() : fNextVolume(1) { }
   const char *Next();
};

//______________________________________________________________________________
const char *TMacVolumeIter::Next()
{
   static unsigned char nameBuf[kMAXPATH];
   VolumeParam vpb;

   nameBuf[0] = 0;
   vpb.ioVolIndex = fNextVolume++;
   vpb.ioNamePtr = nameBuf;
   if (PBGetVInfo((ParamBlockRec*)&vpb, FALSE) != noErr)
      return 0;
   p2cstr(nameBuf);
   return (const char*) nameBuf;
}

//______________________________________________________________________________
static Bool_t IsVolume(const char *path)
{
   TMacVolumeIter iter;
   const char *name;

   while (name = iter.Next())
      if (strcmp(path, name) == 0)
         return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
static int VolumeId(const char *path)
{
   TMacVolumeIter iter;
   const char *name;

   for (int i = 1; name = iter.Next(); i++)
      if (strcmp(path, name) == 0)
         return i;
   return 0;
}


ClassImp(TMacSystem)

//______________________________________________________________________________
TMacSystem::TMacSystem() : TSystem("MacOS", "MacOS System")
{
   fMouseMovedRegion = 0;
}

//______________________________________________________________________________
TMacSystem::~TMacSystem()
{
   if (fMouseMovedRegion) {
      DisposeRgn(fMouseMovedRegion);
      fMouseMovedRegion = 0;
   }
}

//______________________________________________________________________________
Bool_t TMacSystem::Init()
{
   // Initialize MacOS system interface.

   fMouseMovedRegion = 0;

   // check to see if WaitNextEvent is implemented (else we use GetNextEvent)
   SysEnvRec theSysEnv;
   fHasWaitNextEvent = kFALSE;
   SysEnvirons(1, &theSysEnv);
   if (theSysEnv.machineType >= 1 &&
       NGetTrapAddress(0xA860, (TrapType)1) != (UniversalProcPtr)0xA89F)
      fHasWaitNextEvent = kTRUE;

   //VeryFirstInit();
   //SetupMenu();

   // if this allocs memory, it should be behind the mac init calls
   if (TSystem::Init())
      return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TMacSystem::VeryFirstInit()
{
   // Initialize the MacOS toolbox.

   MaxApplZone();
   FlushEvents(everyEvent-diskMask, 0);
   InitGraf(&qd.thePort);
   InitFonts();
   InitWindows();
   InitMenus();
   TEInit();
   InitDialogs(nil);
   InitCursor();

#if defined(SIOUX)
   SIOUXSettings.standalone   = kFALSE;
   SIOUXSettings.setupmenus   = kFALSE;
   SIOUXSettings.initializeTB = kFALSE;

   setvbuf(stderr, new char[BUFSIZ], _IOLBF, BUFSIZ);
#endif
}

//______________________________________________________________________________
void TMacSystem::SetupMenu()
{
   fAppleMenu = GetMenu(1); // Apple Menu
   if (fAppleMenu) {
      AddResMenu(fAppleMenu, 'DRVR');  // Add all the desk accessories
      InsertMenu(fAppleMenu, 0);
   }

   fFileMenu = GetMenu(2);
   if (fFileMenu)
      InsertMenu(fFileMenu, 0);

   fEditMenu = GetMenu(3);
   if (fEditMenu)
      InsertMenu(fEditMenu, 0);

#if 0
   fAppMenu = GetMenu(4);
   if (fAppMenu) {
      TClass *cl;
      Str255 name;
      TIter next(gROOT->ListOfClass());
      while (cl = (TClass*)next()) {
         C2Pcpy(name, cl->GetName());
         AppendMenu(fAppMenu, name);
      }
      InsertMenu(fAppMenu, 0);
   }
#endif

   DrawMenuBar();
}

//______________________________________________________________________________
const char *TMacSystem::HostName()
{
   // Return the system's host name.

   if (fHostname == "") {
      struct utsname uts;
      uname(&uts);
      fHostname = uts.nodename;
   }
   return (const char *)fHostname;
}

//______________________________________________________________________________
const char *TMacSystem::Getenv(const char *key)
{
   if (strcmp(key, "ROOTSYS") == 0)
      return ":root";
   return 0;
}

//______________________________________________________________________________
void TMacSystem::Setenv(const char *name, const char *value)
{
   Printf("TMacSystem::Setenv(%s, %s)", name, value);
}

//______________________________________________________________________________
Bool_t TMacSystem::AccessPathName(const char *path, EAccessMode mode)
{
   mpb pb;
   return FileInfo(path, pb);
}

//______________________________________________________________________________
int TMacSystem::GetPathInfo(const char *path, Long_t *id, Long_t *size,
                            Long_t *flags, Long_t *modtime)
{
   int err;
   mpb pb;

   if (*path == '¥') {
      path += 2;
      if (IsVolume(path)) {
         if (id)
            *id = VolumeId(path);
         if (flags) {
            *flags = 0;
            *flags |= 2;
         }
         return 0;
      }
   }

   if (err = FileInfo(path, pb))
      return err;

   if (id)
      *id = pb.hf.ioDirID;

   if (size) {
      Bool_t isHierarchical = (pb.hf.ioFlAttrib & 0x0010) > 0;
      if (isHierarchical)
         *size = 0;
      else
         *size = pb.hf.ioFlLgLen; // logical data fork length
   }
   if (modtime)
      *modtime = pb.hf.ioFlMdDat; // access/mod dates the same
   if (flags) {
      *flags = 0;
      if (pb.hf.ioFlFndrInfo.fdType == 'APPL')
         *flags |= 1;
      if (pb.hf.ioFlAttrib & 16)
         *flags |= 2;
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TMacSystem::ChangeDirectory(const char *path)
{
   if (path && strlen(path) > 0) {
      char ppath[kMAXPATH];
      C2Pcpy(ppath, path);
      if (ppath[ppath[0]] != kSEP) {
         ppath[ppath[0]+1] = kSEP;
         ppath[ppath[0]+2] = 0;
         ppath[0]++;
      }

      WDPBRec wdpb;
      wdpb.ioNamePtr = (unsigned char*)ppath;
      if (PBHSetVol(&wdpb, kFALSE) != noErr) {
         Error("ChangeDirectory", "PBHSetVol to %s failed", path);
         return kFALSE;
      }
      fWdPath = path;
   }
   return kTRUE;
}

//______________________________________________________________________________
int TMacSystem::MakeDirectory(const char *name)
{
   // Make a MacOS file system directory. Returns 0 in case of success and
   // -1 if the directory could not be created.

   return ::mkdir(name, 0755);
}

//______________________________________________________________________________
void *TMacSystem::OpenDirectory(const char *path)
{
   char ppath[kMAXPATH];
   short err;

   union {
      WDPBRec d;
      VolumeParam v;
   } pb;

   if (*path == '¥')
      fprintf(stderr, "TMacSystem::MakeDirectory: %s\n", path);

   if (strcmp(path, ".") == 0)
      path = WorkingDirectory();

   if (strcmp(path, "¥:") == 0 || strcmp(path, "¥") == 0)
      return new TMacVolumeIter;

   if (strncmp(path, "¥:", 2) == 0)
      path+= 2;

   C2Pcpy(ppath, path);
   if (ppath[ppath[0]] != kSEP) {
      ppath[ppath[0]+1] = kSEP;
      ppath[ppath[0]+2] = 0;
      ppath[0]++;
   }

   if (TMacSystem::fgDebug)
      fprintf(stderr, "TMacSystem::MakeDirectory: %s\n", &ppath[1]);

   pb.d.ioNamePtr = (unsigned char*)ppath;
   pb.d.ioVRefNum = 0;
   if (hfsrunning()) {
      pb.d.ioWDProcID = 0;
      pb.d.ioWDDirID  = 0;
      err = PBOpenWD((WDPBPtr)&pb, kFALSE);
   } else {
      pb.v.ioVolIndex = 0;
      err = PBGetVInfo((ParmBlkPtr)&pb, kFALSE);
   }
   if (err != noErr)
      return 0;
   return new TMacDirIter(pb.d.ioVRefNum);
}

//______________________________________________________________________________
void TMacSystem::FreeDirectory(void *d)
{
   TMacIter *dirp = (TMacIter*) d;
   if (dirp)
      delete dirp;
}

//______________________________________________________________________________
const char *TMacSystem::GetDirEntry(void *d)
{
   TMacIter *dirp = (TMacIter*) d;
   if (dirp)
      return dirp->Next();
   return 0;
}

//______________________________________________________________________________
const char *TMacSystem::WorkingDirectory()
{
   // Return working directory.

   if (fWdPath != "")
      return fWdPath.Data();

   static char cwd[kMAXPATH];
   if (getcwd(cwd, kMAXPATH) == 0) {
      fWdPath = ":";
      Error("WorkingDirectory", "getcwd() failed");
   }
   fWdPath = cwd;
   return fWdPath.Data();
}

//______________________________________________________________________________
const char *TMacSystem::BaseName(const char *nm)
{
   if (nm && strlen(nm) > 0) {
      static char name[kMAXPATH];
      char *cp;
      int l = strlen(nm);
      strcpy(name, nm);
      if (name[l-1] == kSEP)
         name[l-1] = 0;
      if (cp = strrchr(name, kSEP))
         return ++cp;
      return name;
   }
   return "MAC";
}

// volume:folder:file      absolute pathnames
// volume:folder:          absolute pathnames
// volume:                 absolute pathnames
// :folder:file            relative pathnames
// :folder:                relative pathnames
// :file                   relative pathnames
// file                    relative pathnames

//______________________________________________________________________________
Bool_t TMacSystem::IsAbsoluteFileName(const char *path)
{
   if (TMacSystem::fgDebug)
      fprintf(stderr, "TMacSystem::IsAbsoluteFileName(%s)\n", path);
   if (*path == '¥')
      return kTRUE;
   if (path && strlen(path) > 0) {
      if (strchr(path, kSEP))
         return path[0] != kSEP;
      return IsVolume(path);
   }
   return kFALSE;
}

// strange mac file conventions: for volume names, we need
// the trailing colon, i.e. "Disk:", even though "Disk:DataÄ" is legal also
//______________________________________________________________________________
const char *TMacSystem::DirName(const char *path)
{
   if (strchr(path, kSEP)) {
      static char buf[kMAXPATH];
      strcpy(buf, path);
      if (buf[strlen(buf)-1] == kSEP)
         buf[strlen(buf)-1] = 0;
      char *r = strrchr(buf, kSEP);
      if (r != buf)
         *r= '\0';
      return buf;
   }
   if (IsVolume(path))
      return "¥";
   if (TMacSystem::fgDebug)
      fprintf(stderr, "TMacSystem::DirName(%s)\n", path);
   return 0;
}

//______________________________________________________________________________
char *TMacSystem::ConcatFileName(const char *dir, const char *name)
{
   // Concatenate directory and file using proper separator. Returned
   // string must be deleted by user.

   if (name == 0 || strlen(name) <= 0 || strcmp(name, ".") == 0 || strcmp(name, ":") == 0)
      return StrDup(dir);

   if (dir && strlen(dir) > 0) {
      char buf[1000];
      strcpy(buf, dir);
      if (buf[strlen(dir)-1] == kSEP)
         buf[strlen(dir)-1] = 0;
      strcat(buf, ":");
      if (name[0] == kSEP)
         name++;
      strcat(buf, name);
      return StrDup(buf);
   } else
      return StrDup(name);
}

//______________________________________________________________________________
const char *TMacSystem::UnixPathName(const char *name)
{
   static char buf[400];
   const char *cp;
   char *bp = buf;
   int c;
   for (cp = name; c = *cp; cp++)
      if (c == '/')
         *bp++ = ':';
      else
         *bp++ = c;
   *bp = 0;
   return buf;
}

//______________________________________________________________________________
int TMacSystem::Unlink(const char *name)
{
   // Unlink, i.e. remove, a file or directory.

   if (IsDirectory(name))
      return ::rmdir(name);
   else
      return ::remove(name);
}

//______________________________________________________________________________
void TMacSystem::Exit(int code, Bool_t mode)
{
   // Exit the application.

   if (mode)
      ::exit(code);
   else
      ::exit(code);  //::_exit(code);
}

//______________________________________________________________________________
void TMacSystem::Abort(int)
{
   // Abort the application.

   ::abort();
}
