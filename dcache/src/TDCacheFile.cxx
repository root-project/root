// @(#)root/dcache:$Name:  $:$Id: TDCacheFile.cxx,v 1.10 2003/07/19 00:14:15 rdm Exp $
// Author: Grzegorz Mazur   20/01/2002
// Modified: William Tanenbaum 01/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDCacheFile                                                          //
//                                                                      //
// A TDCacheFile is like a normal TFile except that it may read and     //
// write its data via a dCache server (for more on the dCache daemon    //
// see http://www-dcache.desy.de/. Given a path which doesn't belong    //
// to the dCache managed filesystem, it falls back to the ordinary      //
// TFile behaviour.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDCacheFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <dcap.h>
#ifndef R__WIN32
#include <unistd.h>
#if defined(R__SUN) || defined(R__SGI) || defined(R__HPUX) || \
    defined(R__AIX) || defined(R__LINUX) || defined(R__SOLARIS) || \
            defined(R__ALPHA) || defined(R__HIUX) || defined(R__FBSD) || \
	                defined(R__MACOSX) || defined(R__HURD)
#define HAS_DIRENT
#endif
#endif

#ifdef HAS_DIRENT
#include <dirent.h>
#endif

static const char* const DCACHE_PREFIX = "dcache:";
static const size_t DCACHE_PREFIX_LEN = strlen(DCACHE_PREFIX);
static const char* const DCAP_PREFIX = "dcap:";
static const size_t DCAP_PREFIX_LEN = strlen(DCAP_PREFIX);


ClassImp(TDCacheFile)

//______________________________________________________________________________
TDCacheFile::TDCacheFile(const char *path, Option_t *option,
                         const char *ftitle, Int_t compress):
   TFile(path, "NET", ftitle, compress)
{
   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   fOffset = 0;
   fOption = option;
   fOption.ToUpper();

   if (fOption == "NEW")
      fOption = "CREATE";

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   TString stmp;
   const char *fname;

   if (!strncmp(path, DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      fname = path;
   } else {
      // Metadata provided by PNFS
      char *tname;
      if ((tname = gSystem->ExpandPathName(path))) {
         stmp = tname;
         delete [] tname;
         fname = stmp.Data();
      } else {
         Error("TDCacheFile", "error expanding path %s", path);
         goto zombie;
      }
   }

   if (recreate) {
      if (!gSystem->AccessPathName(fname, kFileExists))
         gSystem->Unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && !gSystem->AccessPathName(fname, kFileExists)) {
      Error("TDCacheFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && gSystem->AccessPathName(fname, kWritePermission)) {
         Error("TDCacheFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   if (read) {
      if (gSystem->AccessPathName(fname, kFileExists)) {
         Error("TDCacheFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (gSystem->AccessPathName(fname, kReadPermission)) {
         Error("TDCacheFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
   }

   // Connect to file system stream
   fRealName = fname;

   if (create || update) {
#ifndef WIN32
      fD = SysOpen(fname, O_RDWR | O_CREAT, 0644);
#else
      fD = SysOpen(fname, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TDCacheFile", "file %s can not be opened", fname);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fD = SysOpen(fname, O_RDONLY, 0644);
#else
      fD = SysOpen(fname, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TFile", "file %s can not be opened for reading", fname);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   // Disable dCache read-ahead buffer, as it doesn't cooperate well
   // with usually non-sequential file access pattern typical for
   // TFile. TCache performs much better, so UseCache() should be used
   // instead.
   dc_noBuffering(fD);

   Init(create);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TDCacheFile::~TDCacheFile()
{
   // Close and cleanup dCache file.

   Close();
}

//______________________________________________________________________________
Bool_t TDCacheFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from remote file via dCache daemon.
   // Returns kTRUE in case of error.

   if (fCache) {
      Int_t st;
      Seek_t off = fOffset;
      if ((st = fCache->ReadBuffer(fOffset, buf, len)) < 0) {
         Error("ReadBuffer", "error reading from cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::ReadBuffer(), reset it
         Seek(off + len);
         return kFALSE;
      }
   }

   return TFile::ReadBuffer(buf, len);
}

//______________________________________________________________________________
Bool_t TDCacheFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via dCache daemon.
   // Returns kTRUE in case of error.

   if (!IsOpen() || !fWritable) return kTRUE;

   if (fCache) {
      Int_t st;
      Seek_t off = fOffset;
      if ((st = fCache->WriteBuffer(fOffset, buf, len)) < 0) {
         Error("WriteBuffer", "error writing to cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::WriteBuffer(), reset it
         Seek(off + len);
         return kFALSE;
      }
   }

   return TFile::WriteBuffer(buf, len);
}

//______________________________________________________________________________
Bool_t TDCacheFile::Stage(const char *path, UInt_t after, const char *location)
{
   // Stage() returns kTRUE on success and kFALSE on failure.

   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   dc_errno = 0;

   if (dc_stage(path, after, location) == 0)
      return kTRUE;

   if (dc_errno != 0)
      gSystem->SetErrorStr(dc_strerror(dc_errno));
   else
      gSystem->SetErrorStr(strerror(errno));

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TDCacheFile::CheckFile(const char *path, const char *location)
{
   // Note: Name of the method was changed to avoid collision with Check
   // macro #defined in ROOT.
   // CheckFile() returns kTRUE on success and kFALSE on failure.  In
   // case the file exists but is not cached, CheckFile() returns
   // kFALSE and errno is set to EAGAIN.

   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   dc_errno = 0;

   if (dc_check(path, location) == 0)
      return kTRUE;

   if (dc_errno != 0)
      gSystem->SetErrorStr(dc_strerror(dc_errno));
   else
      gSystem->SetErrorStr(strerror(errno));

   return kFALSE;
}

//______________________________________________________________________________
void TDCacheFile::SetOpenTimeout(UInt_t n)
{
   dc_setOpenTimeout(n);
}

//______________________________________________________________________________
void TDCacheFile::SetOnError(OnErrorAction a)
{
   dc_setOnError(a);
}

//______________________________________________________________________________
void TDCacheFile::SetReplyHostName(const char *host_name)
{
   dc_setReplyHostName((char*)host_name);
}

//______________________________________________________________________________
const char *TDCacheFile::GetDcapVersion()
{
   return getDcapVersion();
}

//______________________________________________________________________________
Bool_t TDCacheFile::EnableSSL()
{
#ifdef DCAP_USE_SSL
   dc_enableSSL();
   return kTRUE;
#else
   return kFALSE;
#endif
}

//______________________________________________________________________________
Int_t TDCacheFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   dc_errno = 0;

   Int_t rc = dc_open(pathname, flags, (Int_t) mode);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysClose(Int_t fd)
{
   dc_errno = 0;

   Int_t rc = dc_close(fd);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   fOffset += len;

   dc_errno = 0;

   Int_t rc = dc_read(fd, buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   fOffset += len;

   dc_errno = 0;

   Int_t rc =  dc_write(fd, (char *)buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Seek_t TDCacheFile::SysSeek(Int_t fd, Seek_t offset, Int_t whence)
{
   if (whence == SEEK_SET && offset == fOffset) return offset;

   dc_errno = 0;

   Seek_t rc = dc_lseek(fd, offset, whence);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   } else
      fOffset = rc;

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysSync(Int_t)
{
   // dCache always keep it's files sync'ed, so there's no need to
   // sync() them manually.

   return 0;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysStat(Int_t, Long_t *id, Long_t *size,
                           Long_t *flags, Long_t *modtime)
{
   // Return file stat information. The interface and return value is
   // identical to TDCacheSystem::GetPathInfo(). The function returns 0 in
   // case of success and 1 if the file could not be stat'ed.

   return gSystem->GetPathInfo(GetName(), id, size, flags, modtime);
}

//______________________________________________________________________________
void TDCacheFile::ResetErrno() const
{
   // Method resetting the dc_errno and errno.

   dc_errno = 0;
   TSystem::ResetErrno();
}

//______________________________________________________________________________
TString TDCacheFile::GetDcapPath(const char *path)
{
   // Transform the input path into a path usuable by 
   // the dcap C library.
   // i.e either dcap://nodename.org/where/filename.root
   // either //pnfs/where/filename.root

   if (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN)) {
      path += DCACHE_PREFIX_LEN;
   }
   if (!strncmp(path, DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      path += DCAP_PREFIX_LEN;
   }
   TString pathString(path);
   if (!strncmp(path, "//", 2)) {
      pathString  = DCAP_PREFIX + pathString;
   }
   return pathString;
}

//______________________________________________________________________________
TDCacheSystem::TDCacheSystem() : TSystem("-DCache", "DCache Helper System")
{
   // Create helper class that allows directory access .

   // name must start with '-' to bypass the TSystem singleton check
   SetName("DCache");

   fDirp = 0;
}


//---- Directories -------------------------------------------------------------

//______________________________________________________________________________
int TDCacheSystem::MakeDirectory(const char *name)
{
   // The DCache Library does not yet have a mkdir function. 
   // For now, just invoke the standard UNIX functionality.

   return ::mkdir(name, 0755);
}

//______________________________________________________________________________
void *TDCacheSystem::OpenDirectory(const char *name)
{
   // The DCache Library does not yet have a opendir function. 
   // For now, just invoke the standard UNIX functionality.

   struct stat finfo;

   if (stat(name, &finfo) < 0)
     return 0;

   if (!S_ISDIR(finfo.st_mode))
     return 0;

   return (void*) opendir(name);

}

//______________________________________________________________________________
void TDCacheSystem::FreeDirectory(void *dirp)
{
   // The DCache Library does not yet have a closedir function. 
   // For now, just invoke the standard UNIX functionality.

   if (dirp)
      ::closedir((DIR*)dirp);
}

#if defined(_POSIX_SOURCE)
// Posix does not require that the d_ino field be present, and some
// systems do not provide it.
#   define REAL_DIR_ENTRY(dp) 1
#else
#   define REAL_DIR_ENTRY(dp) (dp->d_ino != 0)
#endif

//______________________________________________________________________________
const char *TDCacheSystem::GetDirEntry(void *dirp1)
{
   // The DCache Library does not yet have a readdir function. 
   // For now, just invoke the standard UNIX functionality.

   DIR *dirp = (DIR*)dirp1;
#ifdef HAS_DIRENT
   struct dirent *dp;
#else
   struct direct *dp;
#endif

   if (dirp) {
      for (;;) {
         dp = readdir(dirp);
         if (dp == 0)
            return 0;
         if (REAL_DIR_ENTRY(dp))
            return dp->d_name;
      }
   }
   return 0;
}

//---- Paths & Files -----------------------------------------------------------

//______________________________________________________________________________
Bool_t TDCacheSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the Unix access(2) function.
   // Attention, bizarre convention of return value!!

   // The DCache Library does not yet have an access function.  Use dc_stat()

   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   struct stat statbuf;

   if (path != 0 && dc_stat(path, &statbuf) >= 0) {
     switch (mode) {
       case kReadPermission:
         if (statbuf.st_mode | S_IRUSR) return kFALSE;
         break;
       case kWritePermission:
         if (statbuf.st_mode | S_IWUSR) return kFALSE;
         break;
       case kExecutePermission:
         if (statbuf.st_mode | S_IXUSR) return kFALSE;
	 break;
       default:
         return kFALSE;
     }
   }

   fLastErrorString = GetError();
   return kTRUE;
}

//______________________________________________________________________________
int TDCacheSystem::GetPathInfo(const char *path, Long_t *id, Long_t *size,
                               Long_t *flags, Long_t *modtime)
{
   // Get info about a file: id, size, flags, modification time.
   // Id      is (statbuf.st_dev << 24) + statbuf.st_ino
   // Size    is the file size
   // Flags   is file type: 0 is regular file, bit 0 set executable,
   //                       bit 1 set directory, bit 2 set special file
   //                       (socket, fifo, pipe, etc.)
   // Modtime is modification time.
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   struct stat statbuf;

   if (path != 0 && dc_stat(path, &statbuf) >= 0) {
      if (id)
#if defined(R__KCC) && defined(R__LINUX)
         *id = (statbuf.st_dev.__val[0] << 24) + statbuf.st_ino;
#else
         *id = (statbuf.st_dev << 24) + statbuf.st_ino;
#endif
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
