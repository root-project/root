// @(#)root/gfal:$Id$
// Author: Fons Rademakers   8/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TGFALFile
\ingroup IO

Read and write data via the underlying Grid access mechanism.

A TGFALFile is like a normal TFile except that it reads and writes
its data via the underlying Grid access mechanism.
TGFALFile file names are either a logical file name, a guid, an
SURL or a TURL, like:
    gfal:/lfn/user/r/rdm/galice.root
Grid storage interactions today require using several existing
software components:
  - The replica catalog services to locate valid replicas of files.
  - The SRM software to ensure:
    - Files  exist on disk (they are recalled from mass storage if necessary) or
    - Space is allocated on disk for new files (they are possibly migrated to mass storage later).
  - A file access mechanism to access files from the storage system on the worker node.
The GFAL library hides these interactions and presents a Posix
interface for the I/O operations. The currently supported protocols
are: file for local access, dcap, gsidcap and kdcap (dCache access
protocol).

### File naming convention
A file name can be a Logical File Name (LFN), a Grid Unique
IDentifier (GUID), a file replica (SURL) or a Transport file
name (TURL):
  - an LFN starts with lfn. Example: \a lfn:baud/testgfal15
  - a GUID starts with guid. Example: \a guid:2cd59291-7ae7-4778-af6d-b1f423719441
  - an SURL starts with srm://. Example: \a srm://wacdr002d.cern.ch:8443/castor/cern.ch/user/b/baud/testgfal15
  - a TURL starts with a protocol name. Example: \a gfal:/lfn/user/r/rdm/galice.root
Note that for the TGFALFile plugin to work, all these pathnames
should be prepended by gfal:.
*/

#include <ROOT/RConfig.hxx>
#include "TROOT.h"
#include "TUrl.h"

#include <gfal_api.h>

// GFAL2 doesn't use special names for 64 bit versions
#if defined(_GFAL2_API_) || defined(GFAL2_API_) || defined(_GFAL2_API) || defined(_GFAL2_API_H_) || defined(GFAL2_API_H_) || defined(_GFAL2_API_H)
#define gfal_lseek64   gfal_lseek
#define gfal_open64    gfal_open
#define gfal_readdir64 gfal_readdir
#define gfal_stat64    gfal_stat
#define dirent64       dirent
#define stat64         stat
#endif

#include "TGFALFile.h"

ClassImp(TGFALFile);
ClassImp(TGFALSystem);

////////////////////////////////////////////////////////////////////////////////
/// Create a GFAL file object.
///
/// A GFAL file is the same as a TFile
/// except that it is being accessed via the underlaying Grid access
/// mechanism. The url argument must be of the form: gfal:/lfn/file.root
/// If the file specified in the URL does not exist, is not accessable
/// or can not be created the kZombie bit will be set in the TGFALFile
/// object. Use IsZombie() to see if the file is accessable.
/// For a description of the option and other arguments see the TFile ctor.
/// The preferred interface to this constructor is via TFile::Open().

TGFALFile::TGFALFile(const char *url, Option_t *option, const char *ftitle,
                     Int_t compress)
         : TFile(url, "NET", ftitle, compress)
{
   fStatCached = kFALSE;

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
   char *fname;
   if ((fname = gSystem->ExpandPathName(fUrl.GetFileAndOptions()))) {
      stmp = fname;
      delete [] fname;
      fname = (char *)stmp.Data();
   } else {
      Error("TGFALFile", "error expanding path %s", fUrl.GetFileAndOptions());
      goto zombie;
   }

   if (recreate) {
      if (::gfal_access(fname, kFileExists) == 0)
         ::gfal_unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && ::gfal_access(fname, kFileExists) == 0) {
      Error("TGFALFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (::gfal_access(fname, kFileExists) != 0) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && ::gfal_access(fname, kWritePermission) != 0) {
         Error("TGFALFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   if (read) {
#ifdef GFAL_ACCESS_FIXED
      if (::gfal_access(fname, kFileExists) != 0) {
         Error("TGFALFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (::gfal_access(fname, kReadPermission) != 0) {
         Error("TGFALFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
#endif
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
         SysError("TGFALFile", "file %s can not be opened", fname);
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
         SysError("TGFALFile", "file %s can not be opened for reading", fname);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   Init(create);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

////////////////////////////////////////////////////////////////////////////////
/// GFAL file dtor. Close and flush directory structure.

TGFALFile::~TGFALFile()
{
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system open. All arguments like in POSIX open.

Int_t TGFALFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   Int_t ret = ::gfal_open64(pathname, flags, (Int_t) mode);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system close. All arguments like in POSIX close.

Int_t TGFALFile::SysClose(Int_t fd)
{
   Int_t ret = ::gfal_close(fd);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system read. All arguments like in POSIX read.

Int_t TGFALFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   Int_t ret = ::gfal_read(fd, buf, len);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system write. All arguments like in POSIX write.

Int_t TGFALFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   Int_t ret = ::gfal_write(fd, buf, len);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system lseek. All arguments like in POSIX lseek
/// except that the offset and return value are Long_t to be able to
/// handle 64 bit file systems.

Long64_t TGFALFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   Long64_t ret = ::gfal_lseek64(fd, offset, whence);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to TSystem:GetPathInfo(). Generally implemented via
/// stat() or fstat().

Int_t TGFALFile::SysStat(Int_t /*fd*/, Long_t *id, Long64_t *size, Long_t *flags,
                         Long_t *modtime)
{
   struct stat64 &statbuf = fStatBuffer;

   if (fOption != "READ" || !fStatCached) {
      // We are not in read mode, or the file status information is not yet
      // in the cache. Update or read the status information with gfal_stat().

      if (::gfal_stat64(fRealName, &statbuf) >= 0)
         fStatCached = kTRUE;
   }

   if (fStatCached) {
      if (id)
         *id = (statbuf.st_dev << 24) + statbuf.st_ino;
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

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via GFAL.
/// Returns kTRUE in case of error.

Bool_t TGFALFile::ReadBuffer(char *buf, Int_t len)
{
   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   return TFile::ReadBuffer(buf, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via GFAL.
/// Returns kTRUE in case of error.

Bool_t TGFALFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   SetOffset(pos);
   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   return TFile::ReadBuffer(buf, pos, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Write specified byte range to remote file via GFAL.
/// Returns kTRUE in case of error.

Bool_t TGFALFile::WriteBuffer(const char *buf, Int_t len)
{
   if (!IsOpen() || !fWritable) return kTRUE;

   Int_t st;
   if ((st = WriteBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   return TFile::WriteBuffer(buf, len);
}

/**
\class TGFALSystem
\ingroup IO

Directory handler for GFAL.
*/

////////////////////////////////////////////////////////////////////////////////
/// Create helper class that allows directory access via GFAL.

TGFALSystem::TGFALSystem() : TSystem("-gfal", "GFAL Helper System")
{
   // name must start with '-' to bypass the TSystem singleton check
   SetName("gfal");

   fDirp = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Make a directory via GFAL.

Int_t TGFALSystem::MakeDirectory(const char *dir)
{
   TUrl url(dir);

   Int_t ret = ::gfal_mkdir(url.GetFileAndOptions(), 0755);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Open a directory via GFAL. Returns an opaque pointer to a dir
/// structure. Returns 0 in case of error.

void *TGFALSystem::OpenDirectory(const char *dir)
{
   if (fDirp) {
      Error("OpenDirectory", "invalid directory pointer (should never happen)");
      fDirp = 0;
   }

   TUrl url(dir);

   struct stat64 finfo;

   if (::gfal_stat64(url.GetFileAndOptions(), &finfo) < 0)
      return 0;

   if ((finfo.st_mode & S_IFMT) != S_IFDIR)
      return 0;

   fDirp = (void*) ::gfal_opendir(url.GetFileAndOptions());

   return fDirp;
}

////////////////////////////////////////////////////////////////////////////////
/// Free directory via GFAL.

void TGFALSystem::FreeDirectory(void *dirp)
{
   if (dirp != fDirp) {
      Error("FreeDirectory", "invalid directory pointer (should never happen)");
      return;
   }

   if (dirp)
      ::gfal_closedir((DIR*)dirp);

   fDirp = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get directory entry via GFAL. Returns 0 in case no more entries.

const char *TGFALSystem::GetDirEntry(void *dirp)
{
   if (dirp != fDirp) {
      Error("GetDirEntry", "invalid directory pointer (should never happen)");
      return 0;
   }

   struct dirent64 *dp;

   if (dirp) {
      dp = ::gfal_readdir64((DIR*)dirp);
      if (!dp)
         return 0;
      return dp->d_name;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

Int_t TGFALSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   TUrl url(path);

   struct stat64 sbuf;

   if (path && ::gfal_stat64(url.GetFileAndOptions(), &sbuf) >= 0) {

      buf.fDev    = sbuf.st_dev;
      buf.fIno    = sbuf.st_ino;
      buf.fMode   = sbuf.st_mode;
      buf.fUid    = sbuf.st_uid;
      buf.fGid    = sbuf.st_gid;
      buf.fSize   = sbuf.st_size;
      buf.fMtime  = sbuf.st_mtime;
      buf.fIsLink = kFALSE;

      return 0;
   }
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns FALSE if one can access a file using the specified access mode.
/// Mode is the same as for the Unix access(2) function.
/// Attention, bizarre convention of return value!!

Bool_t TGFALSystem::AccessPathName(const char *path, EAccessMode mode)
{
   TUrl url(path);

   if (::gfal_access(url.GetFileAndOptions(), mode) == 0)
      return kFALSE;

   return kTRUE;
}
