// @(#)root/dcache:$Name:  $:$Id: TDCacheFile.cxx,v 1.28 2006/04/07 15:35:29 rdm Exp $
// Author: Grzegorz Mazur   20/01/2002
// Modified: William Tanenbaum 01/12/2003
// Modified: Tigran Mkrtchyan 29/06/2004


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
    defined(R__MACOSX) || defined(R__HURD) || defined(R__OBSD)
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
   // Create a dCache file object. A dCache file is the same as a TFile
   // except that it is being accessed via a dCache server. The url
   // argument must be of the form: dcache:/pnfs/<path>/<file>.root or
   // dcap://<nodename.org>/<path>/<file>.root. If the file specified in the
   // URL does not exist, is not accessable or can not be created the kZombie
   // bit will be set in the TDCacheFile object. Use IsZombie() to see if the
   // file is accessable. For a description of the option and other arguments
   // see the TFile ctor. The preferred interface to this constructor is
   // via TFile::Open().

   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   fOffset = 0;
   fOption = option;
   fOption.ToUpper();
   fStatCached = kFALSE;

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
   TString stmp2;
   const char *fname;
   const char *fnameWithPrefix;

   if (!strncmp(path, DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      fnameWithPrefix = fname = path;
   } else {
      // Metadata provided by PNFS
      char *tname;
      if ((tname = gSystem->ExpandPathName(path))) {
         stmp = tname;
         stmp2 = DCACHE_PREFIX;
         stmp2 += tname;
         delete [] tname;
         fname = stmp;
         fnameWithPrefix = stmp2;
      } else {
         Error("TDCacheFile", "error expanding path %s", path);
         goto zombie;
      }
   }

   if (recreate) {
      if (!gSystem->AccessPathName(fnameWithPrefix, kFileExists))
         dc_unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && !gSystem->AccessPathName(fnameWithPrefix, kFileExists)) {
      Error("TDCacheFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (gSystem->AccessPathName(fnameWithPrefix, kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && gSystem->AccessPathName(fnameWithPrefix, kWritePermission)) {
         Error("TDCacheFile", "no write permission, could not open file %s", fname);
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
         if (gSystem->AccessPathName(fnameWithPrefix, kFileExists)) {
            Error("TDCacheFile", "file %s does not exist", fname);
            goto zombie;
         }
         if (gSystem->AccessPathName(fnameWithPrefix, kReadPermission)) {
            Error("TDCacheFile", "no read permission, could not open file %s", fname);
            goto zombie;
         }
         SysError("TDCacheFile", "file %s can not be opened for reading", fname);
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

   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   return TFile::ReadBuffer(buf, len);
}

//______________________________________________________________________________
Bool_t TDCacheFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via dCache daemon.
   // Returns kTRUE in case of error.

   if (!IsOpen() || !fWritable) return kTRUE;

   Int_t st;
   if ((st = WriteBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
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

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TDCacheFile::CheckFile(const char *path, const char *location)
{
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

   return kFALSE;
}

//______________________________________________________________________________
void TDCacheFile::SetOpenTimeout(UInt_t n)
{
   // Set file open timeout.

   dc_setOpenTimeout(n);
}

//______________________________________________________________________________
void TDCacheFile::SetOnError(OnErrorAction a)
{
   // Set on error handler.

   dc_setOnError(a);
}

//______________________________________________________________________________
void TDCacheFile::SetReplyHostName(const char *host_name)
{
   // Set reply host name.

   dc_setReplyHostName((char*)host_name);
}

//______________________________________________________________________________
const char *TDCacheFile::GetDcapVersion()
{
   // Return dCache version string.

   return getDcapVersion();
}

//______________________________________________________________________________
Int_t TDCacheFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in POSIX open.

   dc_errno = 0;

   Int_t rc = dc_open(pathname, flags, (Int_t) mode);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysClose(Int_t fd)
{
   // Interface to system close. All arguments like in POSIX close.

   dc_errno = 0;

   Int_t rc = dc_close(fd);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   // Interface to system read. All arguments like in POSIX read.

   fOffset += len;

   dc_errno = 0;

   Int_t rc = dc_read(fd, buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   // Interface to system write. All arguments like in POSIX write.

   fOffset += len;

   dc_errno = 0;

   Int_t rc =  dc_write(fd, (char *)buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
Long64_t TDCacheFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   // Interface to system seek. All arguments like in POSIX lseek.

   if (whence == SEEK_SET && offset == fOffset) return offset;

   dc_errno = 0;

   Long64_t rc = dc_lseek64(fd, offset, whence);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   } else
      fOffset = rc;

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysSync(Int_t fd)
{
   // Interface to system sync. All arguments like in POSIX fsync.
   // dCache always keep it's files sync'ed, so there's no need to
   // sync() them manually.

   Int_t rc;
   dc_errno = 0;

   rc = dc_fsync(fd);
   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysStat(Int_t, Long_t *id, Long64_t *size,
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

   // If in read mode, uses the cached file status, if available, to avoid
   // costly dc_stat() call.

   struct stat64 & statbuf = fStatBuffer; // reference the cache

   if (fOption != "READ" || !fStatCached) {
      // We are not in read mode, or the file status information is not yet
      // in the cache. Update or read the status information with dc_stat().

      const char *path = GetName();
      TString pathString = GetDcapPath(path);
      path = pathString.Data();

      if (path && (dc_stat64(path, &statbuf) >= 0)) {
         fStatCached = kTRUE;
      }
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
   // Transform the input path into a path usuable by the dcap C library,
   // i.e either dcap://nodename.org/where/filename.root or
   // /pnfs/where/filename.root

   if (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN)) {
      path += DCACHE_PREFIX_LEN;
   }
   if (!strncmp(path, DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      path += DCAP_PREFIX_LEN;
   }
   TString pathString(path);
   if (!strncmp(path, "///", 3)) {
      path += 2;
      pathString = path;
   }
   if (!strncmp(path, "//", 2)) {
      pathString = DCAP_PREFIX + pathString;
   }
   return pathString;
}


//______________________________________________________________________________
TDCacheSystem::TDCacheSystem() : TSystem("-DCache", "DCache Helper System")
{
   // Create helper class that allows directory access via dCache.

   // name must start with '-' to bypass the TSystem singleton check
   SetName("DCache");

   fDirp = 0;
}

//______________________________________________________________________________
int TDCacheSystem::MakeDirectory(const char *path)
{
   // Create a directory.

   Int_t rc;
   dc_errno = 0;
   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   rc = dc_mkdir(path, 0755);
   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

//______________________________________________________________________________
void *TDCacheSystem::OpenDirectory(const char *path)
{
   // Open a directory.

   dc_errno = 0;
   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   fDirp = dc_opendir(path);
   if (fDirp == 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return fDirp;
}

//______________________________________________________________________________
void TDCacheSystem::FreeDirectory(void * dirp)
{
   // Close a directory.

   Int_t rc;
   dc_errno = 0;

   rc = dc_closedir((DIR *)dirp);
   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   fDirp = 0;
   return;
}

//______________________________________________________________________________
const char *TDCacheSystem::GetDirEntry(void * dirp)
{
   // Get a directory entry.

   struct dirent *ent;
   dc_errno = 0;

   ent = dc_readdir((DIR *)dirp);
   if (ent == 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return !ent ? 0 : ent->d_name;
}

//______________________________________________________________________________
Bool_t TDCacheSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the Unix access(2) function.
   // Attention, bizarre convention of return value!!

   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   return dc_access(path, mode);
}

//______________________________________________________________________________
int TDCacheSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   // Get info about a file. Info is returned in the form of a FileStat_t
   // structure (see TSystem.h).
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   struct stat64 sbuf;

   if (path && (dc_stat64(path, &sbuf) >= 0)) {

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
