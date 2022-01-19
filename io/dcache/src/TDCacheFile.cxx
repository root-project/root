// @(#)root/dcache:$Id$
// Author: Grzegorz Mazur   20/01/2002
// Modified: William Tanenbaum 01/12/2003
// Modified: Tigran Mkrtchyan 29/06/2004
// Modified: Tigran Mkrtchyan 06/07/2007

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TDCacheFile
\ingroup IO
A TDCacheFile is like a normal TFile except that it may read and
write its data via a dCache server (for more on the dCache daemon
see http://www-dcache.desy.de/. Given a path which doesn't belong
to the dCache managed filesystem, it falls back to the ordinary
TFile behaviour.
*/

#include "TDCacheFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include <cstdlib>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <dcap.h>
#ifndef R__WIN32
#include <unistd.h>
#if defined(R__SUN) || defined(R__HPUX) || \
    defined(R__AIX) || defined(R__LINUX) || defined(R__SOLARIS) || \
    defined(R__HIUX) || defined(R__FBSD) || defined(R__MACOSX) || \
    defined(R__HURD) || defined(R__OBSD)
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


ClassImp(TDCacheFile);

////////////////////////////////////////////////////////////////////////////////
/// Create a dCache file object.
///
/// A dCache file is the same as a TFile
/// except that it is being accessed via a dCache server. The url
/// argument must be of the form: `dcache:/pnfs/<path>/<file>.root` or
/// `dcap://<nodename.org>/<path>/<file>.root`. If the file specified in the
/// URL does not exist, is not accessable or can not be created the kZombie
/// bit will be set in the TDCacheFile object. Use IsZombie() to see if the
/// file is accessable. For a description of the option and other arguments
/// see the TFile ctor. The preferred interface to this constructor is
/// via TFile::Open().

TDCacheFile::TDCacheFile(const char *path, Option_t *option,
                         const char *ftitle, Int_t compress):
   TFile(path, "NET", ftitle, compress)
{
   TString pathString = GetDcapPath(path);
   path = pathString.Data();

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

   // use 128K ( default ) read-ahead buffer to get file header,
   // the buffer size can be overriden by env var "DCACHE_RA_BUFFER",
   // vector read are not affected by read-ahead buffer
   if (read) {
     int dcache_RAHEAD_SIZE = RAHEAD_BUFFER_SIZE;
     const char *DCACHE_RA_BUFFER = gSystem->Getenv("DCACHE_RA_BUFFER");
     if (DCACHE_RA_BUFFER) {
        int ra_buffer = atoi(DCACHE_RA_BUFFER);
        dcache_RAHEAD_SIZE = ra_buffer<=0 ? dcache_RAHEAD_SIZE : ra_buffer;
     }
     dc_setBufferSize(fD, dcache_RAHEAD_SIZE);
   } else {
     dc_noBuffering(fD);
   }

   Init(create);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

////////////////////////////////////////////////////////////////////////////////
/// Close and cleanup dCache file.

TDCacheFile::~TDCacheFile()
{
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via dCache daemon.
/// Returns kTRUE in case of error.

Bool_t TDCacheFile::ReadBuffer(char *buf, Int_t len)
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
/// Read specified byte range from remote file via dCache daemon.
/// Returns kTRUE in case of error.

Bool_t TDCacheFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
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
/// Read the nbuf blocks described in arrays pos and len,
/// where pos[i] is the seek position of block i of length len[i].
/// Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer.
/// This function is overloaded by TNetFile, TWebFile, etc.
/// Returns kTRUE in case of failure.

Bool_t TDCacheFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
#ifdef _IOVEC2_

   iovec2 *vector;

   vector = (iovec2 *)malloc(sizeof(iovec2)*nbuf);

   Int_t total_len = 0;
   for (Int_t i = 0; i < nbuf; i++) {
      vector[i].buf    = &buf[total_len];
      vector[i].offset = pos[i] + fArchiveOffset;
      vector[i].len    = len[i];
      total_len       += len[i];
   }

   Int_t rc = dc_readv2(fD, vector, nbuf);
   free(vector);

   if (rc == 0) {
      fBytesRead += total_len;
      SetFileBytesRead(GetFileBytesRead() + total_len);
      return kFALSE;
   }

#endif

   // if we failed to get with dc_readv2 (old server), try to loop over

   Int_t k = 0;
   Bool_t result = kTRUE;
   TFileCacheRead *old = fCacheRead;
   fCacheRead = 0;

   Long64_t low  = pos[0];
   Long64_t high = pos[nbuf-1] + len[nbuf-1] - pos[0];

   Long64_t total = 0;
   for(Int_t j=0; j < nbuf; j++) {
      total += len[j];
   }

   if ( total && high / total < 10 ) {

      char *temp = new char[high];
      Seek(low);
      result = ReadBuffer(temp,high);

      if (result==0) {
         for (Int_t i = 0; i < nbuf; i++) {
            memcpy(&buf[k], &(temp[pos[i]-pos[0]]), len[i]);
            k += len[i];
         }
      }

      delete [] temp;

   } else {

      for (Int_t i = 0; i < nbuf; i++) {
         Seek(pos[i]);
         result = ReadBuffer(&buf[k], len[i]);
         if (result) break;
         k += len[i];
      }

   }

   fCacheRead = old;
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Write specified byte range to remote file via dCache daemon.
/// Returns kTRUE in case of error.

Bool_t TDCacheFile::WriteBuffer(const char *buf, Int_t len)
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

////////////////////////////////////////////////////////////////////////////////
/// Stage() returns kTRUE on success and kFALSE on failure.

Bool_t TDCacheFile::Stage(const char *path, UInt_t after, const char *location)
{
   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   dc_errno = 0;

   if (dc_stage(path, after, location) == 0)
      return kTRUE;

   if (dc_errno != 0)
      gSystem->SetErrorStr(dc_strerror(dc_errno));

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// CheckFile() returns kTRUE on success and kFALSE on failure.  In
/// case the file exists but is not cached, CheckFile() returns
/// kFALSE and errno is set to EAGAIN.

Bool_t TDCacheFile::CheckFile(const char *path, const char *location)
{
   TString pathString = GetDcapPath(path);
   path = pathString.Data();

   dc_errno = 0;

   if (dc_check(path, location) == 0)
      return kTRUE;

   if (dc_errno != 0)
      gSystem->SetErrorStr(dc_strerror(dc_errno));

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set file open timeout.

void TDCacheFile::SetOpenTimeout(UInt_t n)
{
   dc_setOpenTimeout(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Set on error handler.

void TDCacheFile::SetOnError(EOnErrorAction a)
{
   dc_setOnError(a);
}

////////////////////////////////////////////////////////////////////////////////
/// Set reply host name.

void TDCacheFile::SetReplyHostName(const char *host_name)
{
   dc_setReplyHostName((char*)host_name);
}

////////////////////////////////////////////////////////////////////////////////
/// Return dCache version string.

const char *TDCacheFile::GetDcapVersion()
{
   return getDcapVersion();
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system open. All arguments like in POSIX open.

Int_t TDCacheFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // often there is a filewall on front of storage system.
   // let clients connect to the data servers
   // if it's an old dCache version, pool will try to connect to the client
   // (if it's fine with firewall)

   dc_setClientActive();

   dc_errno = 0;

   Int_t rc = dc_open(pathname, flags, (Int_t) mode);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system close. All arguments like in POSIX close.

Int_t TDCacheFile::SysClose(Int_t fd)
{
   dc_errno = 0;

   Int_t rc = dc_close(fd);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system read. All arguments like in POSIX read.

Int_t TDCacheFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   dc_errno = 0;

   Int_t rc = dc_read(fd, buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system write. All arguments like in POSIX write.

Int_t TDCacheFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   dc_errno = 0;

   Int_t rc =  dc_write(fd, (char *)buf, len);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system seek. All arguments like in POSIX lseek.

Long64_t TDCacheFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   dc_errno = 0;

   Long64_t rc = dc_lseek64(fd, offset, whence);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to system sync. All arguments like in POSIX fsync.
/// dCache always keep it's files sync'ed, so there's no need to
/// sync() them manually.

Int_t TDCacheFile::SysSync(Int_t fd)
{
   Int_t rc;
   dc_errno = 0;

   rc = dc_fsync(fd);
   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file: id, size, flags, modification time.
///
/// \param[in] fd ignored
/// \param[in] id (statbuf.st_dev << 24) + statbuf.st_ino
/// \param[in] size The file size
/// \param[in] flags File type: 0 is regular file, bit 0 set executable, bit 1 set directory, bit 2 set special file (socket, fifo, pipe, etc.)
/// \param[in] modtime Modification time.
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

Int_t TDCacheFile::SysStat(Int_t /*fd*/, Long_t *id, Long64_t *size,
                           Long_t *flags, Long_t *modtime)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Method resetting the dc_errno and errno.

void TDCacheFile::ResetErrno() const
{
   dc_errno = 0;
   TSystem::ResetErrno();
}

////////////////////////////////////////////////////////////////////////////////
/// Transform the input path into a path usuable by the dcap C library,
/// i.e either \a dcap://nodename.org/where/filename.root or
/// \a /pnfs/where/filename.root

TString TDCacheFile::GetDcapPath(const char *path)
{
   // eat all 'dcache:' prefixes
   while (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN)) {
      path += DCACHE_PREFIX_LEN;
   }

   TUrl url(path);
   TString pathString(url.GetUrl());

   // convert file://path url and dcap:///path to /path
   if(!strncmp(url.GetProtocol(), "file", 4) || !strcmp(url.GetHost(),"")){
       pathString = url.GetFile();
   }

   return pathString;
}


////////////////////////////////////////////////////////////////////////////////
/// Create helper class that allows directory access via dCache.

TDCacheSystem::TDCacheSystem() : TSystem("-DCache", "DCache Helper System")
{
   // name must start with '-' to bypass the TSystem singleton check
   SetName("DCache");

   fDirp = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a directory.

int TDCacheSystem::MakeDirectory(const char *path)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Open a directory.

void *TDCacheSystem::OpenDirectory(const char *path)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Close a directory.

void TDCacheSystem::FreeDirectory(void * dirp)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get a directory entry.

const char *TDCacheSystem::GetDirEntry(void * dirp)
{
   struct dirent *ent;
   dc_errno = 0;

   ent = dc_readdir((DIR *)dirp);
   if (ent == 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
   }

   return !ent ? 0 : ent->d_name;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns FALSE if one can access a file using the specified access mode.
/// Mode is the same as for the Unix access(2) function.
/// Attention, bizarre convention of return value!!

Bool_t TDCacheSystem::AccessPathName(const char *path, EAccessMode mode)
{
   TString pathString = TDCacheFile::GetDcapPath(path);
   path = pathString.Data();

   return dc_access(path, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

int TDCacheSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
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
