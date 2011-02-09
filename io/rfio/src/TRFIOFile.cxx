// @(#)root/rfio:$Id$
// Author: Fons Rademakers 20/01/99 + Giulia Taurelli 29/06/2006 + Andreas Peters 07/12/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// TRFIOFile                                                             //
//                                                                       //
// A TRFIOFile is like a normal TFile except that it reads and writes    //
// its data via a rfiod server (for more on the rfiod daemon see         //
// http://wwwinfo.cern.ch/pdp/serv/shift.html). TRFIOFile file names     //
// are in standard URL format with protocol "rfio". The following are    //
// valid TRFIOFile URL's:                                                //
//                                                                       //
//    rfio:/afs/cern.ch/user/r/rdm/galice.root                           //
//         where galice.root is a symlink of the type /shift/.../...     //
//    rfio:na49db1:/data1/raw.root                                       //
//    rfio:/castor/cern.ch/user/r/rdm/test.root                          //
//                                                                       //
// If Castor 2.1 is used the file names can be given also in the         //
// following ways:                                                       //
//                                                                       //
//  rfio://host:port/?path=FILEPATH                                      //
//  rfio://host/?path=FILEPATH                                           //
//  rfio:///castor?path=FILEPATH                                         //
//  rfio://stager_host:stager_port/?path=/castor/cern.ch/user/r/         //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//  rfio://stager_host/?path=/castor/cern.ch/user/r/                     //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//  rfio:///castor?path=/castor/cern.ch/user/r/                          //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//                                                                       //
// path is mandatory as parameter but all the other ones are optional.   //
//                                                                       //
// For the ultimate description of supported urls see:                   //
//    https://twiki.cern.ch/twiki/bin/view/FIOgroup/RfioRootTurl         //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "TRFIOFile.h"
#include "TROOT.h"
#include "TTimeStamp.h"
#include "TVirtualPerfStats.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#ifndef R__WIN32
#include <unistd.h>
#if defined(R__SUN) || defined(R__SGI) || defined(R__HPUX) ||         \
defined(R__AIX) || defined(R__LINUX) || defined(R__SOLARIS) ||        \
defined(R__ALPHA) || defined(R__HIUX) || defined(R__FBSD) ||          \
defined(R__MACOSX) || defined(R__HURD) || defined(R__OBSD)
#define HAS_DIRENT
#endif
#endif

#ifdef HAS_DIRENT
#include <dirent.h>
#endif

#include <rfio.h>
#include <rfio_api.h>
#include <serrno.h>


ClassImp(TRFIOFile)
ClassImp(TRFIOSystem)

//______________________________________________________________________________
TRFIOFile::TRFIOFile(const char *url, Option_t *option, const char *ftitle,
                     Int_t compress)
   : TFile(url, "NET", ftitle, compress)
{
   // Create a RFIO file object. A RFIO file is the same as a TFile
   // except that it is being accessed via a rfiod server. The url
   // argument must be of the form: rfio:/path/file.root (where file.root
   // is a symlink of type /shift/aaa/bbb/ccc) or rfio:server:/path/file.root.
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TRFIOFile
   // object. Use IsZombie() to see if the file is accessable.
   // For a description of the option and other arguments see the TFile ctor.
   // The preferred interface to this constructor is via TFile::Open().

   fOption = option;
   fOption.ToUpper();

   Int_t readopt = RFIO_READBUF;
   ::rfiosetopt(RFIO_READOPT, &readopt, 4);

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

   // to be able to use the turl starting with castor:
   if (!strcmp(fUrl.GetProtocol(), "castor"))
      fUrl.SetProtocol("rfio");

   // old RFIO client does not ignore ?filetpye=raw, remove it
   TString opt = fUrl.GetOptions();
   if (opt.Contains("&filetype=raw")) {
      opt.ReplaceAll("&filetype=raw", "");
      fUrl.SetOptions(opt);
   } else if (opt.Contains("filetype=raw")) {
      opt.ReplaceAll("filetype=raw", "");
      fUrl.SetOptions(opt);
   }

   // old RFIO client lib does not support :///, need to change to :////
   Bool_t addSlash = kFALSE;
   if ((strstr(url, ":/")   && !strstr(url, "://")) ||
       (strstr(url, ":///") && !strstr(url, ":////")))
      addSlash = kTRUE;

   // the complete turl in fname
   TString fname;
   if (!addSlash)
      fname.Form("%s://%s", fUrl.GetProtocol(), fUrl.GetFile());
   else
      fname.Form("%s:///%s", fUrl.GetProtocol(), fUrl.GetFile());
   if (strlen(fUrl.GetOptions()))
      fname += Form("?%s", fUrl.GetOptions());

   if (recreate) {
      if (::rfio_access((char*)fname.Data(), kFileExists) == 0)
         ::rfio_unlink((char*)fname.Data());
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && ::rfio_access((char*)fname.Data(), kFileExists) == 0) {
      Error("TRFIOFile", "file %s already exists", fname.Data());
      goto zombie;
   }
   if (update) {
      if (::rfio_access((char*)fname.Data(), kFileExists) != 0) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && ::rfio_access((char*)fname.Data(), kWritePermission) != 0) {
         Error("TRFIOFile", "no write permission, could not open file %s", fname.Data());
         goto zombie;
      }
   }

   // Connect to file system stream
   fRealName = fname;

   if (create || update) {
#ifndef WIN32
      fD = SysOpen(fname.Data(), O_RDWR | O_CREAT, 0644);
#else
      fD = SysOpen(fname.Data(), O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TRFIOFile", "file %s can not be opened", fname.Data());
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fD = SysOpen(fname.Data(), O_RDONLY, 0644);
#else
      fD = SysOpen(fname.Data(), O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TRFIOFile", "file %s can not be opened for reading", fname.Data());
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

//______________________________________________________________________________
TRFIOFile::~TRFIOFile()
{
   // RFIO file dtor. Close and flush directory structure.

   Close();
}

//______________________________________________________________________________
Bool_t TRFIOFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
   // Read a list of buffers given in pos[] and len[] and return it
   // in a single buffer. Returns kTRUE in case of error.

   static struct iovec64 *iov = 0;
   static Int_t iovsize = 128;
   Int_t n;

   if (IsZombie()) {
      Error("ReadBuffers", "cannot read because object is in 'zombie' state");
      return kTRUE;
   }

   if (!IsOpen()) {
      Error("ReadBuffers", "the remote file is not open");
      return kTRUE;
   }

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   // we maintain a static iove64 buffer to avoid malloc/free with every call
   if (!iov) {
      if (nbuf > iovsize)
         iovsize = nbuf;

      iov = (struct iovec64*)malloc(sizeof(struct iovec64) * iovsize);
      if (gDebug > 1)
         Info("TRFIOFile", "allocating iovec64 with size %d", iovsize);
      if (!iov) {
         Error("TRFIOFile", "error allocating preseek vector of size %ld",
               (Long_t)sizeof(struct iovec64) * iovsize);
         return kTRUE;
      }
   } else {
      if (nbuf > iovsize) {
         iovsize = nbuf;
         iov = (struct iovec64*) realloc(iov, sizeof(struct iovec64) * iovsize);
         if (gDebug > 1)
            Info("TRFIOFile", "re-allocating iovec64 with size %d", iovsize);
         if (!iov) {
            Error("TRFIOFile", "error reallocating preseek vector of size %ld",
                  (Long_t)sizeof(struct iovec64) * iovsize);
            return kTRUE;
         }
      }
   }

   for (n = 0; n < nbuf; n++) {
      if (gDebug>1)
         Info("TFIOFile", "adding chunk %d, %lld %d", n, pos[n], len[n]);
      iov[n].iov_base = pos[n] + fArchiveOffset;
      iov[n].iov_len  = len[n];
   }

   // prefetch the stuff
   if (rfio_preseek64(fD, iov, nbuf) < 0) {
      Error("TRFIOFile", "error doing rfio_preseek64");
      return kTRUE;
   }

   // read the chunks
   Int_t k = 0;

   for (n = 0; n < nbuf; n++) {
      if (rfio_lseek64(fD, iov[n].iov_base, SEEK_SET) < 0) {
         Error("TRFIOFile", "error doing rfio_lseek64");
         return kTRUE;
      }
      if (rfio_read(fD, buf+k, iov[n].iov_len) < 0) {
         Error("TRFIOFile", "error doing rfio_read");
         return kTRUE;
      }
      k += iov[n].iov_len;
   }

   fBytesRead += k;
#ifdef WIN32
   SetFileBytesRead(GetFileBytesRead() + k);
   SetFileReadCalls(GetFileReadCalls() + 1);
#else
   fgBytesRead += k;
   fgReadCalls++;
#endif

   if (gPerfStats)
      gPerfStats->FileReadEvent(this, k, start);

   return kFALSE;
}

//______________________________________________________________________________
Int_t TRFIOFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in POSIX open.
   Int_t ret = ::rfio_open64((char*)pathname, flags, (Int_t) mode);
   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());
   return ret;
}

//______________________________________________________________________________
Int_t TRFIOFile::SysClose(Int_t fd)
{
   // Interface to system close. All arguments like in POSIX close.

   Int_t ret = ::rfio_close(fd);
   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());
   return ret;
}

//______________________________________________________________________________
Int_t TRFIOFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   // Interface to system read. All arguments like in POSIX read.

   Int_t ret = ::rfio_read(fd, (char *)buf, len);
   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());
   return ret;
}

//______________________________________________________________________________
Int_t TRFIOFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   // Interface to system write. All arguments like in POSIX write.

   Int_t ret = ::rfio_write(fd, (char *)buf, len);
   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());
   return ret;
}

//______________________________________________________________________________
Long64_t TRFIOFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   // Interface to system lseek. All arguments like in POSIX lseek
   // except that the offset and return value are Long_t to be able to
   // handle 64 bit file systems.

   Long64_t ret = ::rfio_lseek64(fd, offset, whence);

   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());

   return ret;
}

//______________________________________________________________________________
Int_t TRFIOFile::SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags,
                         Long_t *modtime)
{
   // Interface to TSystem:GetPathInfo(). Generally implemented via
   // stat() or fstat().

   struct stat64 statbuf;

   if (::rfio_fstat64(fd, &statbuf) >= 0) {
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

   gSystem->SetErrorStr(::rfio_serror());
   return 1;
}

//______________________________________________________________________________
Int_t TRFIOFile::GetErrno() const
{
   // Method returning rfio_errno. For RFIO files must use this
   // function since we need to check rfio_errno then serrno and finally errno.

   if (rfio_errno)
      return rfio_errno;
   if (serrno)
      return serrno;
   return TSystem::GetErrno();
}

//______________________________________________________________________________
void TRFIOFile::ResetErrno() const
{
   // Method resetting the rfio_errno, serrno and errno.

   rfio_errno = 0;
   serrno = 0;
   TSystem::ResetErrno();
}


//______________________________________________________________________________
TRFIOSystem::TRFIOSystem() : TSystem("-rfio", "RFIO Helper System")
{
   // Create helper class that allows directory access via rfiod.
   // The name must start with '-' to bypass the TSystem singleton check.

   SetName("rfio");

   fDirp = 0;
}

//______________________________________________________________________________
Int_t TRFIOSystem::MakeDirectory(const char *dir)
{
   // Make a directory via rfiod.

   TUrl url(dir);
   Int_t ret = ::rfio_mkdir((char*)url.GetFileAndOptions(), 0755);
   if (ret < 0)
      gSystem->SetErrorStr(::rfio_serror());
   return ret;
}

//______________________________________________________________________________
void *TRFIOSystem::OpenDirectory(const char *dir)
{
   // Open a directory via rfiod. Returns an opaque pointer to a dir
   // structure. Returns 0 in case of error.

   if (fDirp) {
      Error("OpenDirectory", "invalid directory pointer (should never happen)");
      fDirp = 0;
   }

   TUrl url(dir);

   struct stat finfo;
   if (::rfio_stat((char*)url.GetFileAndOptions(), &finfo) < 0)
      return 0;

   if ((finfo.st_mode & S_IFMT) != S_IFDIR)
      return 0;

   fDirp = (void*) ::rfio_opendir((char*)url.GetFileAndOptions());

   if (!fDirp)
      gSystem->SetErrorStr(::rfio_serror());

   return fDirp;
}

//______________________________________________________________________________
void TRFIOSystem::FreeDirectory(void *dirp)
{
   // Free directory via rfiod.

   if (dirp != fDirp) {
      Error("FreeDirectory", "invalid directory pointer (should never happen)");
      return;
   }

   if (dirp)
      ::rfio_closedir((DIR*)dirp);

   fDirp = 0;
}

//______________________________________________________________________________
const char *TRFIOSystem::GetDirEntry(void *dirp)
{
   // Get directory entry via rfiod. Returns 0 in case no more entries.

   if (dirp != fDirp) {
      Error("GetDirEntry", "invalid directory pointer (should never happen)");
      return 0;
   }

   struct dirent *dp;

   if (dirp) {
      dp = (struct dirent *) ::rfio_readdir((DIR*)dirp);
      if (!dp)
         return 0;
      return dp->d_name;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TRFIOSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   // Get info about a file. Info is returned in the form of a FileStat_t
   // structure (see TSystem.h).
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   TUrl url(path);

   struct stat64 sbuf;
   if (path && ::rfio_stat64((char*)url.GetFileAndOptions(), &sbuf) >= 0) {

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

//______________________________________________________________________________
Bool_t TRFIOSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // Mode is the same as for the Unix access(2) function.
   // Attention, bizarre convention of return value!!

   TUrl url(path);
   if (::rfio_access((char*)url.GetFileAndOptions(), mode) == 0)
      return kFALSE;
   gSystem->SetErrorStr(::rfio_serror());
   return kTRUE;
}

//______________________________________________________________________________
Int_t TRFIOSystem::Unlink(const char *path)
{
   // Unlink, i.e. remove, a file or directory. Returns 0 when succesfull,
   // -1 in case of failure.

   TUrl url(path);

   struct stat finfo;
   if (rfio_stat((char*)url.GetFileAndOptions(), &finfo) < 0)
      return -1;

   if (R_ISDIR(finfo.st_mode))
      return rfio_rmdir((char*)url.GetFileAndOptions());
   else
      return rfio_unlink((char*)url.GetFileAndOptions());
}
