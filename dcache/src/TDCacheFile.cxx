// @(#)root/dcache:$Name: v3-03-05 $:$Id: TDCacheFile.cxx,v 1.2 2002/03/25 16:43:16 rdm Exp $
// Author: Grzegorz Mazur   20/01/2002

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
#include <unistd.h>

#include <dcap.h>

//______________________________________________________________________________
static const char* const DCACHE_PREFIX = "dcache:";
static const size_t DCACHE_PREFIX_LEN = strlen(DCACHE_PREFIX);
static const char* const DCAP_PREFIX = "dcap:";
static const size_t DCAP_PREFIX_LEN = strlen(DCAP_PREFIX);

//______________________________________________________________________________
ClassImp(TDCacheFile)

//______________________________________________________________________________
TDCacheFile::TDCacheFile(const char *path, Option_t *option,
                         const char *ftitle, Int_t compress):
   TFile(path, "NET", ftitle, compress)
{
   // get rid of the optional URI
   if (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN))
      path += 7;

   fOption = option;
   fOffset = 0;

   Bool_t create = kFALSE;
   if (!fOption.CompareTo("NEW", TString::kIgnoreCase) ||
       !fOption.CompareTo("CREATE", TString::kIgnoreCase))
      create = kTRUE;
   Bool_t recreate = fOption.CompareTo("RECREATE", TString::kIgnoreCase)
      ? kFALSE : kTRUE;
   Bool_t update   = fOption.CompareTo("UPDATE", TString::kIgnoreCase)
      ? kFALSE : kTRUE;
   Bool_t read     = fOption.CompareTo("READ", TString::kIgnoreCase)
      ? kFALSE : kTRUE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   const char *fname;

   if (!strncmp(path, DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      // Ugh, no PNFS support
      if (create || recreate || update) {
         Error("TDCacheFile", "Without PNFS support only reading access is allowed.");
         goto zombie;
      }
      SetName(path);
      fname = GetName();
   } else {
      // Metadata provided by PNFS
      if ((fname = gSystem->ExpandPathName(path))) {
         SetName(fname);
         delete [] (char*)fname;
         fname = GetName();
      } else {
         Error("TDCacheFile", "error expanding path %s", path);
         goto zombie;
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
   }
   // Connect to file system stream
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
         fOffset = off + len;
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
         fOffset = off + len;
         return kFALSE;
      }
   }

   return TFile::WriteBuffer(buf, len);
}

//______________________________________________________________________________
Bool_t TDCacheFile::Stage(const char *path, UInt_t after, const char *location)
{
   // Stage() returns kTRUE on success and kFALSE on failure.

   if (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN))
      path += 7;

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

   if (!strncmp(path, DCACHE_PREFIX, DCACHE_PREFIX_LEN))
      path += 7;

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
   switch (whence) {
   case SEEK_SET:
      if (offset == fOffset)
         return offset;
      else
         fOffset = offset;
      break;
   case SEEK_CUR:
      fOffset += offset;
      break;
   case SEEK_END:
      fOffset = fEND + offset;
      break;
   }

   dc_errno = 0;

   Int_t rc = dc_lseek(fd, offset, whence);

   if (rc < 0) {
      if (dc_errno != 0)
         gSystem->SetErrorStr(dc_strerror(dc_errno));
      else
         gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysSync(Int_t fd)
{
   // dCache always keep it's files sync'ed, so there's no need to
   // sync() them manually.

   return 0;
}

//______________________________________________________________________________
Int_t TDCacheFile::SysStat(Int_t fd, Long_t *id, Long_t *size,
                           Long_t *flags, Long_t *modtime)
{
   // FIXME: dcap library doesn't (yet) provide any stat()
   // capabilities, relying on PNFS to provide all meta-level data. In
   // spite of this, it is possible to open a file which is not
   // accessible via PNFS. In such case we do some dirty tricks to
   // provide as much information as possible, but not everything can
   // really be done correctly.

   if (!strncmp(GetName(), DCAP_PREFIX, DCAP_PREFIX_LEN)) {
      // Ugh, no PNFS support.

      // Try to provide a unique id
      *id = ::Hash(GetName());

      // Funny way of checking the file size, isn't it?
      Seek_t offset = fOffset;
      *size = SysSeek(fd, 0, SEEK_END);
      SysSeek(fd, offset, SEEK_SET);

      *flags = 0;          // This one is easy, only ordinary files are allowed here
      *modtime = 0;        // FIXME: no way to get it right without dc_stat()
      return 0;
   } else {
      return TFile::SysStat(fd, id, size, flags, modtime);
   }
}

//______________________________________________________________________________
void TDCacheFile::ResetErrno() const
{
   // Method resetting the dc_errno and errno.

   dc_errno = 0;
   TSystem::ResetErrno();
}
