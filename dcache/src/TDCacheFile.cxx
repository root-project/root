// @(#)root/dcache:$Name:$:$Id:$
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
// Author: Grzegorz Mazur <mazur@mail.desy.de>                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDCacheFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

// Declarations copied from dcap.h file to avoid additional parameters
// in the configuration process.
extern "C" {

/* POSIX like IO*/
extern int      dc_open(const char *, int, ...);
extern int      dc_creat(const char *, mode_t);
extern int      dc_close(int);
extern int      dc_close2(int);
extern ssize_t  dc_read(int, void *, size_t);
extern ssize_t  dc_write(int, const void *, size_t);
extern off_t    dc_lseek(int, off_t, int);

/* pre-stage */
extern int      dc_stage(const char *, time_t, const char *);
extern int      dc_check(const char *, const char *);

/* user control */
#define onErrorRetry 1
#define onErrorFail  0
#define onErrorDefault -1

void dc_setOpenTimeout(time_t);
void dc_setOnError(int);

/* read ahead buffering */
void dc_noBuffering(int);
void dc_setBufferSize(int, ssize_t);

extern void dc_setReplyHostName( char *s);
extern char * getDcapVersion();

extern void dc_setDebugLevel(int);
extern void debug(int, const char *, ...);
extern void dc_error(const char *);

/* thread-safe errno hack */
#ifdef _REENTRANT
extern int *__dc_errno();
#define dc_errno (*(__dc_errno()))
#else
extern int dc_errno;
#endif /* _REENTRANT */

#ifdef DCAP_USE_SSL
extern void dc_enableSSL();
#endif /* DCAP_USE_SSL */

}


ClassImp(TDCacheFile)

//______________________________________________________________________________
TDCacheFile::TDCacheFile(const char *path, Option_t *option,
                         const char *ftitle, Int_t compress):
    TFile(path, "NET", ftitle, compress)
{
   // get rid of the optional URI
   if (!strncmp(path, "dcache:", 7))
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

//*-*--------------Connect to file system stream
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
    // Note: Stage() returns kTRUE on success and kFALSE on
    // failure. This is _not_ compatible with the convention taken by
    // the dcap library authors.

    dc_stage(path, after, location);

    return errno == EAGAIN ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TDCacheFile::CheckFile(const char *path, const char *location)
{
    // Note: Name of the method was changed to avoid collision with Check
    // macro #defined in ROOT
    // Note: CheckFile() returns kTRUE on success and kFALSE on
    // failure. This is _not_ compatible with the convention taken by
    // the dcap library authors.

    dc_check(path, location);

    return errno == EAGAIN ? kTRUE : kFALSE;
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
Int_t TDCacheFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   return dc_open(pathname, flags, (Int_t) mode);
}

//______________________________________________________________________________
Int_t TDCacheFile::SysClose(Int_t fd)
{
   return dc_close(fd);
}

//______________________________________________________________________________
Int_t TDCacheFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   fOffset += len;
   return dc_read(fd, buf, len);
}

//______________________________________________________________________________
Int_t TDCacheFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   fOffset += len;
   return dc_write(fd, (char *)buf, len);
}

//______________________________________________________________________________
Seek_t TDCacheFile::SysSeek(Int_t fd, Seek_t offset, Int_t whence)
{
   switch (whence) {
   case SEEK_SET:
      if (offset == fOffset)
         return 0;
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

   return dc_lseek(fd, offset, whence);
}

//______________________________________________________________________________
Int_t TDCacheFile::SysSync(Int_t fd)
{
   return 0;
}
