// @(#)root/rfio:$Name:  $:$Id: TRFIOFile.cxx,v 1.5 2001/01/08 09:11:14 rdm Exp $
// Author: Fons Rademakers   20/01/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRFIOFile                                                            //
//                                                                      //
// A TRFIOFile is like a normal TFile except that it reads and writes   //
// its data via a rfiod server (for more on the rfiod daemon see        //
// http://wwwinfo.cern.ch/pdp/serv/shift.html). TRFIOFile file names    //
// are in standard URL format with protocol "rfio". The following are   //
// valid TRFIOFile URL's:                                               //
//                                                                      //
//    rfio:/afs/cern.ch/user/r/rdm/galice.root                          //
//         where galice.root is a symlink of the type /shift/.../...    //
//    rfio:na49db1:/data1/raw.root                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRFIOFile.h"
#include "TSystem.h"
#include "TROOT.h"
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
   int rfio_open(char *filepath, int flags, int mode);
   int rfio_close(int s);
   int rfio_read(int s, char *ptr, int size);
   int rfio_write(int s, char *ptr, int size);
   int rfio_lseek(int s, int offset, int how);
   int rfio_access(char *filepath, int mode);
   int rfio_unlink(char *filepath);
   int rfio_parse(char *name, char **host, char **path);
   int rfio_fstat(int s, struct stat *statbuf);
};


ClassImp(TRFIOFile)

//______________________________________________________________________________
TRFIOFile::TRFIOFile(const char *url, Option_t *option, const Text_t *ftitle, Int_t compress)
         : TFile(url, "NET", ftitle, compress), fUrl(url)
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

   char *fname;
   if ((fname = gSystem->ExpandPathName(fUrl.GetFile()))) {
      if (!strstr(fname, ":/")) {
         char *host;
         char *name;
         if (::rfio_parse(fname, &host, &name))
            SetName(Form("%s:%s", host, name));
         else
            SetName(fname);
      } else
         SetName(fname);
      delete [] fname;
      fname = (char *)GetName();
   } else {
      Error("TRFIOFile", "error expanding path %s", fUrl.GetFile());
      goto zombie;
   }

   if (recreate) {
      if (::rfio_access(fname, kFileExists) == 0)
         ::rfio_unlink(fname);
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }
   if (create && ::rfio_access(fname, kFileExists) == 0) {
      Error("TRFIOFile", "file %s already exists", fname);
      goto zombie;
   }
   if (update) {
      if (::rfio_access(fname, kFileExists) != 0) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update && ::rfio_access(fname, kWritePermission) != 0) {
         Error("TRFIOFile", "no write permission, could not open file %s", fname);
         goto zombie;
      }
   }
   if (read) {
      if (::rfio_access(fname, kFileExists) != 0) {
         Error("TRFIOFile", "file %s does not exist", fname);
         goto zombie;
      }
      if (::rfio_access(fname, kReadPermission) != 0) {
         Error("TRFIOFile", "no read permission, could not open file %s", fname);
         goto zombie;
      }
   }

   if (create || update) {
#ifndef WIN32
      fD = SysOpen(fname, O_RDWR | O_CREAT, 0644);
#else
      fD = SysOpen(fname, O_RDWR | O_CREAT | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TRFIOFile", "file %s can not be opened", fname);
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
         SysError("TRFIOFile", "file %s can not be opened for reading", fname);
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
Int_t TRFIOFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   // Interface to system open. All arguments like in "man 2 open".

   return ::rfio_open((char *)pathname, flags, (Int_t) mode);
}

//______________________________________________________________________________
Int_t TRFIOFile::SysClose(Int_t fd)
{
   // Interface to system close. All arguments like in "man 2 close".

   return ::rfio_close(fd);
}

//______________________________________________________________________________
Int_t TRFIOFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   // Interface to system read. All arguments like in "man 2 read".

   fOffset += len;
   return ::rfio_read(fd, (char *)buf, len);
}

//______________________________________________________________________________
Int_t TRFIOFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   // Interface to system write. All arguments like in "man 2 write".

   fOffset += len;
   return ::rfio_write(fd, (char *)buf, len);
}

//______________________________________________________________________________
Seek_t TRFIOFile::SysSeek(Int_t fd, Seek_t offset, Int_t whence)
{
   // Interface to system lseek. All arguments like in "man 2 lseek"
   // except that the offset and return value are Long_t to be able to
   // handle 64 bit file systems.

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

   return ::rfio_lseek(fd, offset, whence);
}

//______________________________________________________________________________
Int_t TRFIOFile::SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags,
                         Long_t *modtime)
{
   // Interface to TSystem:GetPathInfo(). Generally implemented via
   // stat() or fstat().

   struct stat statbuf;

   if (::rfio_fstat(fd, &statbuf) >= 0) {
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

//______________________________________________________________________________
Bool_t TRFIOFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from remote file via rfiod daemon.
   // Returns kTRUE in case of error.

   if (fCache) {
      Int_t st;
      if ((st = fCache->ReadBuffer(fOffset, buf, len)) < 0) {
         Error("ReadBuffer", "error reading from cache");
         return kTRUE;
      }
      if (st > 0)
         return kFALSE;
   }

   return TFile::ReadBuffer(buf, len);
}

//______________________________________________________________________________
Bool_t TRFIOFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (!IsOpen() || !fWritable) return kTRUE;

   if (fCache) {
      Int_t st;
      if ((st = fCache->WriteBuffer(fOffset, buf, len)) < 0) {
         Error("WriteBuffer", "error writing to cache");
         return kTRUE;
      }
      if (st > 0)
         return kFALSE;
   }

   return TFile::WriteBuffer(buf, len);
}
