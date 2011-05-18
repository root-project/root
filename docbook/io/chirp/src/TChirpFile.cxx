// @(#)root/chirp:$Id$
// Author: Dan Bradley   17/12/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChirpFile                                                           //
//                                                                      //
// A TChirpFile is like a normal TFile except that it may read and      //
// write its data via a Chirp server (for more on the Chirp protocol    //
// see http://www.cs.wisc.edu/condor/chirp).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TChirpFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include "chirp_client.h"


static const char* const CHIRP_PREFIX = "chirp:";
static const size_t CHIRP_PREFIX_LEN = 6;


ClassImp(TChirpFile)

//______________________________________________________________________________
TChirpFile::TChirpFile(const char *path, Option_t *option,
                       const char *ftitle, Int_t compress):
   TFile(path, "NET", ftitle, compress)
{
   //Passing option "NET" to prevent base-class from doing any local access.

   chirp_client = 0;

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

   char const *path_part;
   const char *fname;

   if (OpenChirpClient(path, &path_part)) {
      SysError("TChirpFile", "chirp client for %s can not be opened", path);
      goto zombie;
   }

   fname = path_part;

   fRealName = fname;

   if (create || update || recreate) {
      Int_t mode = O_RDWR | O_CREAT;
      if (recreate) mode |= O_TRUNC;

#ifndef WIN32
      fD = SysOpen(fname, mode, 0644);
#else
      fD = SysOpen(fname, mode | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TChirpFile", "file %s can not be opened", fname);
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

   Init(create || recreate);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TChirpFile::~TChirpFile()
{
   // Close and cleanup Chirp file.

   Close();
   CloseChirpClient();
}

//______________________________________________________________________________
Bool_t TChirpFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from remote file via Chirp daemon.
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
Bool_t TChirpFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read specified byte range from remote file via Chirp daemon.
   // Returns kTRUE in case of error.

   SetOffset(pos);
   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   return TFile::ReadBuffer(buf, pos, len);
}

//______________________________________________________________________________
Bool_t TChirpFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via Chirp daemon.
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
Int_t TChirpFile::OpenChirpClient(char const *URL, char const **path_part)
{
   // Caller should delete [] path when finished.
   // URL format: chirp:machine.name:port/path
   // or:         chirp:path      (use default connection to Condor job manager)

   *path_part = 0;

   CloseChirpClient();

   chirp_client = chirp_client_connect_url(URL, path_part);

   if (!chirp_client) {
      gSystem->SetErrorStr(strerror(errno));
      return -1;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TChirpFile::CloseChirpClient()
{
   if (chirp_client) {
      struct chirp_client *c = chirp_client;
      chirp_client = 0;

      chirp_client_disconnect(c);
   }

   return 0;
}

//______________________________________________________________________________
Int_t TChirpFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   char open_flags[8];
   char *f = open_flags;

   if ((flags & O_WRONLY) || (flags & O_RDWR)) *(f++) = 'w';
   if ((flags & O_RDONLY) || (flags & O_RDWR) || !flags) *(f++) = 'r';
   if (flags & O_APPEND) *(f++) = 'a';
   if (flags & O_CREAT)  *(f++) = 'c';
   if (flags & O_TRUNC)  *(f++) = 't';
   if (flags & O_EXCL)   *(f++) = 'x';

   *f = '\0';

   Int_t rc = chirp_client_open(chirp_client, pathname, open_flags, (Int_t) mode);

   if (rc < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TChirpFile::SysClose(Int_t fd)
{
   Int_t rc = chirp_client_close(chirp_client,fd);

   if (rc < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TChirpFile::SysRead(Int_t fd, void *buf, Int_t len)
{
   Int_t rc = chirp_client_read(chirp_client, fd, buf, len);

   if (rc < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TChirpFile::SysWrite(Int_t fd, const void *buf, Int_t len)
{
   Int_t rc = chirp_client_write(chirp_client, fd, (char *)buf, len);

   if (rc < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Long64_t TChirpFile::SysSeek(Int_t fd, Long64_t offset, Int_t whence)
{
   Long64_t rc = chirp_client_lseek(chirp_client, fd, offset, whence);

   if (rc < 0)
      gSystem->SetErrorStr(strerror(errno));

   return rc;
}

//______________________________________________________________________________
Int_t TChirpFile::SysSync(Int_t fd)
{
   Int_t rc = chirp_client_fsync(chirp_client, fd);

   if (rc < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }

   return rc;
}

//______________________________________________________________________________
Int_t TChirpFile::SysStat(Int_t fd, Long_t *id, Long64_t *size,
                           Long_t *flags, Long_t *modtime)
{
   // FIXME: chirp library doesn't (yet) provide any stat() capabilities.

   *id = ::Hash(fRealName);

   Long64_t offset = SysSeek(fd, 0, SEEK_CUR);
   *size = SysSeek(fd, 0, SEEK_END);
   SysSeek(fd, offset, SEEK_SET);

   *flags = 0;
   *modtime = 0;
   return 0;
}

//______________________________________________________________________________
void TChirpFile::ResetErrno() const
{
   TSystem::ResetErrno();
}
