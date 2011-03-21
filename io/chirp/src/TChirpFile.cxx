// @(#)root/chirp:$Id$
// Authors: Dan Bradley, Michael Albrecht, Douglas Thain

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
// write its data via a Chirp server. The primary API for accessing     //
// Chirp is through the chirp_reli interface, which corresponds closely //
// to Unix.  Most operations return an integer where >=0 indicates      //
// success and <0 indicates failure, setting the global errno.          //
// This allows most TFile methods to be implemented with a single       //
// line or two of Chirp (for more on the Chirp filesystem.              //
//
// Note that this class overrides ReadBuffers so as to take advantage   //
// of the Chirp "bulk I/O" feature which does multiple remote ops       //
// in a single call.                                                    //
//                                                                      //
// Most users of Chirp will access a named remote server url:           //
//     chirp://host.somewhere.edu/path                                  //
//                                                                      //
// The special host CONDOR is used to indicate a connection to the      //
// Chirp I/O proxy service when running inside of Condor:               //
//     chirp://CONDOR/path                                              //
//                                                                      //
// This module recognizes the following environment variables:          //
//    CHIRP_DEBUG_FILE  - Send debugging output to this file.           //
//    CHIRP_DEBUG_FLAGS - Turn on select debugging flags (e.g. 'all')   //
//    CHIRP_AUTH        - Select a specific auth type (e.g. 'globus')   //
//    CHIRP_TIMEOUT     - Specify how long to attempt each op, in secs  //
//                                                                      //
// For more information about the Chirp fileystem and protocol:         //
//    http://www.cse.nd.edu/~ccl/software/chirp                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TChirpFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>

extern "C" {
#include "chirp_reli.h"
#include "auth_all.h"
#include "debug.h"
}

// If the path component of a url is a blank string,
// then convert it to the root directory of that server.
#define FIXPATH(x) ( x[0]==0 ? "/" : x )

static int chirp_root_timeout = 3600;

static void chirp_root_global_setup()
{
   static int did_setup = 0;
   if (did_setup) return;

   debug_config("chirp_root");

   const char *debug_file = getenv("CHIRP_DEBUG_FILE");
   if (debug_file) debug_config_file(debug_file);

   const char *debug_flags = getenv("CHIRP_DEBUG_FLAGS");
   if (debug_flags) debug_flags_set(debug_flags);

   const char *auth_flags = getenv("CHIRP_AUTH");
   if (auth_flags) {
      auth_register_byname(auth_flags);
   } else {
      auth_register_all();
   }

   const char *timeout_string = getenv("CHIRP_TIMEOUT");
   if (timeout_string) chirp_root_timeout = atoi(timeout_string);

   did_setup = 1;
}

ClassImp(TChirpFile)

//_____________________________________________________________________________
TChirpFile::TChirpFile(const char *path, Option_t * option, const char *ftitle, Int_t compress):TFile(path, "NET", ftitle, compress)
{
   chirp_root_global_setup();

   chirp_file_ptr = 0;

   fOption = option;
   fOption.ToUpper();

   if (fOption == "NEW")
      fOption = "CREATE";

   Bool_t create = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read = (fOption == "READ") ? kTRUE : kFALSE;

   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      fOption = "READ";
   }

   fRealName = path;

   if (create || update || recreate) {
      Int_t mode = O_RDWR | O_CREAT;
      if (recreate)
         mode |= O_TRUNC;

#ifndef WIN32
      fD = SysOpen(path, mode, 0644);
#else
      fD = SysOpen(path, mode | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TChirpFile", "file %s can not be created", path);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
#ifndef WIN32
      fD = SysOpen(path, O_RDONLY, 0644);
#else
      fD = SysOpen(path, O_RDONLY | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("TChirpFile", "file %s can not be opened for reading", path);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   Init(create || recreate);

   return;

zombie:
   MakeZombie();
   gDirectory = gROOT;
}

//_____________________________________________________________________________
TChirpFile::~TChirpFile()
{
   Close();
}

//_____________________________________________________________________________
Bool_t TChirpFile::ReadBuffers(char *buf, Long64_t * pos, Int_t * len, Int_t nbuf)
{
   struct chirp_bulkio bulkio[nbuf];
   int i;

   char *nextbuf = buf;

   for (i = 0; i < nbuf; i++) {
      bulkio[i].type = CHIRP_BULKIO_PREAD;
      bulkio[i].file = chirp_file_ptr;
      bulkio[i].offset = pos[i];
      bulkio[i].length = len[i];
      bulkio[i].buffer = nextbuf;
      nextbuf += len[i];
   }

   INT64_T result = chirp_reli_bulkio(bulkio, nbuf, time(0) + chirp_root_timeout);

   if (result >= 0) {
      return kFALSE;
   } else {
      return kTRUE;
   }
}

//_____________________________________________________________________________
Int_t TChirpFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
   TUrl url(pathname);
   chirp_file_ptr = chirp_reli_open(url.GetHost(), FIXPATH(url.GetFile()), flags, (Int_t) mode, time(0) + chirp_root_timeout);
   if (chirp_file_ptr) {
      return 1;
   } else {
      return -1;
   }
}

//_____________________________________________________________________________
Int_t TChirpFile::SysClose(Int_t)
{
   return chirp_reli_close(chirp_file_ptr, time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
Int_t TChirpFile::SysRead(Int_t, void *buf, Int_t len)
{
   Int_t rc = chirp_reli_pread(chirp_file_ptr, buf, len, fOffset, time(0) + chirp_root_timeout);
   if (rc > 0) fOffset += rc;
   return rc;
}

//_____________________________________________________________________________
Int_t TChirpFile::SysWrite(Int_t, const void *buf, Int_t len)
{
   Int_t rc = chirp_reli_pwrite(chirp_file_ptr, buf, len, fOffset, time(0) + chirp_root_timeout);
   if (rc > 0) fOffset += rc;
   return rc;
}

//_____________________________________________________________________________
Long64_t TChirpFile::SysSeek(Int_t, Long64_t offset, Int_t whence)
{
   if (whence == SEEK_SET) {
      fOffset = offset;
   } else if(whence == SEEK_CUR) {
      fOffset += offset;
   } else if(whence == SEEK_END) {
      struct chirp_stat info;

      Int_t rc = chirp_reli_fstat(chirp_file_ptr, &info, time(0) + chirp_root_timeout);
      if (rc < 0) {
         SysError("TChirpFile", "Unable to seek from end of file");
         return -1;
      }

      fOffset = info.cst_size + offset;

   } else {
      SysError("TChirpFile", "Unknown whence!");
      return -1;
   }

   return fOffset;
}

//_____________________________________________________________________________
Int_t TChirpFile::SysSync(Int_t fd)
{
   return chirp_reli_fsync(chirp_file_ptr, time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
Int_t TChirpFile::SysStat(Int_t, Long_t * id, Long64_t * size, Long_t * flags, Long_t * modtime)
{
   struct chirp_stat cst;

   int rc = chirp_reli_fstat(chirp_file_ptr, &cst, time(0) + chirp_root_timeout);

   if (rc < 0) return rc;

   *id =::Hash(fRealName);
   *size = cst.cst_size;
   *flags = cst.cst_mode;
   *modtime = cst.cst_mtime;

   return 0;
}

ClassImp(TChirpSystem)

//_____________________________________________________________________________
TChirpSystem::TChirpSystem():TSystem("-chirp", "Chirp Helper System")
{
   SetName("chirp");
   chirp_root_global_setup();
}

//_____________________________________________________________________________
TChirpSystem::~TChirpSystem()
{
}

//_____________________________________________________________________________
Int_t TChirpSystem::MakeDirectory(const char *path)
{
   TUrl url(path);
   return chirp_reli_mkdir(url.GetHost(), FIXPATH(url.GetFile()), 0777, time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
void *TChirpSystem::OpenDirectory(const char *path)
{
   TUrl url(path);
   return chirp_reli_opendir(url.GetHost(), FIXPATH(url.GetFile()), time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
void TChirpSystem::FreeDirectory(void *dirp)
{
   return chirp_reli_closedir((struct chirp_dir *) dirp);
}

//_____________________________________________________________________________
const char *TChirpSystem::GetDirEntry(void *dirp)
{
   struct chirp_dirent *d = chirp_reli_readdir((struct chirp_dir *) dirp);
   if (d) {
      return d->name;
   } else {
      return 0;
   }
}

//_____________________________________________________________________________
Int_t TChirpSystem::GetPathInfo(const char *path, FileStat_t & buf)
{
   TUrl url(path);
   struct chirp_stat info;
   Int_t rc = chirp_reli_stat(url.GetHost(), FIXPATH(url.GetFile()), &info, time(0) + chirp_root_timeout);
   if (rc >= 0) {
      buf.fDev = info.cst_dev;
      buf.fIno = info.cst_ino;
      buf.fMode = info.cst_mode;
      buf.fUid = info.cst_uid;
      buf.fGid = info.cst_gid;
      buf.fSize = info.cst_size;
      buf.fMtime = info.cst_mtime;
      buf.fIsLink = S_ISLNK(info.cst_mode);
      buf.fUrl = TString(path);
   }
   return rc;
}

//_____________________________________________________________________________
Bool_t TChirpSystem::AccessPathName(const char *path, EAccessMode mode)
{
   TUrl url(path);

   int cmode = F_OK;

   if (mode & kExecutePermission) cmode |= X_OK;
   if (mode & kWritePermission)   cmode |= W_OK;
   if (mode & kReadPermission)    cmode |= R_OK;

   if (chirp_reli_access(url.GetHost(), FIXPATH(url.GetFile()), cmode, time(0) + chirp_root_timeout) == 0) {
      return kFALSE;
   } else {
      return kTRUE;
   }
}

//_____________________________________________________________________________
Int_t TChirpSystem::Unlink(const char *path)
{
   TUrl url(path);
   Int_t rc = chirp_reli_unlink(url.GetHost(), FIXPATH(url.GetFile()), time(0) + chirp_root_timeout);
   if (rc < 0 && errno == EISDIR) {
      rc = chirp_reli_rmdir(url.GetHost(), FIXPATH(url.GetFile()), time(0) + chirp_root_timeout);
   }
   return rc;
}

//_____________________________________________________________________________
int TChirpSystem::Rename(const char *from, const char *to)
{
   TUrl fromurl(from);
   TUrl tourl(to);

   if (strcmp(fromurl.GetHost(), tourl.GetHost())) {
      errno = EXDEV;
      return -1;
   }

   return chirp_reli_rename(fromurl.GetHost(), FIXPATH(fromurl.GetFile()), FIXPATH(tourl.GetFile()), time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
int TChirpSystem::Link(const char *from, const char *to)
{
   TUrl fromurl(from);
   TUrl tourl(to);

   if (strcmp(fromurl.GetHost(), tourl.GetHost())) {
      errno = EXDEV;
      return -1;
   }

   return chirp_reli_link(fromurl.GetHost(), FIXPATH(fromurl.GetFile()), FIXPATH(tourl.GetFile()), time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
int TChirpSystem::Symlink(const char *from, const char *to)
{
   TUrl fromurl(from);
   TUrl tourl(to);

   if (strcmp(fromurl.GetHost(), tourl.GetHost())) {
      errno = EXDEV;
      return -1;
   }

   return chirp_reli_symlink(fromurl.GetHost(), FIXPATH(fromurl.GetFile()), FIXPATH(tourl.GetFile()), time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
int TChirpSystem::GetFsInfo(const char *path, Long_t * id, Long_t * bsize, Long_t * blocks, Long_t * bfree)
{
   TUrl url(path);

   struct chirp_statfs info;

   int rc = chirp_reli_statfs(url.GetHost(), FIXPATH(url.GetFile()), &info, time(0) + chirp_root_timeout);
   if (rc >= 0) {
      *id = info.f_type;
      *bsize = info.f_bsize;
      *blocks = info.f_blocks;
      *bfree = info.f_bfree;
   }
   return rc;
}

//_____________________________________________________________________________
int TChirpSystem::Chmod(const char *path, UInt_t mode)
{
   TUrl url(path);
   return chirp_reli_chmod(url.GetHost(), FIXPATH(url.GetFile()), mode, time(0) + chirp_root_timeout);
}

//_____________________________________________________________________________
int TChirpSystem::Utime(const char *path, Long_t modtime, Long_t actime)
{
   TUrl url(path);
   return chirp_reli_utime(url.GetHost(), FIXPATH(url.GetFile()), modtime, actime, time(0) + chirp_root_timeout);
}
