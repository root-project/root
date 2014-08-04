// @(#)root/base:$Id$
// Author: Andreas Peters   15/05/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienSystem                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <errno.h>

#include "Riostream.h"
#include "TAlienSystem.h"
#include "TError.h"
#include "TUrl.h"
#include "TGrid.h"
#include "gapi_dir_operations.h"
#include "gapi_file_operations.h"
#include "gapi_stat.h"

ClassImp(TAlienSystem)

//______________________________________________________________________________
TAlienSystem::TAlienSystem(const char *name, const char *title) : TSystem(name, title)
{
   // Create a new OS interface.

   fWorkingDirectory[0] = '\0';
}

//______________________________________________________________________________
TAlienSystem::~TAlienSystem()
{
   // Delete the OS interface.
}

//______________________________________________________________________________
Bool_t TAlienSystem::Init()
{
   // Initialize the OS interface.
  return kTRUE;
}

//______________________________________________________________________________
int TAlienSystem::MakeDirectory(const char* dirname)
{
   // Make a directory. Returns 0 in case of success and
   // -1 if the directory could not be created (either already exists or
   // illegal path name).

  if (!gGrid)
    return -1;

  if (strcmp(gGrid->GetGrid(),"alien")) {
    Error("TAlienSystem","You are not connected to AliEn");
    return -1;
  }

  TUrl url(dirname);
  url.CleanRelativePath();
  if (strcmp(url.GetProtocol(),"alien")) {
    Info("OpenDirectory","Assuming an AliEn URL alien://%s",dirname);
    url.SetProtocol("alien",kTRUE);
  }
  return gapi_mkdir(url.GetUrl(),0);
}

//______________________________________________________________________________
void *TAlienSystem::OpenDirectory(const char* name)
{
   // Open a directory. Returns 0 if directory does not exist.

   TUrl url(name);
   url.CleanRelativePath();
   if (strcmp(url.GetProtocol(),"alien")) {
     Info("OpenDirectory","Assuming an AliEn URL alien://%s",name);
     url.SetProtocol("alien",kTRUE);
   }
   return (void*) gapi_opendir(url.GetUrl());
}

//______________________________________________________________________________
void TAlienSystem::FreeDirectory(void* ptr)
{
   // Free a directory.

   gapi_closedir( (GAPI_DIR*)ptr);
   return;
}

//______________________________________________________________________________
const char *TAlienSystem::GetDirEntry(void* ptr)
{
   // Get a directory entry. Returns 0 if no more entries.
   struct dirent* retdir;
   retdir = gapi_readdir( (GAPI_DIR*) ptr);
   //   AbstractMethod("GetDirEntry");
   if (retdir)
     return retdir->d_name;
   return 0;
}

//______________________________________________________________________________
Bool_t TAlienSystem::ChangeDirectory(const char* dirname)
{
   // Change directory.
   //   AbstractMethod("ChangeDirectory");
   //   return kFALSE;

  TUrl url(dirname);
  url.CleanRelativePath();
  if (strcmp(url.GetProtocol(),"alien")) {
    Info("OpenDirectory","Assuming an AliEn URL alien://%s",dirname);
    url.SetProtocol("alien",kTRUE);
  }
  return gapi_chdir(url.GetUrl());
  //  return gGrid->Cd(url.GetFile(),kFALSE);
}

//______________________________________________________________________________
const char *TAlienSystem::WorkingDirectory()
{
   // Return working directory.
  return gapi_getcwd(fWorkingDirectory,1024);
}

//______________________________________________________________________________
const char *TAlienSystem::HomeDirectory(const char*)
{
   // Return the user's home directory.
  if (!gGrid)
    return 0;

  if (strcmp(gGrid->GetGrid(),"alien")) {
    Error("TAlienSystem","You are not connected to AliEn");
    return 0;
  }
  return (gGrid->GetHomeDirectory());
}

//______________________________________________________________________________
int TAlienSystem::mkdir(const char *name, Bool_t recursive)
{
   // Make a file system directory. Returns 0 in case of success and
   // -1 if the directory could not be created (either already exists or
   // illegal path name).
   // If 'recursive' is true, makes parent directories as needed.

   if (recursive) {
      TString dirname = DirName(name);
      if (dirname.Length()==0) {
         // well we should not have to make the root of the file system!
         // (and this avoid infinite recursions!)
         return -1;
      }
      if (AccessPathName(dirname, kFileExists)) {
         int res = mkdir(dirname, kTRUE);
         if (res) return res;
      }
      if (!AccessPathName(name, kFileExists)) {
         return -1;
      }
   }

   return MakeDirectory(name);
}

//______________________________________________________________________________
int TAlienSystem::CopyFile(const char *, const char *, Bool_t)
{
   // Copy a file. If overwrite is true and file already exists the
   // file will be overwritten. Returns 0 when successful, -1 in case
   // of failure, -2 in case the file already exists and overwrite was false.

   AbstractMethod("CopyFile");
   return -1;
}

//______________________________________________________________________________
int TAlienSystem::Rename(const char *oldname, const char *newname)
{
   // Rename a file.
   return gapi_rename(oldname,newname);
   //  AbstractMethod("Rename");
   //   return -1;
}

//______________________________________________________________________________
int TAlienSystem::Link(const char *, const char *)
{
   // Create a link from file1 to file2.

   AbstractMethod("Link");
   return -1;
}

//______________________________________________________________________________
int TAlienSystem::Symlink(const char *, const char *)
{
   // Create a symbolic link from file1 to file2.

   AbstractMethod("Symlink");
   return -1;
}

//______________________________________________________________________________
int TAlienSystem::Unlink(const char * filename)
{
   // Unlink, i.e. remove, a file.

   return gapi_unlink(filename);
   //   AbstractMethod("Unlink");
   //   return -1;
}

//______________________________________________________________________________
int TAlienSystem::GetPathInfo(const char *path, Long_t *id, Long_t *size,
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

   Long64_t lsize;

   int res = GetPathInfo(path, id, &lsize, flags, modtime);

   if (res == 0 && size) {
      if (sizeof(Long_t) == 4 && lsize > kMaxInt) {
         Error("GetPathInfo", "file %s > 2 GB, use GetPathInfo() with Long64_t size", path);
         *size = kMaxInt;
      } else {
         *size = (Long_t)lsize;
      }
   }

   return res;
}

//______________________________________________________________________________
int TAlienSystem::GetPathInfo(const char *path, Long_t *id, Long64_t *size,
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

   FileStat_t buf;

   int res = GetPathInfo(path, buf);

   if (res == 0) {
      if (id)
         *id = (buf.fDev << 24) + buf.fIno;
      if (size)
         *size = buf.fSize;
      if (modtime)
         *modtime = buf.fMtime;
      if (flags) {
         *flags = 0;
         if (buf.fMode & (kS_IXUSR|kS_IXGRP|kS_IXOTH))
            *flags |= 1;
         if (R_ISDIR(buf.fMode))
            *flags |= 2;
         if (!R_ISREG(buf.fMode) && !R_ISDIR(buf.fMode))
            *flags |= 4;
      }
   }

   return res;
}

//______________________________________________________________________________
int TAlienSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   // Get info about a file. Info is returned in the form of a FileStat_t
   // structure (see TSystem.h).
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   //   AbstractMethod("GetPathInfo(const char*, FileStat_t&)");
  return AlienFilestat(path,buf);
}

//______________________________________________________________________________
int TAlienSystem::AlienFilestat(const char *fpath, FileStat_t &buf)
{
   // Get info about a file. Info is returned in the form of a FileStat_t
   // structure (see TSystem.h).
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.

   TUrl url(fpath);
   url.CleanRelativePath();
   if (strcmp(url.GetProtocol(),"alien")) {
     Info("AlienFilestat","Assuming an AliEn URL alien://%s",fpath);
     url.SetProtocol("alien",kTRUE);
   }
#if defined(R__SEEK64)
   struct stat64 sbuf;
   if ((gapi_lstat(url.GetUrl(), (GAPI_STAT*)(&sbuf))) == 0) {
#else
   struct stat sbuf;
   if ((gapi_lstat(url.GetUrl(), (GAPI_STAT*)(&sbuf))) == 0) {
#endif
      buf.fIsLink = S_ISLNK(sbuf.st_mode);
      buf.fDev   = sbuf.st_dev;
      buf.fIno   = sbuf.st_ino;
      buf.fMode  = sbuf.st_mode;
      buf.fUid   = sbuf.st_uid;
      buf.fGid   = sbuf.st_gid;
      buf.fSize  = sbuf.st_size;
      buf.fMtime = sbuf.st_mtime;

      return 0;
   }
   return 1;
}

//______________________________________________________________________________
int TAlienSystem::GetFsInfo(const char *, Long_t *, Long_t *, Long_t *, Long_t *)
{
   // Get info about a file system: fs type, block size, number of blocks,
   // number of free blocks.

   AbstractMethod("GetFsInfo");
   return 1;
}

//______________________________________________________________________________
int TAlienSystem::Chmod(const char *file, UInt_t mode)
{
   // Set the file permission bits. Returns -1 in case or error, 0 otherwise.
   TUrl url(file);
   url.CleanRelativePath();
   if (strcmp(url.GetProtocol(),"alien")) {
     Info("AlienFilestat","Assuming an AliEn URL alien://%s",file);
     url.SetProtocol("alien",kTRUE);
   }
   return gapi_chmod(url.GetUrl(),mode);
 }

//______________________________________________________________________________
int TAlienSystem::Umask(Int_t)
{
   // Set the process file creation mode mask.

   AbstractMethod("Umask");
   return -1;
}

//______________________________________________________________________________
int TAlienSystem::Utime(const char *, Long_t, Long_t)
{
   // Set the a files modification and access times. If actime = 0 it will be
   // set to the modtime. Returns 0 on success and -1 in case of error.

   AbstractMethod("Utime");
   return -1;
}

//______________________________________________________________________________
const char *TAlienSystem::FindFile(const char *, TString&, EAccessMode)
{
   // Find location of file in a search path.
   // Returns 0 in case file is not found.

   AbstractMethod("Which");
   return 0;
}

//______________________________________________________________________________
Bool_t TAlienSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // The file name must not contain any special shell characters line ~ or $,
   // in those cases first call ExpandPathName().
   // Attention, bizarre convention of return value!!

   if (!gGrid)
      return -1;

   if (strcmp(gGrid->GetGrid(),"alien")) {
      Error("TAlienSystem","You are not connected to AliEn");
      return -1;
   }

   TString strippath = path ;
   // remove trailing '/'
   while (strippath.EndsWith("/")) {strippath.Remove(strippath.Length()-1);}
   TUrl url(strippath);
   url.CleanRelativePath();

   if (strcmp(url.GetProtocol(),"alien")) {
      Info("AccessPathName","Assuming an AliEn URL alien://%s",path);
      url.SetProtocol("alien",kTRUE);
   }
   if(!gapi_access(url.GetUrl(),mode)) {
      return kFALSE;
   } else {
      return kTRUE;
   }
}


//---- Users & Groups ----------------------------------------------------------

//______________________________________________________________________________
Int_t TAlienSystem::GetUid(const char * /*user*/)
{
   // Returns the user's id. If user = 0, returns current user's id.

   AbstractMethod("GetUid");
   return 0;
}

//______________________________________________________________________________
Int_t TAlienSystem::GetEffectiveUid()
{
   // Returns the effective user id. The effective id corresponds to the
   // set id bit on the file being executed.

   AbstractMethod("GetEffectiveUid");
   return 0;
}

//______________________________________________________________________________
Int_t TAlienSystem::GetGid(const char * /*group*/)
{
   // Returns the group's id. If group = 0, returns current user's group.

   AbstractMethod("GetGid");
   return 0;
}

//______________________________________________________________________________
Int_t TAlienSystem::GetEffectiveGid()
{
   // Returns the effective group id. The effective group id corresponds
   // to the set id bit on the file being executed.

   AbstractMethod("GetEffectiveGid");
   return 0;
}

//______________________________________________________________________________
UserGroup_t *TAlienSystem::GetUserInfo(Int_t /*uid*/)
{
   // Returns all user info in the UserGroup_t structure. The returned
   // structure must be deleted by the user. In case of error 0 is returned.

   AbstractMethod("GetUserInfo");
   return 0;
}

//______________________________________________________________________________
UserGroup_t *TAlienSystem::GetUserInfo(const char * /*user*/)
{
   // Returns all user info in the UserGroup_t structure. If user = 0, returns
   // current user's id info. The returned structure must be deleted by the
   // user. In case of error 0 is returned.

   AbstractMethod("GetUserInfo");
   return 0;
}

//______________________________________________________________________________
UserGroup_t *TAlienSystem::GetGroupInfo(Int_t /*gid*/)
{
   // Returns all group info in the UserGroup_t structure. The only active
   // fields in the UserGroup_t structure for this call are:
   //    fGid and fGroup
   // The returned structure must be deleted by the user. In case of
   // error 0 is returned.

   AbstractMethod("GetGroupInfo");
   return 0;
}

//______________________________________________________________________________
UserGroup_t *TAlienSystem::GetGroupInfo(const char * /*group*/)
{
   // Returns all group info in the UserGroup_t structure. The only active
   // fields in the UserGroup_t structure for this call are:
   //    fGid and fGroup
   // If group = 0, returns current user's group. The returned structure
   // must be deleted by the user. In case of error 0 is returned.

   AbstractMethod("GetGroupInfo");
   return 0;
}
