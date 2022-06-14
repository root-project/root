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

ClassImp(TAlienSystem);

////////////////////////////////////////////////////////////////////////////////
/// Create a new OS interface.

TAlienSystem::TAlienSystem(const char *name, const char *title) : TSystem(name, title)
{
   fWorkingDirectory[0] = '\0';
}

////////////////////////////////////////////////////////////////////////////////
/// Delete the OS interface.

TAlienSystem::~TAlienSystem()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the OS interface.

Bool_t TAlienSystem::Init()
{
  return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Make a directory. Returns 0 in case of success and
/// -1 if the directory could not be created (either already exists or
/// illegal path name).

int TAlienSystem::MakeDirectory(const char* dirname)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Open a directory. Returns 0 if directory does not exist.

void *TAlienSystem::OpenDirectory(const char* name)
{
   TUrl url(name);
   url.CleanRelativePath();
   if (strcmp(url.GetProtocol(),"alien")) {
     Info("OpenDirectory","Assuming an AliEn URL alien://%s",name);
     url.SetProtocol("alien",kTRUE);
   }
   return (void*) gapi_opendir(url.GetUrl());
}

////////////////////////////////////////////////////////////////////////////////
/// Free a directory.

void TAlienSystem::FreeDirectory(void* ptr)
{
   gapi_closedir( (GAPI_DIR*)ptr);
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a directory entry. Returns 0 if no more entries.

const char *TAlienSystem::GetDirEntry(void* ptr)
{
   struct dirent* retdir;
   retdir = gapi_readdir( (GAPI_DIR*) ptr);
   //   AbstractMethod("GetDirEntry");
   if (retdir)
     return retdir->d_name;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Change directory.
///   AbstractMethod("ChangeDirectory");
///   return kFALSE;

Bool_t TAlienSystem::ChangeDirectory(const char* dirname)
{
  TUrl url(dirname);
  url.CleanRelativePath();
  if (strcmp(url.GetProtocol(),"alien")) {
    Info("OpenDirectory","Assuming an AliEn URL alien://%s",dirname);
    url.SetProtocol("alien",kTRUE);
  }
  return gapi_chdir(url.GetUrl());
  //  return gGrid->Cd(url.GetFile(),kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Return working directory.

const char *TAlienSystem::WorkingDirectory()
{
  return gapi_getcwd(fWorkingDirectory,1024);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the user's home directory.

const char *TAlienSystem::HomeDirectory(const char*)
{
  if (!gGrid)
    return 0;

  if (strcmp(gGrid->GetGrid(),"alien")) {
    Error("TAlienSystem","You are not connected to AliEn");
    return 0;
  }
  return (gGrid->GetHomeDirectory());
}

////////////////////////////////////////////////////////////////////////////////
/// Make a file system directory. Returns 0 in case of success and
/// -1 if the directory could not be created (either already exists or
/// illegal path name).
/// If 'recursive' is true, makes parent directories as needed.

int TAlienSystem::mkdir(const char *name, Bool_t recursive)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Copy a file. If overwrite is true and file already exists the
/// file will be overwritten. Returns 0 when successful, -1 in case
/// of failure, -2 in case the file already exists and overwrite was false.

int TAlienSystem::CopyFile(const char *, const char *, Bool_t)
{
   AbstractMethod("CopyFile");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Rename a file.

int TAlienSystem::Rename(const char *oldname, const char *newname)
{
   return gapi_rename(oldname,newname);
   //  AbstractMethod("Rename");
   //   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a link from file1 to file2.

int TAlienSystem::Link(const char *, const char *)
{
   AbstractMethod("Link");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a symbolic link from file1 to file2.

int TAlienSystem::Symlink(const char *, const char *)
{
   AbstractMethod("Symlink");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlink, i.e. remove, a file.

int TAlienSystem::Unlink(const char * filename)
{
   return gapi_unlink(filename);
   //   AbstractMethod("Unlink");
   //   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file: id, size, flags, modification time.
/// Id      is (statbuf.st_dev << 24) + statbuf.st_ino
/// Size    is the file size
/// Flags   is file type: 0 is regular file, bit 0 set executable,
///                       bit 1 set directory, bit 2 set special file
///                       (socket, fifo, pipe, etc.)
/// Modtime is modification time.
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

int TAlienSystem::GetPathInfo(const char *path, Long_t *id, Long_t *size,
                         Long_t *flags, Long_t *modtime)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file: id, size, flags, modification time.
/// Id      is (statbuf.st_dev << 24) + statbuf.st_ino
/// Size    is the file size
/// Flags   is file type: 0 is regular file, bit 0 set executable,
///                       bit 1 set directory, bit 2 set special file
///                       (socket, fifo, pipe, etc.)
/// Modtime is modification time.
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

int TAlienSystem::GetPathInfo(const char *path, Long_t *id, Long64_t *size,
                         Long_t *flags, Long_t *modtime)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

int TAlienSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   //   AbstractMethod("GetPathInfo(const char*, FileStat_t&)");
  return AlienFilestat(path,buf);
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

int TAlienSystem::AlienFilestat(const char *fpath, FileStat_t &buf)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file system: fs type, block size, number of blocks,
/// number of free blocks.

int TAlienSystem::GetFsInfo(const char *, Long_t *, Long_t *, Long_t *, Long_t *)
{
   AbstractMethod("GetFsInfo");
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the file permission bits. Returns -1 in case or error, 0 otherwise.

int TAlienSystem::Chmod(const char *file, UInt_t mode)
{
   TUrl url(file);
   url.CleanRelativePath();
   if (strcmp(url.GetProtocol(),"alien")) {
     Info("AlienFilestat","Assuming an AliEn URL alien://%s",file);
     url.SetProtocol("alien",kTRUE);
   }
   return gapi_chmod(url.GetUrl(),mode);
 }

////////////////////////////////////////////////////////////////////////////////
/// Set the process file creation mode mask.

int TAlienSystem::Umask(Int_t)
{
   AbstractMethod("Umask");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the a files modification and access times. If actime = 0 it will be
/// set to the modtime. Returns 0 on success and -1 in case of error.

int TAlienSystem::Utime(const char *, Long_t, Long_t)
{
   AbstractMethod("Utime");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Find location of file in a search path.
/// Returns 0 in case file is not found.

const char *TAlienSystem::FindFile(const char *, TString&, EAccessMode)
{
   AbstractMethod("Which");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns FALSE if one can access a file using the specified access mode.
/// The file name must not contain any special shell characters line ~ or $,
/// in those cases first call ExpandPathName().
/// Attention, bizarre convention of return value!!

Bool_t TAlienSystem::AccessPathName(const char *path, EAccessMode mode)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns the user's id. If user = 0, returns current user's id.

Int_t TAlienSystem::GetUid(const char * /*user*/)
{
   AbstractMethod("GetUid");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the effective user id. The effective id corresponds to the
/// set id bit on the file being executed.

Int_t TAlienSystem::GetEffectiveUid()
{
   AbstractMethod("GetEffectiveUid");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the group's id. If group = 0, returns current user's group.

Int_t TAlienSystem::GetGid(const char * /*group*/)
{
   AbstractMethod("GetGid");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the effective group id. The effective group id corresponds
/// to the set id bit on the file being executed.

Int_t TAlienSystem::GetEffectiveGid()
{
   AbstractMethod("GetEffectiveGid");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns all user info in the UserGroup_t structure. The returned
/// structure must be deleted by the user. In case of error 0 is returned.

UserGroup_t *TAlienSystem::GetUserInfo(Int_t /*uid*/)
{
   AbstractMethod("GetUserInfo");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns all user info in the UserGroup_t structure. If user = 0, returns
/// current user's id info. The returned structure must be deleted by the
/// user. In case of error 0 is returned.

UserGroup_t *TAlienSystem::GetUserInfo(const char * /*user*/)
{
   AbstractMethod("GetUserInfo");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns all group info in the UserGroup_t structure. The only active
/// fields in the UserGroup_t structure for this call are:
///    fGid and fGroup
/// The returned structure must be deleted by the user. In case of
/// error 0 is returned.

UserGroup_t *TAlienSystem::GetGroupInfo(Int_t /*gid*/)
{
   AbstractMethod("GetGroupInfo");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns all group info in the UserGroup_t structure. The only active
/// fields in the UserGroup_t structure for this call are:
///    fGid and fGroup
/// If group = 0, returns current user's group. The returned structure
/// must be deleted by the user. In case of error 0 is returned.

UserGroup_t *TAlienSystem::GetGroupInfo(const char * /*group*/)
{
   AbstractMethod("GetGroupInfo");
   return 0;
}
