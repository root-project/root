// @(#)root/hdfs:$Id$
// Author: Brian Bockelman 29/09/2009

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class THDFSFile
\ingroup IO

Reads and writes its data via the HDFS protocols

A THDFSFile is like a normal TFile except that it reads and writes
its data via the HDFS protocols. For more information on HDFS, see
http://hadoop.apache.org/hdfs/.
This implementation interfaces with libhdfs, which is a JNI-based
library (i.e., it will start a Java JVM internally the first time
it is called). At a minimum, you will need your environment's
$CLASSPATH variable set up properly to use. Here's an example of
one way to properly set your classpath, assuming you use the OSG
distribution of Hadoop:
    $ source $HADOOP_CONF_DIR/hadoop-env.sh
    $ export CLASSPATH=$HADOOP_CLASSPATH
Additionally, you will need a valid libjvm in your $LD_LIBRARY_PATH
This is usually found in either:
    $JAVA_HOME/jre/lib/i386/server
or
    $JAVA_HOME/jre/lib/amd64/server
This file can only be used if hdfs support is compiled into ROOT.
The HDFS URLs should be of the form:
    hdfs:///path/to/file/in/HDFS.root
Any host or port information will be ignored; this is taken from the
node's HDFS configuration files.
*/

#include "syslog.h"
#include "assert.h"
#include "stdlib.h"

#include "THDFSFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include "hdfs.h"
//#include "hdfsJniHelper.h"

// For now, we don't allow any write/fs modification operations.
static const Bool_t R__HDFS_ALLOW_CHANGES = kFALSE;

// The following snippet is used for developer-level debugging
// Contributed by Pete Wyckoff of the HDFS project
#define THDFSFile_TRACE
#ifndef THDFSFile_TRACE
#define TRACE(x) \
  Debug("THDFSFile", "%s", x);
#else
#define TRACE(x);
#endif

ClassImp(THDFSFile)

////////////////////////////////////////////////////////////////////////////////
/// Usual Constructor.  See the TFile constructor for details.

THDFSFile::THDFSFile(const char *path, Option_t *option,
                     const char *ftitle, Int_t compress):
   TFile(path, "WEB", ftitle, compress)
{
   fHdfsFH    = 0;
   fFS        = 0;
   fSize      = -1;
   fPath      = 0;
   fSysOffset = 0;

   fOption = option;
   fOption.ToUpper();
   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   Bool_t has_authn = kTRUE;

   if (has_authn) {
      UserGroup_t *ugi = gSystem->GetUserInfo(0);
      const char *user = (ugi->fUser).Data();
      fFS = hdfsConnectAsUser("default", 0, user);
      delete ugi;
   } else {
      fFS = hdfsConnect("default", 0);
   }

   if (fFS == 0) {
      SysError("THDFSFile", "HDFS client for %s cannot open the filesystem",
               path);
      goto zombie;
   }

   if (create || update || recreate) {
      Int_t mode = O_RDWR | O_CREAT;
      if (recreate) mode |= O_TRUNC;

#ifndef WIN32
      fD = SysOpen(path, mode, 0644);
#else
      fD = SysOpen(path, mode | O_BINARY, S_IREAD | S_IWRITE);
#endif
      if (fD == -1) {
         SysError("THDFSFile", "file %s can not be opened", path);
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
         SysError("THDFSFile", "file %s can not be opened for reading", path);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   Init(create || recreate);

   return;

zombie:
   // Error in opening file; make this a zombie
   MakeZombie();
   gDirectory = gROOT;
}

////////////////////////////////////////////////////////////////////////////////
/// Close and clean-up HDFS file.

THDFSFile::~THDFSFile()
{
   TRACE("destroy")

   if (fPath)
      delete [] fPath;

   // We assume that the file is closed in SysClose
   // Explicitly release reference to HDFS filesystem object.
   // Turned off now due to compilation issues.
   // The very awkward way of releasing HDFS FS objects (by accessing JNI
   // internals) is going away in the next libhdfs version.
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified number of bytes from current offset into the buffer.
/// See documentation for TFile::SysRead().

Int_t THDFSFile::SysRead(Int_t, void *buf, Int_t len)
{
   TRACE("READ")
   tSize num_read = hdfsPread((hdfsFS)fFS, (hdfsFile)fHdfsFH, fSysOffset, buf, len);
   fSysOffset += len;
   if (num_read < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }
   return num_read;
}

////////////////////////////////////////////////////////////////////////////////
/// Seek to a specified position in the file.  See TFile::SysSeek().
/// Note that THDFSFile does not support seeks when the file is open for write.

Long64_t THDFSFile::SysSeek(Int_t, Long64_t offset, Int_t whence)
{
   TRACE("SEEK")
   if (whence == SEEK_SET)
      fSysOffset = offset;
   else if (whence == SEEK_CUR)
      fSysOffset += offset;
   else if (whence == SEEK_END) {
      if (offset > 0) {
         SysError("THDFSFile", "Unable to seek past end of file");
         return -1;
      }
      if (fSize == -1) {
         hdfsFileInfo *info = hdfsGetPathInfo((hdfsFS)fFS, fPath);
         if (info != 0) {
            fSize = info->mSize;
            free(info);
         } else {
            SysError("THDFSFile", "Unable to seek to end of file");
            return -1;
         }
      }
      fSysOffset = fSize;
   } else {
      SysError("THDFSFile", "Unknown whence!");
      return -1;
   }
   return fSysOffset;
}

////////////////////////////////////////////////////////////////////////////////
/// Open a file in HDFS.

Int_t THDFSFile::SysOpen(const char * pathname, Int_t flags, UInt_t)
{
   // This is given to us as a URL (hdfs://hadoop-name:9000//foo or
   // hdfs:///foo); convert this to a file name.
   TUrl url(pathname);
   const char * file = url.GetFile();
   size_t path_size = strlen(file);
   fPath = new char[path_size+1];
   if (fPath == 0) {
      SysError("THDFSFile", "Unable to allocate memory for path.");
   }
   strlcpy(fPath, file,path_size+1);
   if ((fHdfsFH = hdfsOpenFile((hdfsFS)fFS, fPath, flags, 0, 0, 0)) == 0) {
      SysError("THDFSFile", "Unable to open file %s in HDFS", pathname);
      return -1;
   }
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Close the file in HDFS.

Int_t THDFSFile::SysClose(Int_t)
{
   int result = hdfsCloseFile((hdfsFS)fFS, (hdfsFile)fHdfsFH);
   fFS = 0;
   fHdfsFH = 0;
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Write a buffer into the file; this is not supported currently.

Int_t THDFSFile::SysWrite(Int_t, const void *, Int_t)
{
   errno = ENOSYS;
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Perform a stat on the HDFS file; see TFile::SysStat().

Int_t THDFSFile::SysStat(Int_t, Long_t* id, Long64_t* size, Long_t* flags, Long_t* modtime)
{
   *id = ::Hash(fPath);

   hdfsFileInfo *info = hdfsGetPathInfo((hdfsFS)fFS, fPath);
   if (info != 0) {
      fSize = info->mSize;
      *size = fSize;
      if (info->mKind == kObjectKindFile)
         *flags = 0;
      else if (info->mKind == kObjectKindDirectory)
         *flags = 1;
      *modtime = info->mLastMod;
      free(info);
   } else {
      return 1;
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sync remaining data to disk; Not supported by HDFS.

Int_t THDFSFile::SysSync(Int_t)
{
   errno = ENOSYS;
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// ResetErrno; simply calls TSystem::ResetErrno().

void THDFSFile::ResetErrno() const
{
   TSystem::ResetErrno();
}


/**
\class THDFSSystem
\ingroup IO

Directory handler for HDFS (THDFSFile).
*/


ClassImp(THDFSSystem)

////////////////////////////////////////////////////////////////////////////////

THDFSSystem::THDFSSystem() : TSystem("-hdfs", "HDFS Helper System")
{
   SetName("hdfs");

   Bool_t has_authn = kTRUE;

   if (has_authn) {
      UserGroup_t *ugi = gSystem->GetUserInfo(0);
      const char *user = (ugi->fUser).Data();
      fFH = hdfsConnectAsUser("default", 0, user);
      delete ugi;
   } else {
      fFH = hdfsConnect("default", 0);
   }

   if (fFH == 0) {
      SysError("THDFSSystem", "HDFS client cannot open the filesystem");
      goto zombie;
   }

   fDirp = 0;

   return;

zombie:
   // Error in opening file; make this a zombie
   MakeZombie();
   gDirectory = gROOT;

}

////////////////////////////////////////////////////////////////////////////////
/// Make a directory.

Int_t THDFSSystem::MakeDirectory(const char * path)
{
   if (fFH != 0) {
      Error("MakeDirectory", "No filesystem handle (should never happen)");
      return -1;
   }

   if (R__HDFS_ALLOW_CHANGES == kTRUE) {
      return hdfsCreateDirectory((hdfsFS)fFH, path);
   } else {
      return -1;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Open a directory via hdfs. Returns an opaque pointer to a dir
/// structure. Returns 0 in case of error.

void *THDFSSystem::OpenDirectory(const char * path)
{
   if (fFH == 0) {
       Error("OpenDirectory", "No filesystem handle (should never happen)");
       return 0;
   }

   fDirp = 0;
/*
   if (fDirp) {
      Error("OpenDirectory", "invalid directory pointer (should never happen)");
      fDirp = 0;
   }
*/

   hdfsFileInfo * dir = 0;
   if ((dir = hdfsGetPathInfo((hdfsFS)fFH, path)) == 0) {
      return 0;
   }
   if (dir->mKind != kObjectKindDirectory) {
      return 0;
   }

   fDirp = (void *)hdfsListDirectory((hdfsFS)fFH, path, &fDirEntries);
   fDirCtr = 0;

   fUrlp = new TUrl[fDirEntries];

   return fDirp;
}

////////////////////////////////////////////////////////////////////////////////
/// Free directory via httpd.

void THDFSSystem::FreeDirectory(void *dirp)
{
   if (fFH == 0) {
      Error("FreeDirectory", "No filesystem handle (should never happen)");
      return;
   }
   if (dirp != fDirp) {
      Error("FreeDirectory", "invalid directory pointer (should never happen)");
      return;
   }
   if (fUrlp != 0) {
      delete fUrlp;
   }

   hdfsFreeFileInfo((hdfsFileInfo *)fDirp, fDirEntries);
   fDirp=0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get directory entry via httpd. Returns 0 in case no more entries.

const char *THDFSSystem::GetDirEntry(void *dirp)
{
   if (fFH == 0) {
      Error("GetDirEntry", "No filesystem handle (should never happen)");
      return 0;
   }
   if (dirp != fDirp) {
      Error("GetDirEntry", "invalid directory pointer (should never happen)");
      return 0;
   }
   if (dirp == 0) {
      Error("GetDirEntry", "Passed an invalid directory pointer.");
      return 0;
   }

   if (fDirCtr == fDirEntries-1) {
      return 0;
   }

   hdfsFileInfo *fileInfo = ((hdfsFileInfo *)dirp) + fDirCtr;
   fUrlp[fDirCtr].SetUrl(fileInfo->mName);
   const char * result = fUrlp[fDirCtr].GetFile();
   TUrl tempUrl;
   tempUrl.SetUrl("hdfs:///");
   tempUrl.SetFile(result);
   fUrlp[fDirCtr].SetUrl(tempUrl.GetUrl());
   result = fUrlp[fDirCtr].GetUrl();
   fDirCtr++;

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

Int_t THDFSSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   if (fFH == 0) {
      Error("GetPathInfo", "No filesystem handle (should never happen)");
      return 1;
   }
   hdfsFileInfo *fileInfo = hdfsGetPathInfo((hdfsFS)fFH, path);

   if (fileInfo == 0)
      return 1;

   buf.fDev    = 0;
   buf.fIno    = 0;
   buf.fMode   = fileInfo->mPermissions;
   buf.fUid    = gSystem->GetUid(fileInfo->mOwner);
   buf.fGid    = gSystem->GetGid(fileInfo->mGroup);
   buf.fSize   = fileInfo->mSize;
   buf.fMtime  = fileInfo->mLastAccess;
   buf.fIsLink = kFALSE;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns FALSE if one can access a file using the specified access mode.
/// Mode is the same as for the Unix access(2) function.
/// Attention, bizarre convention of return value!!

Bool_t THDFSSystem::AccessPathName(const char *path, EAccessMode mode)
{
   if (mode & kExecutePermission || mode & kWritePermission)
       return kTRUE;

   if (fFH == 0) {
      Error("AccessPathName", "No filesystem handle (should never happen)");
      return kTRUE;
   }

   if (hdfsExists((hdfsFS)fFH, path) == 0)
      return kFALSE;
   else
      return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlink, i.e. remove, a file or directory. Returns 0 when successful,
/// -1 in case of failure.

Int_t THDFSSystem::Unlink(const char * path)
{
   if (fFH == 0) {
      Error("Unlink", "No filesystem handle (should never happen)");
      return kTRUE;
   }

   if (R__HDFS_ALLOW_CHANGES == kTRUE) {
      return hdfsDelete((hdfsFS)fFH, path, 1);
   } else {
      return -1;
   }
}
