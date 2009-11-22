// @(#)root/hdfs:$Id$
// Author: Brian Bockelman 29/09/2009

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THDFSFile                                                            //
//                                                                      //
// A THDFSFile is like a normal TFile except that it reads and writes   //
// its data via the HDFS protocols.  For more information, see          //
// http://hadoop.apache.org/hdfs/.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "syslog.h"
#include "assert.h"

#include "THDFSFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"

#include "hdfs.h"
//#include "hdfsJniHelper.h"

static const char* const R__HDFS_PREFIX = "hdfs://";
static const size_t R__HDFS_PREFIX_LEN = 7;

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

//______________________________________________________________________________
THDFSFile::THDFSFile(const char *path, Option_t *option,
                     const char *ftitle, Int_t compress):
   TFile(path, "NET", ftitle, compress)
{
   // Usual Constructor.  See the TFile constructor for details.

   fHdfsFH = 0;
   fFS = 0;
   fSize = -1;
   fPath = 0;

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
      const char * groups[1] = {(ugi->fGroup.Data())};
      fFS = hdfsConnectAsUser("default", 0, user, groups, 1);
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

//______________________________________________________________________________
THDFSFile::~THDFSFile()
{
   // Close and clean-up HDFS file.

   TRACE("destroy")

   if (0 != fPath)
      free(fPath);

   // We assume that the file is closed in SysClose
/*
   if (0 != fHdfsFH && 0 != fFS) {
      if (hdfsCloseFile(fFS, (hdfsFile)fHdfsFH) != 0) {
          SysError("THDFSFile", "Error closing file %s", fPath);
      }
   }
*/
   // Explicitly release reference to HDFS filesystem object.
   // Turned off now due to compilation issues.
/*
   if (0 != fFS) {
      JNIEnv* env = getJNIEnv();
      if (env == 0) {
          SysError("THDFSFile", "Internal error; cannot get JNI env");
      } else {
         env->DeleteGlobalRef((jobject)fFS);
      }
   }
*/
}

//______________________________________________________________________________
Bool_t THDFSFile::WriteBuffer(const char *, Int_t)
{
   // Write specified byte range to remote file via HDFS
   // We do not support writes, so an error is always returned.

   SysError("THDFSFile", "Writes are not supported by THDFSFile");
   return kTRUE;
}

//______________________________________________________________________________
Int_t THDFSFile::SysRead(Int_t, void *buf, Int_t len)
{
   // Read specified number of bytes from current offset into the buffer.
   // See documentation for TFile::SysRead().

   TRACE("READ")
   tSize num_read = hdfsPread(fFS, (hdfsFile)fHdfsFH, fOffset, buf, len);
   fOffset += len;
   if (num_read < 0) {
      gSystem->SetErrorStr(strerror(errno));
   }
   return num_read;
}

//______________________________________________________________________________
Long64_t THDFSFile::SysSeek(Int_t, Long64_t offset, Int_t whence)
{
   // Seek to a specified position in the file.  See TFile::SysSeek.
   // Note that THDFSFile does not support seeks when the file is open for write.

   TRACE("SEEK")
   if (whence == SEEK_SET)
      fOffset = offset;
   else if (whence == SEEK_CUR)
      fOffset += offset;
   else if (whence == SEEK_END) {
      if (offset > 0) {
         SysError("THDFSFile", "Unable to seek past end of file");
         return -1;
      }
      if (fSize == -1) {
         hdfsFileInfo *info = hdfsGetPathInfo(fFS, fPath);
         if (info != 0) {
            fSize = info->mSize;
            free(info);
         } else {
            SysError("THDFSFile", "Unable to seek to end of file");
            return -1;
         }
      }
      fOffset = fSize;
   } else {
      SysError("THDFSFile", "Unknown whence!");
      return -1;
   }
   return fOffset;
}

//______________________________________________________________________________
Int_t THDFSFile::SysOpen(const char * pathname, Int_t flags, UInt_t)
{
   // Open a file in HDFS.

   Long64_t path_len = strlen(pathname + R__HDFS_PREFIX_LEN);
   fPath = (char*)malloc((path_len+1)*sizeof(char));
   strcpy(fPath, pathname + R__HDFS_PREFIX_LEN);
   if ((fHdfsFH = hdfsOpenFile(fFS, pathname + R__HDFS_PREFIX_LEN, flags, 0, 0, 0)) == 0) {
      SysError("THDFSFile", "Unable to open file %s in HDFS", pathname + R__HDFS_PREFIX_LEN);
      return -1;
   }
   return 1;
}

//______________________________________________________________________________
Int_t THDFSFile::SysClose(Int_t)
{
   // Close the file in HDFS.

   int result = hdfsCloseFile(fFS, (hdfsFile)fHdfsFH);
   fFS = 0;
   fHdfsFH = 0;
   return result;
}

//______________________________________________________________________________
Int_t THDFSFile::SysWrite(Int_t, const void *, Int_t)
{
   // Write a buffer into the file; this is not supported currently.

   errno = ENOSYS;
   return -1;
}

//______________________________________________________________________________
Int_t THDFSFile::SysStat(Int_t, Long_t* id, Long64_t* size, Long_t* flags, Long_t* modtime)
{
   // Perform a stat on the HDFS file; see TFile::SysStat().

   *id = ::Hash(fPath);

   hdfsFileInfo *info = hdfsGetPathInfo(fFS, fPath);
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

//______________________________________________________________________________
Int_t THDFSFile::SysSync(Int_t)
{
   // Sync remaining data to disk; Not supported by HDFS.

   errno = ENOSYS;
   return -1;
}

//______________________________________________________________________________
void THDFSFile::ResetErrno() const
{
   // ResetErrno; simply calls TSystem::ResetErrno().

   TSystem::ResetErrno();
}
