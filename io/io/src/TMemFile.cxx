// @(#)root/io:$Id$
// Author: Philippe Canal, May 2011

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemFile                                                             //
//                                                                      //
// A TMemFile is like a normal TFile except that it reads and writes    //
// only from memory.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMemFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TArrayC.h"
#include "TKey.h"
#include "TClass.h"
#include "TVirtualMutex.h"
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

// The following snippet is used for developer-level debugging
#define TMemFile_TRACE
#ifndef TMemFile_TRACE
#define TRACE(x) \
  Debug("TMemFile", "%s", x);
#else
#define TRACE(x);
#endif

ClassImp(TMemFile)

Long64_t TMemFile::fgDefaultBlockSize = 32*1024*1024;

//______________________________________________________________________________
TMemFile::TMemFile(const char *path, Option_t *option,
                   const char *ftitle, Int_t compress):
   TFile(path, "WEB", ftitle, compress), fBuffer(0)
{
   // Usual Constructor.  See the TFile constructor for details.

   fSize      = -1;
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

   if (create || update || recreate) {
      Int_t mode = O_RDWR | O_CREAT;
      if (recreate) mode |= O_TRUNC;

      fD = SysOpen(path, O_RDWR | O_CREAT, 0644);
      if (fD == -1) {
         SysError("TMemFile", "file %s can not be opened", path);
         goto zombie;
      }
      fWritable = kTRUE;
   } else {
      fD = SysOpen(path, O_RDONLY, 0644);
      if (fD == -1) {
         SysError("TMemFile", "file %s can not be opened for reading", path);
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
TMemFile::TMemFile(const char *path, char *buffer, Long64_t size, Option_t *option,
                   const char *ftitle, Int_t compress):
TFile(path, "WEB", ftitle, compress), fBuffer(0)
{
   // Usual Constructor.  See the TFile constructor for details.

   fSize      = -1;
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

   if (create || update || recreate) {
      Int_t mode = O_RDWR | O_CREAT;
      if (recreate) mode |= O_TRUNC;

      fD = SysOpen(path, O_RDWR | O_CREAT, 0644);
      if (fD == -1) {
         SysError("TMemFile", "file %s can not be opened", path);
         goto zombie;
      }
      fWritable = kTRUE;

   } else {
      fD = SysOpen(path, O_RDONLY, 0644);
      if (fD == -1) {
         SysError("TMemFile", "file %s can not be opened for reading", path);
         goto zombie;
      }
      fWritable = kFALSE;
   }

   memcpy(fBuffer,buffer,size);

   Init(create || recreate);

   return;

zombie:
   // Error in opening file; make this a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TMemFile::~TMemFile()
{
   // Close and clean-up HDFS file.

   // Need to call close, now as it will need both our virtual table
   // and the content of fBuffer
   Close();
   delete [] fBuffer;  fBuffer = 0;
   TRACE("destroy")
}

//______________________________________________________________________________
Long64_t TMemFile::GetSize() const
{
   // Return the current size of the memory file

   // We could also attempt to read it from the beginning of the buffer
   return fSize;
}

//______________________________________________________________________________
void TMemFile::ResetAfterMerge(TFileMergeInfo *info)
{
   // Wipe all the data from the permanent buffer but keep, the in-memory object
   // alive.

   ResetObjects(this,info);

   fNbytesKeys = 0;
   fSeekKeys = 0;

   fMustFlush = kTRUE;
   fInitDone = kFALSE;

   if (fFree) {
      fFree->Delete();
      delete fFree;
      fFree      = 0;
   }
   fWritten      = 0;
   fSumBuffer    = 0;
   fSum2Buffer   = 0;
   fBytesRead    = 0;
   fBytesReadExtra = 0;
   fBytesWrite   = 0;
   delete fClassIndex;
   fClassIndex   = 0;
   fSeekInfo     = 0;
   fNbytesInfo   = 0;
   delete fProcessIDs;
   fProcessIDs   = 0;
   fNProcessIDs  = 0;
   fOffset       = 0;
   fCacheRead    = 0;
   fCacheWrite   = 0;
   fReadCalls    = 0;
   if (fFree) {
      fFree->Delete();
      delete fFree;
      fFree = 0;
   }

   fSysOffset    = 0;
   
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfFiles()->Remove(this);
   }

   {
      TDirectory::TContext ctxt(gDirectory, this);
      Init(kTRUE);

      // And now we need re-initilize the directories ...

      TIter   next(this->GetList());
      TObject *idcur;
      while ((idcur = next())) {
         if (idcur->IsA() == TDirectoryFile::Class()) {
            ((TDirectoryFile*)idcur)->ResetAfterMerge(info);
         }
      }

   }
}

//______________________________________________________________________________
void TMemFile::ResetObjects(TDirectoryFile *directory, TFileMergeInfo *info) const
{
   // Wipe all the data from the permanent buffer but keep, the in-memory object
   // alive.

   if (directory->GetListOfKeys()) {
      TIter next(directory->GetListOfKeys());
      TKey *key;
      while( (key = (TKey*)next()) ) {
         if (0 ==  directory->GetList()->FindObject(key->GetName())) {
            Warning("ResetObjects","Key/Object %s is not attached to the directory %s and can not be ResetAfterMerge correctly",
                    key->GetName(),directory->GetName());
         }
      }
      directory->GetListOfKeys()->Delete("slow");
   }

   TString listHargs;
   listHargs.Form("(TFileMergeInfo*)0x%lx",(ULong_t)info);

   TIter   next(directory->GetList());
   TObject *idcur;
   while ((idcur = next())) {
      TClass *objcl = idcur->IsA();
      if (objcl == TDirectoryFile::Class()) {
         ResetObjects((TDirectoryFile*)idcur,info);
      } else if (objcl->GetResetAfterMerge()) {
         (objcl->GetResetAfterMerge())(idcur,info);
      } else if (idcur->IsA()->GetMethodWithPrototype("ResetAfterMerge", "TFileMergeInfo*") ) {
         Int_t error = 0;
         idcur->Execute("ResetAfterMerge", listHargs.Data(), &error);
         if (error) {
            Error("ResetObjects", "calling ResetAfterMerge() on '%s' failed.",
                  idcur->GetName());
         }
      } else {
         Error("ResetObjects","In %s, we can not reset %s (not ResetAfterMerge function)",
               directory->GetName(),idcur->GetName());
      }
   }
}

//______________________________________________________________________________
Int_t TMemFile::SysRead(Int_t, void *buf, Int_t len)
{
   // Read specified number of bytes from current offset into the buffer.
   // See documentation for TFile::SysRead().

   TRACE("READ")

   if (fBuffer == 0) {
      errno = EBADF;
      gSystem->SetErrorStr("The memory file is not open.");
      return 0;
   } else {
      if (fSysOffset + len > fSize) {
         len = fSize - fSysOffset;
      }
      memcpy(buf,fBuffer+fSysOffset,len);
      fSysOffset += len;
      return len;
   }
}

//______________________________________________________________________________
Long64_t TMemFile::SysSeek(Int_t, Long64_t offset, Int_t whence)
{
   // Seek to a specified position in the file.  See TFile::SysSeek().
   // Note that TMemFile does not support seeks when the file is open for write.

   TRACE("SEEK")
   if (whence == SEEK_SET)
      fSysOffset = offset;
   else if (whence == SEEK_CUR)
      fSysOffset += offset;
   else if (whence == SEEK_END) {
      if (offset > 0) {
         SysError("TMemFile", "Unable to seek past end of file");
         return -1;
      }
      if (fSize == -1) {
         SysError("TMemFile", "Unable to seek to end of file");
         return -1;
      }
      fSysOffset = fSize;
   } else {
      SysError("TMemFile", "Unknown whence!");
      return -1;
   }
   return fSysOffset;
}

//______________________________________________________________________________
Int_t TMemFile::SysOpen(const char * /* pathname */, Int_t /* flags */, UInt_t /* mode */)
{
   // Open a file in 'MemFile'.

   if (!fBuffer) {
      fBuffer = new UChar_t[fgDefaultBlockSize];
      fSize = fgDefaultBlockSize;
   }
   if (fBuffer) {
      return 0;
   } else {
      return -1;
   }
}

//______________________________________________________________________________
Int_t TMemFile::SysClose(Int_t /* fd */)
{
   // Close the mem file.

   return 0;
}

//______________________________________________________________________________
Int_t TMemFile::SysWrite(Int_t /* fd */, const void *buf, Int_t len)
{
   // Write a buffer into the file;

   if (fSysOffset + len > fSize) {
      len = fSize - fSysOffset;
   }
   memcpy(fBuffer+fSysOffset,buf,len);
   fSysOffset += len;
   return len;
}

//______________________________________________________________________________
Int_t TMemFile::SysStat(Int_t, Long_t* /* id */, Long64_t* /* size */, Long_t* /* flags */, Long_t* /* modtime */)
{
   // Perform a stat on the HDFS file; see TFile::SysStat().

   MayNotUse("SysStat");
   return 0;
}

//______________________________________________________________________________
Int_t TMemFile::SysSync(Int_t)
{
   // Sync remaining data to disk;
   // Nothing to do here.

   return 0;
}

//______________________________________________________________________________
void TMemFile::ResetErrno() const
{
   // ResetErrno; simply calls TSystem::ResetErrno().

   TSystem::ResetErrno();
}
