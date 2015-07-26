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

Long64_t TMemFile::fgDefaultBlockSize = 2*1024*1024;

//______________________________________________________________________________
TMemFile::TMemBlock::TMemBlock() : fPrevious(0), fNext(0), fBuffer(0), fSize(0)
{
   // Default constructor
}

//______________________________________________________________________________
TMemFile::TMemBlock::TMemBlock(Long64_t size, TMemBlock *previous) :
   fPrevious(previous), fNext(0), fBuffer(0), fSize(0)
{
   // Constructor allocating the memory buffer.

   fBuffer = new UChar_t[size];
   fSize = size;
}

//______________________________________________________________________________
TMemFile::TMemBlock::~TMemBlock()
{
   // Usual destructors.  Delete the block memory.

   delete fNext;
   delete [] fBuffer;
}

//______________________________________________________________________________
void TMemFile::TMemBlock::CreateNext(Long64_t size)
{
   R__ASSERT(fNext == 0);
   fNext = new TMemBlock(size,this);
}

//______________________________________________________________________________
TMemFile::TMemFile(const char *path, Option_t *option,
                   const char *ftitle, Int_t compress) :
   TFile(path, "WEB", ftitle, compress),
   fSize(-1), fSysOffset(0), fBlockSeek(&fBlockList), fBlockOffset(0)
{
   // Usual Constructor.  See the TFile constructor for details.

   fOption = option;
   fOption.ToUpper();
   if (fOption == "NEW")  fOption = "CREATE";
   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   if (!(create || recreate)) {
      Error("TMemFile","Reading a TMemFile requires a memory buffer\n");
      goto zombie;
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
   TFile(path, "WEB", ftitle, compress), fBlockList(size),
   fSize(size), fSysOffset(0), fBlockSeek(&(fBlockList)), fBlockOffset(0)
{
   // Usual Constructor.  See the TFile constructor for details.

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

   SysWrite(fD,buffer,size);

   Init(create || recreate);
   return;

zombie:
   // Error in opening file; make this a zombie
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
TMemFile::TMemFile(const TMemFile &orig) :
   TFile(orig.GetEndpointUrl()->GetUrl(), "WEB", orig.GetTitle(),
         orig.GetCompressionSettings() ), fBlockList(orig.GetEND()),
   fSize(orig.GetEND()), fSysOffset(0), fBlockSeek(&(fBlockList)), fBlockOffset(0)
{
   // Copying the content of the TMemFile into another TMemFile.

   fOption = orig.fOption;

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   fD = orig.fD; // not really used, so it is okay to have the same value.
   fWritable = orig.fWritable;

   // We intentionally allocated just one big buffer for this object.
   orig.CopyTo(fBlockList.fBuffer,fSize);

   Init(create || recreate); // A copy is
}


//______________________________________________________________________________
TMemFile::~TMemFile()
{
   // Close and clean-up HDFS file.

   // Need to call close, now as it will need both our virtual table
   // and the content of the list of blocks
   Close();
   TRACE("destroy")
}

//______________________________________________________________________________
Long64_t TMemFile::CopyTo(void *to, Long64_t maxsize) const
{
   // Copy the binary representation of the TMemFile into
   // the memory area starting at 'to' and of length at most 'maxsize'
   // returns the number of bytes actually copied.

   Long64_t len = GetSize();
   if (len > maxsize) {
      len = maxsize;
   }
   Long64_t storedSysOffset   = fSysOffset;
   Long64_t storedBlockOffset = fBlockOffset;
   TMemBlock *storedBlockSeek = fBlockSeek;

   const_cast<TMemFile*>(this)->SysSeek(fD, 0, SEEK_SET);
   len = const_cast<TMemFile*>(this)->SysRead(fD, to, len);

   const_cast<TMemFile*>(this)->fBlockSeek   = storedBlockSeek;
   const_cast<TMemFile*>(this)->fBlockOffset = storedBlockOffset;
   const_cast<TMemFile*>(this)->fSysOffset   = storedSysOffset;
   return len;
}

//______________________________________________________________________________
void TMemFile::CopyTo(TBuffer &tobuf) const
{
   // Copy the binary representation of the TMemFile into
   // the TBuffer tobuf

   const TMemBlock *current = &fBlockList;
   while(current) {
      tobuf.WriteFastArray(current->fBuffer,current->fSize);
      current = current->fNext;
   }
}

//______________________________________________________________________________
Long64_t TMemFile::GetSize() const
{
   // Return the current size of the memory file

   // We could also attempt to read it from the beginning of the buffer
   return fSize;
}

//______________________________________________________________________________
void TMemFile::Print(Option_t *option /* = "" */) const
{
   Printf("TMemFile: name=%s, title=%s, option=%s", GetName(), GetTitle(), GetOption());
   if (strcmp(option,"blocks")==0) {
      const TMemBlock *current = &fBlockList;
      Int_t counter = 0;
      while(current) {
         Printf("TMemBlock: %d size=%lld addr=%p curr=%p prev=%p next=%p",
                counter,current->fSize,current->fBuffer,
                current,current->fPrevious,current->fNext);
         current = current->fNext;
         ++counter;
      }
   } else {
      GetList()->R__FOR_EACH(TObject,Print)(option);
   }
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

   fSysOffset   = 0;
   fBlockSeek   = &fBlockList;
   fBlockOffset = 0;
   {
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetListOfFiles()->Remove(this);
   }

   {
      TDirectory::TContext ctxt(this);
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
//         Error("ResetObjects","In %s, we can not reset %s (not ResetAfterMerge function)",
//               directory->GetName(),idcur->GetName());
      }
   }
}

//______________________________________________________________________________
Int_t TMemFile::SysRead(Int_t, void *buf, Int_t len)
{
   // Read specified number of bytes from current offset into the buffer.
   // See documentation for TFile::SysRead().

   TRACE("READ")

   if (fBlockList.fBuffer == 0) {
      errno = EBADF;
      gSystem->SetErrorStr("The memory file is not open.");
      return 0;
   } else {
      // Don't read past the end.
      if (fSysOffset + len > fSize) {
         len = fSize - fSysOffset;
      }

      if (fBlockOffset+len <= fBlockSeek->fSize) {
         // 'len' does not go past the end of the current block,
         // so let's make a simple copy.
         memcpy(buf,fBlockSeek->fBuffer+fBlockOffset,len);
         fBlockOffset += len;
      } else {
         // We are going to have to copy data from more than one
         // block.

         // First copy the end of the first block.
         Int_t sublen = fBlockSeek->fSize - fBlockOffset;
         memcpy(buf,fBlockSeek->fBuffer+fBlockOffset,sublen);

         // Move to the next.
         buf = (char*)buf + sublen;
         Int_t len_left = len - sublen;
         fBlockSeek = fBlockSeek->fNext;

         // Copy all the full blocks that are covered by the request.
         while (len_left > fBlockSeek->fSize) {
            R__ASSERT(fBlockSeek);

            memcpy(buf, fBlockSeek->fBuffer, fBlockSeek->fSize);
            buf = (char*)buf + fBlockSeek->fSize;
            len_left -= fBlockSeek->fSize;
            fBlockSeek = fBlockSeek->fNext;
         }

         // Copy the data from the last block.
         R__ASSERT(fBlockSeek);
         memcpy(buf,fBlockSeek->fBuffer, len_left);
         fBlockOffset = len_left;

      }
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
   if (whence == SEEK_SET) {
      fSysOffset = offset;
      fBlockSeek = &fBlockList;
      Long64_t counter = 0;
      while(fBlockSeek->fNext && (counter+fBlockSeek->fSize) < fSysOffset)
      {
         counter += fBlockSeek->fSize;
         fBlockSeek = fBlockSeek->fNext;
      }
      fBlockOffset = fSysOffset - counter;  // If we seek past the 'end' of the file, we now have fBlockOffset > fBlockSeek->fSize
   } else if (whence == SEEK_CUR) {

      if (offset == 0) {
         // nothing to do, really
      } else if (offset > 0) {
         // Move forward.
         if ( (fBlockOffset+offset) < fBlockSeek->fSize) {
            fSysOffset += offset;
            fBlockOffset += offset;
         } else {
            Long64_t counter = fSysOffset;
            fSysOffset += offset;
            while(fBlockSeek->fNext && counter < fSysOffset)
            {
               counter += fBlockSeek->fSize;
               fBlockSeek = fBlockSeek->fNext;
            }
            fBlockOffset = fSysOffset - counter; // If we seek past the 'end' of the file, we now have fBlockOffset > fBlockSeek->fSize
         }
      } else {
         // Move backward in the file (offset < 0).
         Long64_t counter = fSysOffset;
         fSysOffset += offset;
         if (fSysOffset < 0) {
            SysError("TMemFile", "Unable to seek past the beginning of file");
            fSysOffset   = 0;
            fBlockSeek   = &fBlockList;
            fBlockOffset = 0;
            return -1;
         } else {
            if (offset+fBlockOffset >= 0) {
               // We are just moving in the current block.
               fBlockOffset += offset;
            } else {
               while(fBlockSeek->fPrevious && counter > fSysOffset)
               {
                  counter -= fBlockSeek->fSize;
                  fBlockSeek = fBlockSeek->fPrevious;
               }
               fBlockOffset = fSysOffset - counter;
            }
         }
      }
   } else if (whence == SEEK_END) {
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

   if (!fBlockList.fBuffer) {
      fBlockList.fBuffer = new UChar_t[fgDefaultBlockSize];
      fBlockList.fSize = fgDefaultBlockSize;
      fSize = fgDefaultBlockSize;
   }
   if (fBlockList.fBuffer) {
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

   TRACE("WRITE")

   if (fBlockList.fBuffer == 0) {
      errno = EBADF;
      gSystem->SetErrorStr("The memory file is not open.");
      return 0;
   } else {
      if (fBlockOffset+len <= fBlockSeek->fSize) {
         // 'len' does not go past the end of the current block,
         // so let's make a simple copy.
         memcpy(fBlockSeek->fBuffer+fBlockOffset,buf,len);
         fBlockOffset += len;
      } else {
         // We are going to have to copy data into more than one
         // block.

         // First copy to the end of the first block.
         Int_t sublen = fBlockSeek->fSize - fBlockOffset;
         memcpy(fBlockSeek->fBuffer+fBlockOffset,buf,sublen);

         // Move to the next.
         buf = (char*)buf + sublen;
         Int_t len_left = len - sublen;
         if (!fBlockSeek->fNext) {
            fBlockSeek->CreateNext(fgDefaultBlockSize);
            fSize += fgDefaultBlockSize;
         }
         fBlockSeek = fBlockSeek->fNext;

         // Copy all the full blocks that are covered by the request.
         while (len_left > fBlockSeek->fSize) {
            R__ASSERT(fBlockSeek);

            memcpy(fBlockSeek->fBuffer, buf, fBlockSeek->fSize);
            buf = (char*)buf + fBlockSeek->fSize;
            len_left -= fBlockSeek->fSize;
            if (!fBlockSeek->fNext) {
               fBlockSeek->CreateNext(fgDefaultBlockSize);
               fSize += fgDefaultBlockSize;
            }
            fBlockSeek = fBlockSeek->fNext;
         }

         // Copy the data from the last block.
         R__ASSERT(fBlockSeek);
         memcpy(fBlockSeek->fBuffer, buf, len_left);
         fBlockOffset = len_left;

      }
      fSysOffset += len;
      return len;
   }
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
