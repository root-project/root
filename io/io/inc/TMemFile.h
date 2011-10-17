// @(#)root/io:$Id$
// Author: Philippe Canal, May 2011

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemFile
#define ROOT_TMemFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemFile                                                             //
//                                                                      //
// A TMemFile is like a normal TFile except that it reads and writes    //
// its data via in memory.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif

class TMemFile : public TFile {

private:
   struct TMemBlock {
      TMemBlock();
      TMemBlock(Long64_t size, TMemBlock *previous = 0);
      ~TMemBlock();

      void CreateNext(Long64_t size);
      
      TMemBlock *fPrevious;
      TMemBlock *fNext;
      UChar_t   *fBuffer;
      Long64_t   fSize;
   };
   TMemBlock    fBlockList;   // Colletion of memory blocks of size fBlockSize
   Long64_t     fSize;        // Total file size (sum of the size of the chunks)
   Long64_t     fSysOffset;   // Seek offset in file
   TMemBlock   *fBlockSeek;   // Pointer to the block we seeked to.
   Long64_t     fBlockOffset; // Seek offset within the block

   static Long64_t fgDefaultBlockSize;

   Long64_t MemRead(Int_t fd, void *buf, Long64_t len) const;

   // Overload TFile interfaces.
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t fd);

   void ResetObjects(TDirectoryFile *, TFileMergeInfo *) const;

public:
   TMemFile(const char *name, Option_t *option="", const char *ftitle="", Int_t compress=1);
   TMemFile(const char *name, char *buffer, Long64_t size, Option_t *option="", const char *ftitle="", Int_t compress=1);
   TMemFile(const TMemFile &orig);
   virtual ~TMemFile();

   virtual Long64_t CopyTo(void *to, Long64_t maxsize) const;
   virtual void     CopyTo(TBuffer &tobuf) const;
   virtual Long64_t GetSize() const;

   void ResetAfterMerge(TFileMergeInfo *);
   void ResetErrno() const;

   virtual void        Print(Option_t *option="") const;
   
   ClassDef(TMemFile, 0) //A ROOT file that reads/writes via HDFS
};

#endif
