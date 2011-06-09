// @(#)root/hdfs:$Id$
// Author: Brian Bockelman 29/09/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemFile
#define ROOT_TMem

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
   UChar_t  *fBuffer;    // Pointer to a buffer of size 'fSize'.
   Long64_t  fSize;      // file size
   Long64_t  fSysOffset; // Seek offset in file

   static Long64_t fgDefaultBlockSize;

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
   virtual ~TMemFile();

   virtual Long64_t    GetSize() const;
   
   UChar_t *GetBuffer() { return fBuffer; }
   void ResetAfterMerge(TFileMergeInfo *);
   void ResetErrno() const;

   ClassDef(TMemFile, 0) //A ROOT file that reads/writes via HDFS
};



#endif
