// @(#)root/hdfs:$Id$
// Author: Brian Bockelman 29/09/2009

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THDFSFile
#define ROOT_THDFSFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THDFSFile                                                            //
//                                                                      //
// A THDFSFile is like a normal TFile except that it reads and writes   //
// its data via the HDFS protocols.  For more information, see          //
// http://hadoop.apache.org/hdfs/.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif

class THDFSFile : public TFile {

private:
   void* fHdfsFH;
   void* fFS;
   Long64_t fSize;
   char * fPath;

   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t fd);

public:
   THDFSFile(const char *path, Option_t *option="",
             const char *ftitle="", Int_t compress=1);

   ~THDFSFile();

   Bool_t  WriteBuffer(const char *buf, Int_t len);

   void ResetErrno() const;

   ClassDef(THDFSFile, 0) //A ROOT file that reads/writes via HDFS
};

#endif
