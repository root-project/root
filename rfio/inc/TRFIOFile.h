// @(#)root/rfio:$Name:  $:$Id: TRFIOFile.h,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
// Author: Fons Rademakers   20/01/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRFIOFile
#define ROOT_TRFIOFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRFIOFile                                                            //
//                                                                      //
// A TRFIOFile is like a normal TFile except that it reads and writes   //
// its data via a rfiod server.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif


class TRFIOFile : public TFile {

private:
   TUrl      fUrl;        //URL of file
   Seek_t    fOffset;     //seek offet

   TRFIOFile() : fUrl("dummy") { }

   // Interface to basic system I/O routines
   Int_t  SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t  SysClose(Int_t fd);
   Int_t  SysRead(Int_t fd, void *buf, Int_t len);
   Int_t  SysWrite(Int_t fd, const void *buf, Int_t len);
   Seek_t SysSeek(Int_t fd, Seek_t offset, Int_t whence);
   Int_t  SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   Int_t  SysSync(Int_t) { /* no fsync for RFIO */ return 0; }

public:
   TRFIOFile(const char *url, Option_t *option="", const Text_t *ftitle="", Int_t compress=1);
   ~TRFIOFile();

   ClassDef(TRFIOFile,1)  //A ROOT file that reads/writes via a rfiod server
};

#endif
