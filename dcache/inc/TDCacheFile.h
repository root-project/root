// @(#)root/dcache:$Name:  $:$Id: TDCacheFile.h,v 1.1 2002/01/27 17:21:22 rdm Exp $
// Author: Grzegorz Mazur   20/01/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDCacheFile
#define ROOT_TDCacheFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDCacheFile                                                          //
//                                                                      //
// A TDCacheFile is like a normal TFile except that it reads and writes //
// its data via a dCache server.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TFile
#include "TFile.h"
#endif

class TDCacheFile : public TFile {

private:
   Seek_t fOffset;

   TDCacheFile() : fOffset(0) { }

   // Interface to basic system I/O routines
   Int_t  SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t  SysClose(Int_t fd);
   Int_t  SysRead(Int_t fd, void *buf, Int_t len);
   Int_t  SysWrite(Int_t fd, const void *buf, Int_t len);
   Seek_t SysSeek(Int_t fd, Seek_t offset, Int_t whence);
   Int_t  SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   Int_t  SysSync(Int_t fd);

public:
   TDCacheFile(const char *path, Option_t *option="",
               const char *ftitle="", Int_t compress=1);

   ~TDCacheFile();

   Bool_t  ReadBuffer(char *buf, Int_t len);
   Bool_t  WriteBuffer(const char *buf, Int_t len);

   void    ResetErrno() const;

   static Bool_t Stage(const char *path, UInt_t secs,
                       const char *location = 0);
   static Bool_t CheckFile(const char *path, const char *location = 0);

   // Note: This must be kept in sync with values #defined in dcap.h
   enum OnErrorAction {
      kOnErrorRetry   =  1,
      kOnErrorFail    =  0,
      kOnErrorDefault = -1
   };

   static void SetOpenTimeout(UInt_t secs);
   static void SetOnError(OnErrorAction = kOnErrorDefault);

   static void SetReplyHostName(const char *host_name);
   static const char *GetDcapVersion();

   static Bool_t EnableSSL();

   ClassDef(TDCacheFile,1)  //A ROOT file that reads/writes via a dCache server
};

#endif
