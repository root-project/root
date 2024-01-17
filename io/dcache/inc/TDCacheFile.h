// @(#)root/dcache:$Id$
// Author: Grzegorz Mazur   20/01/2002
// Updated: William Tanenbaum 21/11/2003
// Updated: Tgiran Mkrtchyan 28/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDCacheFile
#define ROOT_TDCacheFile

#include "TFile.h"
#include "TSystem.h"
#include "TString.h"

#include <sys/stat.h>

#define RAHEAD_BUFFER_SIZE 131072

class TDCacheFile : public TFile {

private:
   Bool_t fStatCached;       ///<! (transient) is file status cached?
   struct stat64 fStatBuffer;  ///<! (transient) Cached file status buffer (for performance)

   TDCacheFile() : fStatCached(kFALSE) { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode) override;
   Int_t    SysClose(Int_t fd) override;
   Int_t    SysRead(Int_t fd, void *buf, Int_t len) override;
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len) override;
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence) override;
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime) override;
   Int_t    SysSync(Int_t fd) override;

public:
   TDCacheFile(const char *path, Option_t *option="",
               const char *ftitle="", Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);

   ~TDCacheFile() override;

   Bool_t  ReadBuffer(char *buf, Int_t len) override;
   Bool_t  ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   Bool_t  WriteBuffer(const char *buf, Int_t len) override;

   Bool_t  ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf) override;

   void    ResetErrno() const override;

   static Bool_t Stage(const char *path, UInt_t secs,
                       const char *location = 0);
   static Bool_t CheckFile(const char *path, const char *location = 0);

   /// Note: This must be kept in sync with values \#defined in dcap.h
   enum EOnErrorAction {
      kOnErrorRetry   =  1,
      kOnErrorFail    =  0,
      kOnErrorDefault = -1
   };

   static void SetOpenTimeout(UInt_t secs);
   static void SetOnError(EOnErrorAction = kOnErrorDefault);

   static void SetReplyHostName(const char *host_name);
   static const char *GetDcapVersion();
   static TString GetDcapPath(const char *path);


   ClassDefOverride(TDCacheFile,1)  //A ROOT file that reads/writes via a dCache server
};


class TDCacheSystem : public TSystem {

private:
   void    *fDirp;   ///< directory handler

   void    *GetDirPtr() const override { return fDirp; }

public:
   TDCacheSystem();
   ~TDCacheSystem() override { }

   Int_t       MakeDirectory(const char *name) override;
   void       *OpenDirectory(const char *name) override;
   void        FreeDirectory(void *dirp) override;
   const char *GetDirEntry(void *dirp) override;
   Int_t       GetPathInfo(const char *path, FileStat_t &buf) override;
   Bool_t      AccessPathName(const char *path, EAccessMode mode) override;

   ClassDefOverride(TDCacheSystem,0)  // Directory handler for DCache
};

#endif
