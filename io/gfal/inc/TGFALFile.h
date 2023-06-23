// @(#)root/gfal:$Id$
// Author: Fons Rademakers   8/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFALFile
#define ROOT_TGFALFile

#include "TFile.h"
#include "TSystem.h"


class TGFALFile : public TFile {

private:
   Bool_t        fStatCached;  ///<! (transient) is file status cached?
   struct stat64 fStatBuffer;  ///<! (transient) Cached file status buffer (for performance)

   TGFALFile() : fStatCached(kFALSE) { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode) override;
   Int_t    SysClose(Int_t fd) override;
   Int_t    SysRead(Int_t fd, void *buf, Int_t len) override;
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len) override;
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence) override;
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime) override;
   Int_t    SysSync(Int_t) override { /* no fsync for GFAL */ return 0; }

public:
   TGFALFile(const char *url, Option_t *option="",
             const char *ftitle="", Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);
   ~TGFALFile() override;

   Bool_t  ReadBuffer(char *buf, Int_t len) override;
   Bool_t  ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   Bool_t  WriteBuffer(const char *buf, Int_t len) override;

   ClassDefOverride(TGFALFile,1)  //A ROOT file that reads/writes via a GFAL
};


class TGFALSystem : public TSystem {

private:
   void    *fDirp;   // directory handler

   void    *GetDirPtr() const override { return fDirp; }

public:
   TGFALSystem();
   ~TGFALSystem() override { }

   Int_t       MakeDirectory(const char *name) override;
   void       *OpenDirectory(const char *name) override;
   void        FreeDirectory(void *dirp) override;
   const char *GetDirEntry(void *dirp) override;
   Int_t       GetPathInfo(const char *path, FileStat_t &buf) override;
   Bool_t      AccessPathName(const char *path, EAccessMode mode) override;

   ClassDefOverride(TGFALSystem,0)  // Directory handler for GFAL
};

#endif
